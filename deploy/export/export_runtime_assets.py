#!/usr/bin/env python3
"""Export immutable tokenizer/conditioning tables for the C++ runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fairseq2.assets import asset_store

from seamless_communication.models.unity import load_unity_text_tokenizer
from seamless_communication.models.unity.char_tokenizer import (
    load_unity_char_tokenizer,
)
from seamless_communication.models.vocoder.loader import load_vocoder_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--model-name", default="seamless_streaming_unity")
    parser.add_argument("--vocoder-name", default="vocoder_v2")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_properties(
    text_tokenizer: Any, char_tokenizer: Any, index: int
) -> tuple[str, int, bool, bool, list[int]]:
    piece = str(text_tokenizer.model.index_to_token(index))
    characters = list(piece)
    starts_space = len(characters) > 1 and characters[0] == "▁"
    punctuation = (
        len(characters) == 1
        and not characters[0].isalpha()
        and not characters[0].isnumeric()
        and characters[0] != "▁"
    )
    if index == text_tokenizer.vocab_info.unk_idx:
        character_ids = [int(char_tokenizer.vocab_info.unk_idx)]
        codepoint_count = 1
    else:
        character_ids = [
            int(char_tokenizer.model.token_to_index(character))
            for character in characters
        ]
        codepoint_count = len(characters)
    return piece, codepoint_count, starts_space, punctuation, character_ids


def build_inputs_from_exported_properties(
    text_seqs: np.ndarray,
    properties: list[tuple[str, int, bool, bool, list[int]]],
    pad_idx: int,
    unk_idx: int,
    eos_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    if text_seqs.shape[0] != 1:
        raise ValueError("Runtime asset validation requires batch size 1")
    content = text_seqs[0, 2:].copy()
    content[content == eos_idx] = pad_idx
    subword_length = int(np.count_nonzero(content != pad_idx))
    content = content[:subword_length]
    records = [properties[int(index)] for index in content]

    char_lens = np.zeros(text_seqs.shape, dtype=text_seqs.dtype)
    char_ids: list[int] = []
    for index, (token_index, record) in enumerate(zip(content, records)):
        _, codepoint_count, starts_space, punctuation, record_char_ids = record
        next_starts_space = (
            index + 1 < len(records) and records[index + 1][2]
        )
        previous_punctuation = index > 0 and records[index - 1][3]
        length = 1 if int(token_index) == unk_idx else codepoint_count
        if punctuation and next_starts_space:
            length += 1
        elif previous_punctuation and starts_space:
            length -= 1
        char_lens[0, index + 1] = length
        char_ids.extend(record_char_ids)
    return np.asarray([char_ids], dtype=text_seqs.dtype), char_lens


def validate_capture(
    capture_dir: Path,
    properties: list[tuple[str, int, bool, bool, list[int]]],
    pad_idx: int,
    unk_idx: int,
    eos_idx: int,
) -> dict[str, Any]:
    references = sorted(capture_dir.glob("call_*.npz"))
    if not references:
        raise FileNotFoundError(f"No T2U captures under {capture_dir}")
    cases = []
    for path in references:
        with np.load(path) as data:
            chars, lengths = build_inputs_from_exported_properties(
                data["text_seqs"], properties, pad_idx, unk_idx, eos_idx
            )
            chars_equal = bool(np.array_equal(chars, data["char_seqs"]))
            lengths_equal = bool(np.array_equal(lengths, data["char_lens"]))
        cases.append(
            {
                "reference": path.name,
                "char_seqs_equal": chars_equal,
                "char_lens_equal": lengths_equal,
            }
        )
        if not chars_equal or not lengths_equal:
            raise RuntimeError(f"Runtime tokenizer table diverged at {path.name}")
    return {"passed": True, "case_count": len(cases), "cases": cases}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    char_tokenizer = load_unity_char_tokenizer(args.model_name)
    card = asset_store.retrieve_card(args.model_name)
    unit_languages = card.field("unit_langs").as_list(str)

    text_vocab_size = int(text_tokenizer.model.vocabulary_size)
    properties = [
        token_properties(text_tokenizer, char_tokenizer, index)
        for index in range(text_vocab_size)
    ]
    token_table = args.output_dir / "text_tokens.tsv"
    with token_table.open("w", encoding="ascii", newline="\n") as output:
        for index, (piece, length, starts_space, punctuation, char_ids) in enumerate(
            properties
        ):
            output.write(
                "\t".join(
                    [
                        str(index),
                        piece.encode("utf-8").hex(),
                        str(length),
                        "1" if starts_space else "0",
                        "1" if punctuation else "0",
                        ",".join(str(value) for value in char_ids),
                        str(len(char_ids)),
                    ]
                )
                + "\n"
            )

    vocoder = load_vocoder_model(
        args.vocoder_name, device=torch.device("cpu"), dtype=torch.float32
    )
    vocoder_languages = vocoder.lang_spkr_idx_map["multilingual"]
    vocoder_speakers = vocoder.lang_spkr_idx_map["multispkr"]
    language_rows = []
    language_manifest = []
    excluded_languages = []
    for language in unit_languages:
        if language not in vocoder_languages or language not in vocoder_speakers:
            excluded_languages.append(language)
            continue
        prefix = text_tokenizer.create_encoder(
            lang=language, mode="target"
        ).prefix_indices
        if prefix is None or prefix.numel() != 2:
            raise RuntimeError(f"Unexpected target prefix for {language}")
        try:
            vocoder_language = int(vocoder_languages[language])
            vocoder_speaker = int(vocoder_speakers[language][0])
        except (IndexError, TypeError) as error:
            raise RuntimeError(
                f"No vocoder conditioning for UnitY language {language}"
            ) from error
        row = (
            language,
            int(prefix[0]),
            int(prefix[1]),
            vocoder_language,
            vocoder_speaker,
        )
        language_rows.append(row)
        language_manifest.append(
            {
                "language": language,
                "prefix_indices": [row[1], row[2]],
                "vocoder_language_id": vocoder_language,
                "vocoder_speaker_id": vocoder_speaker,
            }
        )

    pad_idx = int(text_tokenizer.vocab_info.pad_idx)
    unk_idx = int(text_tokenizer.vocab_info.unk_idx)
    eos_idx = int(text_tokenizer.vocab_info.eos_idx)
    comma_idx = int(text_tokenizer.model.token_to_index(","))
    metadata = args.output_dir / "runtime.meta"
    with metadata.open("w", encoding="utf-8", newline="\n") as output:
        output.write("schema_version=1\n")
        output.write(f"text_vocab_size={text_vocab_size}\n")
        output.write(f"text_pad_idx={pad_idx}\n")
        output.write(f"text_unk_idx={unk_idx}\n")
        output.write(f"text_eos_idx={eos_idx}\n")
        output.write(f"comma_idx={comma_idx}\n")
        output.write(f"language_count={len(language_rows)}\n")
        for index, row in enumerate(language_rows):
            output.write(f"language.{index}={','.join(map(str, row))}\n")

    validation = None
    if args.capture_dir is not None:
        validation = validate_capture(
            args.capture_dir, properties, pad_idx, unk_idx, eos_idx
        )

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "algorithm": "official NLLB/UnitY tokenizer tables; no model transform",
        "text_vocab_size": text_vocab_size,
        "char_vocab_size": int(char_tokenizer.model.vocabulary_size),
        "indices": {
            "pad": pad_idx,
            "unk": unk_idx,
            "eos": eos_idx,
            "comma": comma_idx,
        },
        "languages": language_manifest,
        "excluded_unit_languages_without_vocoder": excluded_languages,
        "files": {
            token_table.name: sha256(token_table),
            metadata.name: sha256(metadata),
        },
        "capture_validation": validation,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
