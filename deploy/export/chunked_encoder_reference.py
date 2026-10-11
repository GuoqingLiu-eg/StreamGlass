#!/usr/bin/env python3
"""Frozen-prefix + sliding-window chunked encoder study (design doc step 1).

Official weights and official submodules only; measures the architectural
deviation of the cached encoder design against the official cumulative
encoder on real audio. No ONNX, no model modification.
"""
import argparse
import json
from pathlib import Path

import torch
from fairseq2.assets import asset_store
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock
from fairseq2.nn.padding import PaddingMask

from seamless_communication.models.unity import load_unity_config, load_unity_model

CHUNK_FBANK = 32          # 320 ms
CONV_LEFT = 15            # depthwise kernel 31, symmetric padding
START_FBANK = 222         # production minimum profile


def load_model(name, device, dtype):
    config = load_unity_config(name)
    config.use_text_decoder = False
    config.use_text_encoder = False
    config.t2u_config = None
    card = asset_store.retrieve_card(name)
    card.field("model_config").set(config)
    return load_unity_model(card, device=device, dtype=dtype).eval()


def wav_to_fbank(path, device, dtype):
    decoder = AudioDecoder(dtype=torch.float32)
    audio = decoder(MemoryBlock(Path(path).read_bytes()))
    converter = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=False,
        device=device,
        dtype=dtype,
    )
    return converter(audio)["fbank"].unsqueeze(0)


def official_encode(model, fbank):
    lens = torch.tensor([fbank.shape[1]], device=fbank.device)
    mask = PaddingMask(lens, fbank.shape[1])
    seqs, mask = model.speech_encoder_frontend(fbank, mask)
    out, _ = model.speech_encoder(seqs, mask)
    return out


class ChunkedEncoder:
    """Frozen prefix + sliding recompute window over the inner conformer."""

    def __init__(self, model, window, max_frames=1536):
        enc = model.speech_encoder
        self.frontend = model.speech_encoder_frontend
        self.enc = enc
        self.layers = list(enc.inner.layers)
        self.inner_final_norm = getattr(enc.inner, "layer_norm", None)
        self.window = window
        n_layers = len(self.layers)
        dim = enc.model_dim
        kw = dict(device=next(model.parameters()).device,
                  dtype=next(model.parameters()).dtype)
        self.attn_in = torch.zeros(n_layers, max_frames, dim, **kw)
        self.conv_in = torch.zeros(n_layers, max_frames, dim, **kw)
        self.inner_out = torch.zeros(max_frames, dim, **kw)

    @torch.inference_mode()
    def step(self, fbank):
        lens = torch.tensor([fbank.shape[1]], device=fbank.device)
        feats, _ = self.frontend(fbank, PaddingMask(lens, fbank.shape[1]))
        total = feats.shape[1]
        w0 = max(0, total - self.window)
        x = feats[:, w0:total]

        for index, block in enumerate(self.layers):
            x = block._forward_ffn1(x)
            self.attn_in[index, w0:total] = x[0]
            full = self.attn_in[index, :total].unsqueeze(0)
            residual = x
            q = block.self_attn_layer_norm(x)
            kv = block.self_attn_layer_norm(full)
            a = block.self_attn(
                q, None, keys=kv, key_padding_mask=None, values=kv)
            x = a + residual

            self.conv_in[index, w0:total] = x[0]
            residual = x
            tail = max(0, w0 - CONV_LEFT)
            cin = self.conv_in[index, tail:total].unsqueeze(0)
            c = block.conv_layer_norm(cin)
            c = block.conv(c, None)[:, (w0 - tail):]
            x = c + residual

            x = block._forward_ffn2(x)
            x = block.layer_norm(x)

        if self.inner_final_norm is not None:
            x = self.inner_final_norm(x)
        self.inner_out[w0:total] = x[0]

        seqs = self.inner_out[:total].unsqueeze(0)
        if self.enc.inner_layer_norm is not None:
            seqs = self.enc.inner_layer_norm(seqs)
        seqs = seqs + 0.5 * self.enc._expand_contract(seqs)
        mask = None
        for layer in self.enc.adaptor_layers:
            seqs, mask = layer(seqs, mask)
        return self.enc.layer_norm(seqs)


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1), b.float().reshape(1, -1)).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--model-name", default="seamless_streaming_unity")
    parser.add_argument("--windows", type=int, nargs="+",
                        default=[100000, 128, 64, 32])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16
    model = load_model(args.model_name, device, dtype)
    fbank = wav_to_fbank(args.wav, device, dtype)
    total_fbank = fbank.shape[1]
    print(f"fbank frames: {total_fbank}")

    steps = list(range(CHUNK_FBANK, total_fbank + 1, CHUNK_FBANK))
    if steps[-1] != total_fbank:
        steps.append(total_fbank)

    with torch.inference_mode():
        official = {k: official_encode(model, fbank[:, :k]) for k in steps
                    if k >= START_FBANK}

    report = {"wav": args.wav, "fbank_frames": total_fbank, "windows": {}}
    for window in args.windows:
        chunked = ChunkedEncoder(model, window)
        rows = []
        for k in steps:
            out = chunked.step(fbank[:, :k])
            if k < START_FBANK:
                continue
            ref = official[k]
            rows.append({
                "fbank": k,
                "frames": int(out.shape[1]),
                "cos_all": cosine(out, ref),
                "cos_last8": cosine(out[:, -8:], ref[:, -8:]),
                "max_abs": float((out.float() - ref.float()).abs().max()),
            })
        summary = {
            "min_cos_all": min(r["cos_all"] for r in rows),
            "min_cos_last8": min(r["cos_last8"] for r in rows),
            "final_cos_all": rows[-1]["cos_all"],
            "final_cos_last8": rows[-1]["cos_last8"],
            "max_abs": max(r["max_abs"] for r in rows),
        }
        report["windows"][str(window)] = {"summary": summary, "steps": rows}
        print(f"R={window}: {json.dumps(summary)}")

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
