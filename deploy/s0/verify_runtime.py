#!/usr/bin/env python3
import json
import platform

import torch
import torchaudio
from fairseq2 import __version__ as fairseq2_version

import seamless_communication
from seamless_communication.streaming.agents.seamless_streaming_s2st import (
    SeamlessStreamingS2STAgent,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the AGX CUDA device")

    payload = {
        "architecture": platform.machine(),
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "fairseq2": fairseq2_version,
        "seamless_communication": seamless_communication.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "streaming_agent": SeamlessStreamingS2STAgent.__name__,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
