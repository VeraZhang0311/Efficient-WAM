from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import torch


class WanT5Embedder:
    """Thin wrapper around WAN UMT5 used by dataset conversion and inference."""

    def __init__(
        self,
        wan_root: str | Path,
        device: str = "cuda:0",
        text_len: int = 512,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from third_party.wan.modules.t5 import T5EncoderModel

        self.wan_root = Path(wan_root)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.encoder = T5EncoderModel(
            text_len=int(text_len),
            dtype=dtype,
            device=self.device,
            checkpoint_path=str(self.wan_root / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(self.wan_root / "google" / "umt5-xxl"),
        )

    def encode(self, prompts: Iterable[str]) -> list[torch.Tensor]:
        encoded = self.encoder(list(prompts), self.device)
        if isinstance(encoded, torch.Tensor):
            if encoded.dim() != 3:
                raise ValueError(f"Unexpected T5 tensor shape: {tuple(encoded.shape)}")
            return [item.detach().cpu() for item in encoded]
        if isinstance(encoded, list):
            tensors = []
            for item in encoded:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.detach().cpu())
                else:
                    tensors.append(torch.as_tensor(item))
            return tensors
        raise TypeError(f"Unexpected T5 output type: {type(encoded)!r}")


def save_prompt_embedding(
    encoded_prompt: str,
    output_path: str | Path,
    wan_root: str | Path,
    device: str = "cuda:0",
    text_len: int = 512,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    embedder = WanT5Embedder(wan_root=wan_root, device=device, text_len=text_len)
    torch.save(embedder.encode([encoded_prompt]), output)
    return output


def load_text_embedding(path: str | Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Text embedding file is empty: {path}")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.dim() == 3:
        value = value.squeeze(0)
    if value.dim() != 2:
        raise ValueError(f"Expected text embedding [seq, dim], got {tuple(value.shape)} from {path}")
    return value
