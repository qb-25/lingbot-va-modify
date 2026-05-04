import torch

from .model_spatial_forcing import WanTransformer3DModel
from .utils import load_vae, load_text_encoder, load_tokenizer, patchify, WanVAEStreamingWrapper


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
):
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
    )
    return model.to(torch_device)
