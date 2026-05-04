from .model_repa_vae import WanTransformer3DModel
from .utils import WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_vae, patchify


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
