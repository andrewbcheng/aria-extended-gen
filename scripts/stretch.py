import torch
from safetensors.torch import save_file

from aria.model import TransformerLM, ModelConfig
from aria.utils import _load_weight
from aria.config import load_model_config
from aria.tokenizer import AbsTokenizer, SeparatedAbsTokenizer, SecTokenizer

from torch.nn import Embedding, Linear

M_PATH = "large-abs-inst.safetensors"
M_CONFIG = "large"

# MORE TOKENS
tokenizer = SeparatedAbsTokenizer()
_tokenizer = SecTokenizer()
model_state = _load_weight(M_PATH, device="cpu")
model_state = {k: v for k, v in model_state.items() if "rotary_emb" not in k}

model_config = ModelConfig(**load_model_config(M_CONFIG))
model_config.set_vocab_size(tokenizer.vocab_size)
model = TransformerLM(model_config).to("cpu")
model.load_state_dict(model_state)


with torch.no_grad():
    _embedding = Embedding(
        num_embeddings=_tokenizer.vocab_size,
        embedding_dim=model_config.d_model,
    )
    _embedding.weight[: -(_tokenizer.vocab_size - tokenizer.vocab_size)] = (
        model.model.tok_embeddings.weight
    )

    _lm_head = Linear(model_config.d_model, _tokenizer.vocab_size, bias=False)
    _lm_head.weight[: -(_tokenizer.vocab_size - tokenizer.vocab_size)] = (
        model.lm_head.weight
    )

    model.model.tok_embeddings = _embedding
    model.lm_head = _lm_head

# Remove rotary embedding stuff
state_dict = model.state_dict()
state_dict = {k: v for k, v in state_dict.items() if "rotary_emb" not in k}

save_file(
    state_dict,
    "large-stretched.safetensors",
)

print(
    f"Stretched input/output layers from {tokenizer.vocab_size} to {_tokenizer.vocab_size} "
)

