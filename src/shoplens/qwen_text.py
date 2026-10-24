"""Pinned Qwen embeddings: differentiable training and identical cache encoding."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
REVIEW_INSTRUCTION = "Given a product review, retrieve the product description that matches it."
LORA_CONFIG = {"r": 8, "lora_alpha": 16, "lora_dropout": .05, "target_modules": ["q_proj", "v_proj"], "bias": "none"}


def last_token_pool(last_hidden_states, attention_mask):
    if attention_mask.ndim != 2 or torch.any(attention_mask.sum(1) == 0):
        raise ValueError("Each embedding input needs at least one valid token")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last = (positions.unsqueeze(0) * attention_mask.long()).max(1).values
    return last_hidden_states[torch.arange(len(last), device=last.device), last]


def embed_texts(model, tokenizer, texts, device, max_length=256):
    """No inference/no_grad wrapper here: LoRA gradients must survive pooling."""
    if not texts or max_length < 2:
        raise ValueError("Nonempty texts and max_length>=2 are required")
    batch = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    training_cuda = str(device).startswith("cuda") and model.training and torch.is_grad_enabled()
    # Autograd retains this SDPA backward; eval/cache encoding keeps automatic dispatch.
    with sdpa_kernel(SDPBackend.MATH) if training_cuda else nullcontext():
        output = model(**batch, return_dict=True, use_cache=False)
    return F.normalize(last_token_pool(output.last_hidden_state, batch["attention_mask"]).float(), dim=-1)


def encode_texts(model, tokenizer, texts, device, batch_size=64, max_length=256):
    if batch_size < 1 or not texts:
        raise ValueError("Nonempty catalog and positive batch_size required")
    was_training = model.training
    model.eval()
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            blocks.append(embed_texts(model, tokenizer, texts[start:start + batch_size], device, max_length).cpu().numpy())
    model.train(was_training)
    return np.concatenate(blocks).astype(np.float32)


def review_query(text, instruction=REVIEW_INSTRUCTION):
    return f"Instruct: {instruction}\nQuery:{text}" if instruction else str(text)


def load_text_model(model_path, device="cuda", *, create_lora=False, adapter_path=None, lora_config=None, trainable=True):
    from transformers import AutoModel, AutoTokenizer
    path = str(model_path)
    kwargs = {} if Path(path).exists() else {"revision": REVISION}
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left", **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    base = AutoModel.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa", **kwargs).to(device)
    base.config.use_cache = False
    for parameter in base.parameters():
        parameter.requires_grad = False
    if adapter_path is not None:
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=trainable)
    elif create_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        configuration = dict(LORA_CONFIG if lora_config is None else lora_config)
        base = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, **configuration))
    base.train(bool(create_lora and trainable))
    return base, tokenizer


def save_adapter(model, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), safe_serialization=True)


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return {key: tensor.detach().cpu().clone() for key, tensor in get_peft_model_state_dict(model).items()}


def restore_adapter(model, state):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, state)


def adapter_parameters(model):
    return [parameter for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]
