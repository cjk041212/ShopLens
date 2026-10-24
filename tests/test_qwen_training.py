"""Real tiny Qwen3+PEFT tests; synthetic fixtures never enter formal outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import shoplens

# Standalone staging tests extend the existing package without editing baseline.
staging = str(Path(__file__).resolve().parents[1] / "src/shoplens")
if staging not in shoplens.__path__:
    shoplens.__path__.insert(0, staging)

from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3Config, Qwen3Model
from shoplens.models import pad_histories
from shoplens.qwen_models import QwenHistoryRecommender, multi_positive_infonce
from shoplens.qwen_text import embed_texts, encode_texts, last_token_pool
from shoplens import qwen_pipeline as pipeline


class TinyTokenizer:
    padding_side = "left"
    def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
        sequences = []
        for text in texts:
            words = [3 + int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 60 for word in text.split()]
            sequences.append(([1] + words + [2])[:max_length])
        length = max(map(len, sequences))
        ids, masks = [], []
        for values in sequences:
            ids.append([0] * (length - len(values)) + values)
            masks.append([0] * (length - len(values)) + [1] * len(values))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


def tiny_base():
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         pad_token_id=0, eos_token_id=2, bos_token_id=1, attention_dropout=0.)
    return Qwen3Model(config)


def make_lora(base):
    for parameter in base.parameters():
        parameter.requires_grad = False
    return get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=2, lora_alpha=4,
                           target_modules=["q_proj", "v_proj"], lora_dropout=0.))


def test_last_token_pool_is_differentiable_for_padding_sides():
    hidden = torch.randn(2, 4, 5, requires_grad=True)
    masks = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    pooled = last_token_pool(hidden, masks)
    assert torch.equal(pooled[0], hidden[0, 3]) and torch.equal(pooled[1], hidden[1, 1])
    pooled.sum().backward()
    assert hidden.grad[0, 3].sum() == 5 and hidden.grad[1, 1].sum() == 5


def test_preference_loss_reaches_real_peft_and_frozen_base_stays_fixed():
    torch.manual_seed(9); torch.set_num_threads(1)
    encoder, tokenizer = make_lora(tiny_base()), TinyTokenizer()
    base_before = {name: value.detach().clone() for name, value in encoder.named_parameters() if "lora_" not in name}
    rng = np.random.default_rng(3)
    ranker = QwenHistoryRecommender(rng.normal(size=(6, 32)).astype("float32"), rng.normal(size=(6, 8)).astype("float32"), embedding_dim=16)
    embeddings = embed_texts(encoder, tokenizer, ["red hiking bag", "blue running shoe", "small cotton shirt", "wooden baby toy"], "cpu", 24)
    histories = torch.from_numpy(pad_histories([[0, 1], [2]], 20))
    loss = ranker(histories, torch.tensor([3, 4]), torch.tensor([4, 5]), embeddings[:2], embeddings[2:])
    parameters = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters + list(ranker.parameters()), lr=.01)
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None) > 0
    optimizer.step()
    assert all(torch.equal(base_before[name], value) for name, value in encoder.named_parameters() if "lora_" not in name)
    assert any(torch.count_nonzero(value).item() for name, value in encoder.named_parameters() if "lora_B" in name)


def test_semantic_gradient_isolation_and_same_asin_multi_positive():
    torch.manual_seed(11)
    encoder, tokenizer = make_lora(tiny_base()), TinyTokenizer()
    vectors = embed_texts(encoder, tokenizer, ["a soft cotton shirt", "a small red bag", "a large blue bottle", "a winter warm coat"], "cpu", 24)
    rng = np.random.default_rng(4)
    ranker = QwenHistoryRecommender(rng.normal(size=(6, 32)).astype("float32"), rng.normal(size=(6, 8)).astype("float32"), embedding_dim=16)
    history = torch.from_numpy(pad_histories([[0], [1]], 20))
    bpr = ranker(history, torch.tensor([2, 3]), torch.tensor([4, 5]), vectors[:2].detach(), vectors[2:].detach())
    lora = [p for p in encoder.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(bpr, lora, allow_unused=True, retain_graph=True)
    assert all(value is None for value in gradients)
    semantic = multi_positive_infonce(torch.randn(2, 32), vectors[:2], torch.tensor([2, 3]))
    (semantic + bpr).backward()
    assert sum(float(p.grad.abs().sum()) for p in lora if p.grad is not None) > 0
    same_item = multi_positive_infonce(torch.randn(3, 32), torch.randn(3, 32), torch.tensor([5, 5, 5]))
    assert float(same_item) == 0.0


def test_encoder_export_uses_the_same_tokenization_pooling_and_norm():
    torch.manual_seed(5)
    encoder, tokenizer = make_lora(tiny_base()), TinyTokenizer()
    encoder.eval()
    texts = ["one compact product", "a much longer lightweight sporting product description"]
    with torch.inference_mode():
        live = embed_texts(encoder, tokenizer, texts, "cpu", 8).numpy()
    cached = encode_texts(encoder, tokenizer, texts, "cpu", batch_size=1, max_length=8)
    np.testing.assert_allclose(live, cached, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(cached, axis=1), 1, atol=1e-6)


def write_fixture(root):
    processed = root / "data/processed/baby"; artifacts = root / "artifacts/baby"
    processed.mkdir(parents=True); artifacts.mkdir(parents=True)
    ids = [f"product{i:02d}" for i in range(24)]
    items = [{"item_id": item, "title": f"Product {i}", "text": f"Product {i} category {i % 5} useful light material {i % 3}"} for i, item in enumerate(ids)]
    events, reviews = [], []
    for user in range(6):
        for time, split in enumerate(("train", "train", "train", "validation", "test"), 1):
            row = {"user_id": f"user{user}", "item_id": ids[(user * 3 + time) % len(ids)], "timestamp": time * 100 + user, "rating": 5, "split": split}
            events.append(row)
            if split == "train":
                reviews.append({**row, "text": f"This useful product is light and works well for family {user}"})
    for name, rows in (("items", items), ("events", events), ("reviews_train", reviews)):
        (processed / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (processed / "audit.json").write_text(json.dumps({"domain": "baby", "synthetic": True, "scope": "synthetic fixture",
                                                     "cutoffs": {"train_end": 350, "validation_end": 450}}))
    image = np.random.default_rng(13).normal(size=(24, 8)).astype("float32")
    np.savez_compressed(artifacts / "features.npz", item_ids=np.asarray(ids), image=image)
    (artifacts / "features.json").write_text(json.dumps({"synthetic": True, "features_sha256": pipeline.digest(artifacts / "features.npz")}))


@pytest.fixture
def pipeline_setup(tmp_path, monkeypatch):
    torch.set_num_threads(1); torch.manual_seed(100)
    fixed_base = tiny_base()
    base_path = tmp_path / "tiny-model"; fixed_base.save_pretrained(base_path)
    fixed_state = {key: value.clone() for key, value in fixed_base.state_dict().items()}
    def factory(model_path, device, create_lora=False, adapter_path=None, lora_config=None, trainable=True):
        base = tiny_base(); base.load_state_dict(fixed_state); base.config._name_or_path = str(base_path)
        for p in base.parameters():p.requires_grad = False
        if create_lora:
            base = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, **lora_config))
        return base.to(device), TinyTokenizer()
    monkeypatch.setattr(pipeline, "load_text_model", factory)
    config = {"diagnostic": True, "warmup_epochs": 2, "warmup_min_epochs": 1, "warmup_patience": 2,
              "warmup_batch_size": 4, "adaptation_limit": 10, "adaptation_batch_size": 4,
              "embedding_dim": 16, "max_length": 24, "export_batch_size": 8, "checkpoint_steps": 1,
              "lora_config": {"r": 2, "lora_alpha": 4, "lora_dropout": .05, "target_modules": ["q_proj", "v_proj"], "bias": "none"}}
    return config, str(base_path)


def test_full_diagnostic_pipeline_export_scores_match_serving(tmp_path, pipeline_setup):
    config, model_path = pipeline_setup
    root = tmp_path / "fixture"; write_fixture(root)
    result = pipeline.run_one(root, "baby", 42, model_path, "cpu", config)
    assert result["status"] == "diagnostic_complete"
    from shoplens.qwen_inference import QwenScorer
    for mode in pipeline.MODES:
        directory = root / "artifacts/qwen/baby" / mode / "seed42"
        assert (directory / "diagnostic_complete.json").exists()
        assert not (directory / "complete.json").exists() and not (directory / "results.json").exists()
        training = json.loads((directory / "training.json").read_text())
        assert training["selected_epoch"] in (1, 2)
        assert len(training["adaptation_validation_ndcg@10"]) == 2
        assert training["training_numerics"] == {
            "candidate_training_sdpa_backend": "not_applicable" if mode == "qwen_frozen" else "automatic",
            "nonfinite_gradient_guard": True, "cached_encoding_sdpa_backend": "automatic"}
        evidence = training["gradient_evidence"]
        assert evidence["head_gradient_max_norm"] > 0
        if mode == "qwen_preference":assert evidence["bpr_to_lora_max_norm"] > 0
        if mode == "qwen_semantic":assert evidence["bpr_to_lora_max_norm"] == 0 and evidence["total_lora_gradient_max_norm"] > 0
        with np.load(root / "artifacts/qwen/baby/base/features.npz") as base, np.load(directory / "candidate.npz") as candidate:
            ranker = QwenHistoryRecommender(base["text"], base["image"], candidate["text"], embedding_dim=16)
        ranker.load_state_dict(torch.load(directory / "models" / f"{mode}.pt", weights_only=True)["state_dict"]); ranker.eval()
        histories = [[0, 1], [2, 3, 4]]
        with torch.inference_mode():
            direct = (ranker.user_vectors(torch.from_numpy(pad_histories(histories, 20))) @ ranker.item_vectors().T).numpy()
        served = QwenScorer(root / "data/processed/baby", directory, mode, "cpu").score_histories(histories)
        np.testing.assert_allclose(direct, served, atol=1e-6)


def test_partial_step_resume_matches_uninterrupted_real_peft(tmp_path, pipeline_setup, monkeypatch):
    config, model_path = pipeline_setup
    uninterrupted, resumed = tmp_path / "uninterrupted", tmp_path / "resumed"
    write_fixture(uninterrupted); write_fixture(resumed)
    pipeline.run_one(uninterrupted, "baby", 42, model_path, "cpu", config)
    triggered = {"value": False}
    def stop_after_first_preference_step(start, settings):
        checkpoint = resumed / "artifacts/qwen/baby/qwen_preference/seed42/resume.pt"
        if checkpoint.exists() and not triggered["value"]:
            progress = torch.load(checkpoint, weights_only=False)["progress"]
            if progress["epoch"] == 0 and 0 < progress["next_start"] < 10:
                triggered["value"] = True
                return True
        return False
    monkeypatch.setattr(pipeline, "_deadline_reached", stop_after_first_preference_step)
    with pytest.raises(pipeline.ResumableStop):
        pipeline.run_one(resumed, "baby", 42, model_path, "cpu", config)
    monkeypatch.setattr(pipeline, "_deadline_reached", lambda *_: False)
    pipeline.run_one(resumed, "baby", 42, model_path, "cpu", config)
    for mode in pipeline.MODES:
        first = uninterrupted / "artifacts/qwen/baby" / mode / "seed42"
        second = resumed / "artifacts/qwen/baby" / mode / "seed42"
        a = torch.load(first / "models" / f"{mode}.pt", weights_only=True)["state_dict"]
        b = torch.load(second / "models" / f"{mode}.pt", weights_only=True)["state_dict"]
        assert all(torch.equal(a[key], b[key]) for key in a)
        with np.load(first / "candidate.npz") as av, np.load(second / "candidate.npz") as bv:
            np.testing.assert_array_equal(av["text"], bv["text"])


def test_finite_loss_nonfinite_lora_gradient_stops_before_optimizer(tmp_path, pipeline_setup, monkeypatch):
    config, model_path = pipeline_setup
    root = tmp_path / "bad-gradient"; write_fixture(root)
    factory = pipeline.load_text_model
    observed = {}

    def inject_bad_backward(*args, **kwargs):
        encoder, tokenizer = factory(*args, **kwargs)
        if kwargs.get("create_lora"):
            trainable = {name: value for name, value in encoder.named_parameters() if value.requires_grad}
            observed["parameters"] = trainable
            observed["before"] = {name: value.detach().clone() for name, value in trainable.items()}
            next(value for name, value in trainable.items() if "lora_B" in name).register_hook(
                lambda gradient: torch.full_like(gradient, float("nan")))
        return encoder, tokenizer

    semantic_loss = pipeline.multi_positive_infonce
    def record_finite_loss(*args, **kwargs):
        loss = semantic_loss(*args, **kwargs)
        observed["forward_loss_finite"] = bool(torch.isfinite(loss))
        return loss

    optimizer_step = torch.optim.AdamW.step
    def observe_update(optimizer, *args, **kwargs):
        if len(optimizer.param_groups) == 2:
            observed["adaptation_optimizer_called"] = True
        return optimizer_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(pipeline, "load_text_model", inject_bad_backward)
    monkeypatch.setattr(pipeline, "multi_positive_infonce", record_finite_loss)
    monkeypatch.setattr(torch.optim.AdamW, "step", observe_update)
    with pytest.raises(RuntimeError, match="non-finite"):
        pipeline.run_one(root, "baby", 42, model_path, "cpu", config)
    assert observed["forward_loss_finite"] and not observed.get("adaptation_optimizer_called", False)
    assert any(not torch.isfinite(value.grad).all() for value in observed["parameters"].values() if value.grad is not None)
    assert all(torch.equal(observed["before"][name], value) for name, value in observed["parameters"].items())
    failed = root / "artifacts/qwen/baby/qwen_semantic/seed42"
    assert not (failed / "diagnostic_complete.json").exists()
    assert not list((root / "artifacts/qwen/baby").glob("qwen_*/seed*/complete.json"))
