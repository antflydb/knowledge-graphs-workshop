"""Real, lazy upstream inference for the stable workshop track.

Use plain GLiNER2 (not its ``local`` extra, which constrains transformers<5):
``gliner2 @ git+https://github.com/fastino-ai/GLiNER2.git@55656fbfa01d3d4a77485e1a1eeeaf682990ccdf``
and ``transformers @ git+https://github.com/huggingface/transformers.git@92cd495f2720c064bc78eb2d93e28704c5bce51f``.
The latter is the official Antfly EmbeddingGemma 2 Python reference pin.
Use torch==2.14.0, peft==0.20.0, numpy==2.3.5, safetensors==0.8.0 and
tokenizers==0.23.1 alongside those packages; lock their transitive dependencies.
No models load, download, or import torch merely by importing this module.
"""

from __future__ import annotations

import gc
import importlib.metadata
import math
import threading
from pathlib import Path
from typing import Any


DEFAULT_MODELS = {
    "extract": "fastino/gliner2.5-base-v1",
    "embed": "google/embeddinggemma-2",
    "decide": "fastino/GLiNER2.5-Decide",
}
DEFAULT_REVISIONS = {
    "extract": "72ac19b486cd4557424c8d61114e7530c243e9b0",
    "embed": "914f7f89142e33e77833254d9c9b90c3cef7303b",
    "decide": "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6",
}


def _anchored(text: str, span: dict) -> dict | None:
    """Validate supplied codepoint offsets; resolve absent offsets only if unique."""
    surface = span.get("text")
    if not isinstance(surface, str) or not surface:
        return None
    start, end = span.get("start"), span.get("end")
    if (
        type(start) is int and type(end) is int
        and 0 <= start < end <= len(text) and text[start:end] == surface
    ):
        return {"text": surface, "start": start, "end": end}
    if start is not None or end is not None:
        raise ValueError("Model span does not match source prose at supplied offsets")
    start = text.find(surface)
    if start < 0 or text.find(surface, start + 1) >= 0:
        return None
    return {"text": surface, "start": start, "end": start + len(surface)}


def _score(value: Any) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"model returned invalid probability: {value!r}")
    return score


def _normalize_extraction(text: str, raw: dict, schema: dict) -> dict:
    entities = []
    for label, spans in raw.get("entities", {}).items():
        if label not in schema["entities"]:
            raise ValueError(f"model returned an unrequested entity label: {label}")
        for span in spans:
            anchored = _anchored(text, span)
            if anchored is not None:
                entities.append({**anchored, "label": label, "score": _score(span["confidence"])})
    relations = []
    for label, rows in raw.get("relation_extraction", {}).items():
        if label not in schema["relations"]:
            raise ValueError(f"model returned an unrequested relation label: {label}")
        for row in rows:
            head = _anchored(text, row["head"])
            tail = _anchored(text, row["tail"])
            if head is None or tail is None:
                continue
            # BoundaryExtractor repeats its pair probability on both endpoints.
            # For a supplied legacy span model, use the lower endpoint score.
            score = min(_score(row["head"]["confidence"]), _score(row["tail"]["confidence"]))
            relations.append({"label": label, "head": head, "tail": tail, "score": score})
    return {"entities": entities, "relations": relations}


class PythonModels:
    """One resident F32 model at a time; serial requests, no inference fallback.

    ``models`` maps extract/embed/decide to HF IDs or local snapshot directories.
    Non-default remote IDs require an explicit revision. Local directories are
    operator-supplied artifacts, not asserted to match a default HF revision.
    Switching roles releases the old model before loading the next one. Peak
    load memory can exceed resident weight size: upstream GLiNER's loader also
    materializes its state dict. Use fresh processes for tight memory budgets.
    """

    def __init__(
        self, models: dict, *, device: str = "cpu", cache_dir: str | None = None,
        local_files_only: bool = False, token: str | None = None,
        revisions: dict | None = None,
    ):
        if set(models) != set(DEFAULT_MODELS):
            raise ValueError("models must contain exactly extract, embed, decide")
        self.models = dict(models)
        self.revisions = {}
        revisions = revisions or {}
        if set(revisions) - set(DEFAULT_MODELS):
            raise ValueError("revisions has an unknown model role")
        for role, name in self.models.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{role} model must be a nonempty model ID or directory")
            revision = revisions.get(role)
            if not Path(name).expanduser().is_dir():
                if revision is None and name == DEFAULT_MODELS[role]:
                    revision = DEFAULT_REVISIONS[role]
                if not isinstance(revision, str) or not revision:
                    raise ValueError(f"{role}: remote models require an explicit revision")
            self.revisions[role] = revision
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("device must be cpu, mps, or cuda; no automatic fallback")
        self.device = device
        self._hub_options = {
            "cache_dir": cache_dir, "local_files_only": local_files_only, "token": token,
        }
        self._lock = threading.RLock()
        self._model = None
        self._tokenizer = None
        self._role = None
        self._closed = False
        self._loaded = {}
        self._runtime = {}

    @property
    def metadata(self) -> dict:
        """Configured pins are distinct from artifacts actually loaded so far."""
        with self._lock:
            return {
                "provider": "python", "device": self.device, "dtype": "float32",
                "models": dict(self.models), "revisions": dict(self.revisions),
                "loaded": {role: dict(info) for role, info in self._loaded.items()},
                "runtime_versions": dict(self._runtime),
                "embedding_recipe": "embeddinggemma2-f32-mean-v1",
                "classification_method": "gliner2_single_label_softmax",
            }

    def _release(self) -> None:
        self._model = None
        self._tokenizer = None
        self._role = None
        gc.collect()
        # Import only after a model has actually been used.
        if self._runtime:
            import torch

            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

    def _load(self, role: str):
        if self._closed:
            raise RuntimeError("PythonModels is closed")
        if role == self._role:
            return self._model
        self._release()
        try:
            import torch
            from huggingface_hub import snapshot_download

            if self.device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("requested MPS device is not available")
            if self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("requested CUDA device is not available")
            name = self.models[role]
            local = Path(name).expanduser()
            is_local = local.is_dir()
            directory = str(local.resolve()) if is_local else snapshot_download(
                name, revision=self.revisions[role],
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
                **self._hub_options,
            )
            if role == "embed":
                from transformers import AutoModel, AutoTokenizer

                model = AutoModel.from_pretrained(
                    directory, dtype=torch.float32, local_files_only=True,
                    trust_remote_code=False,
                ).eval().to(self.device)
                tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            else:
                from gliner2 import AutoExtractor

                model = AutoExtractor.from_pretrained(
                    directory, local_files_only=True, use_flashdeberta=False,
                ).float().eval().to(self.device)
                tokenizer = None
            runtime = {}
            for package in ("gliner2", "transformers", "torch", "huggingface-hub", "peft", "tokenizers", "safetensors"):
                try:
                    runtime[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    runtime[package] = "not installed"
            self._runtime = runtime
            self._loaded[role] = {
                "model": name, "revision": None if is_local else self.revisions[role],
                "snapshot_directory": directory, "source": "local" if is_local else "huggingface",
            }
            self._model, self._tokenizer, self._role = model, tokenizer, role
            return model
        except Exception as exc:
            raise RuntimeError(
                f"cannot load {role} model {self.models[role]!r} "
                f"at revision {self.revisions[role]!r}: {exc}"
            ) from exc

    @staticmethod
    def _preflight(model, text: str, schema) -> int:
        """Use the upstream schema tokenizer, not a guessed words/token ratio."""
        words = sum(1 for _ in model.processor.word_splitter(text, lower=False))
        schema_dicts, _ = model._build_schema_dicts_and_metadata([schema])
        batch = model.processor.collate_fn_inference(
            [(text, schema_dicts[0])], max_len=None, architecture=model.architecture,
            error_policy="raise", build_targets=False, on_capacity_exceeded="raise",
        )
        encoded = batch.input_ids.shape[-1]
        limit = min(512, model.encoder.config.max_position_embeddings)
        if encoded > limit:
            raise ValueError(
                f"schema, instruction and text require {encoded} tokens; limit is {limit}; "
                "supply a smaller excerpt (input is never truncated)"
            )
        return words

    def extract(self, text: str, schema: dict, threshold: float) -> dict:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if set(schema) != {"entities", "relations"}:
            raise ValueError("extraction schema must contain entities and relations")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        with self._lock:
            model = self._load("extract")
            compiled = model.create_schema().entities(schema["entities"]).relations(schema["relations"])
            words = self._preflight(model, text, compiled)
            raw = model.extract(
                text, compiled, threshold=threshold, include_confidence=True,
                include_spans=True, max_len=words,
            )
            return _normalize_extraction(text, raw, schema)

    def embed(self, texts: list[str], task_type: str, dimensions: int) -> list[list[float]]:
        if task_type not in {"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"}:
            raise ValueError("task_type must be RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY")
        if type(dimensions) is not int or dimensions not in {128, 256, 512, 768}:
            raise ValueError("EmbeddingGemma 2 dimensions must be 128, 256, 512, or 768")
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be a list of strings")
        with self._lock:
            if self._closed:
                raise RuntimeError("PythonModels is closed")
            if not texts:
                return []
            model = self._load("embed")
            import torch

            prefix = "title: none | text: " if task_type == "RETRIEVAL_DOCUMENT" else "task: search result | query: "
            vectors = []
            with torch.inference_mode():
                # Serial, unpadded rows bound activation memory on 8 GiB hosts.
                for text in texts:
                    inputs = self._tokenizer(prefix + text, return_tensors="pt", truncation=False)
                    count = inputs["input_ids"].shape[-1]
                    if count > 8192:
                        raise ValueError(f"embedding input needs {count} tokens including wrappers; limit is 8192")
                    inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                    hidden = model(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                    vector = torch.nn.functional.normalize(pooled[:, :dimensions], p=2, dim=-1)
                    if vector.shape[-1] != dimensions or not bool(torch.isfinite(vector).all()) or not bool(vector.norm() > 0):
                        raise ValueError("embedding model returned an invalid vector")
                    vectors.append(vector[0].cpu().tolist())
            return vectors

    def classify(self, text: str, labels: dict[str, str], instruction: str) -> dict:
        if not isinstance(text, str) or not isinstance(instruction, str):
            raise TypeError("text and instruction must be strings")
        if not isinstance(labels, dict) or not labels or any(
            not isinstance(label, str) or not label or not isinstance(description, str)
            for label, description in labels.items()
        ):
            raise ValueError("labels must map nonempty label names to descriptions")
        with self._lock:
            model = self._load("decide")
            compiled = model.create_schema().classification(
                "classification", labels, multi_label=False, class_act="softmax", prompt=instruction,
            )
            words = self._preflight(model, text, compiled)
            captured = []

            def capture_logits(module, args, output):
                captured.append(output.detach().squeeze(-1).cpu())

            # Ordinary classify_text exposes only the winning score. Capture
            # its actual classifier logits, as Antfly's upstream oracle does,
            # to expose every softmax score without an extra model invocation.
            hook = model.classifier.register_forward_hook(capture_logits)
            try:
                result = model.classify_text(
                    text, {"classification": {
                        "labels": labels, "multi_label": False,
                        "class_act": "softmax", "prompt": instruction,
                    }}, include_confidence=True, max_len=words,
                )
            finally:
                hook.remove()
            if len(captured) != 1 or captured[0].numel() != len(labels):
                raise RuntimeError("unexpected upstream classification logits shape")
            import torch

            probabilities = torch.softmax(captured[0].reshape(-1), dim=-1).tolist()
            scores = {label: _score(score) for label, score in zip(labels, probabilities)}
            label = max(scores, key=scores.__getitem__)
            if result["classification"]["label"] != label:
                raise RuntimeError("upstream classification label and captured probabilities disagree")
            return {"label": label, "scores": scores, "method": "gliner2_single_label_softmax"}

    def close(self) -> None:
        with self._lock:
            self._release()
            self._closed = True
