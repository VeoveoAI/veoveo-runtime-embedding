"""Upstream-only model loading/preprocessing/encoding on an explicit FP32 GPU."""
import importlib.metadata
import os
import platform
from . import __version__
from .contract import *


def package_identity(source_revision: str) -> dict:
    return {
        "package": "veoveo-runtime-embedding", "version": __version__,
        "sourceRevision": source_revision, "python": platform.python_version(),
        "sentenceTransformers": importlib.metadata.version("sentence-transformers"),
        "transformers": importlib.metadata.version("transformers"),
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
        "pillow": importlib.metadata.version("Pillow"),
    }


class ModelBackend:
    def __init__(self, device: str, source_revision: str):
        if (platform.system(), platform.machine()) not in (("Darwin", "arm64"), ("Linux", "x86_64")):
            raise RuntimeError("unsupported GPU runtime platform")
        if device not in ("mps", "cuda") or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
            raise RuntimeError("an explicit GPU with CPU fallback disabled is required")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # Import after disabling fallback and downloads. Neither is request-controlled.
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.device = device
        self.runtime = package_identity(source_revision)
        expected = {"python": "3.14.6", "sentenceTransformers": "6.1.0", "transformers": "5.20.2", "pillow": "13.0.0"}
        expected["torch"] = "2.14.1+cu130" if device == "cuda" else "2.14.1"
        expected["torchvision"] = "0.29.1+cu130" if device == "cuda" else "0.29.1"
        if any(self.runtime[key] != value for key, value in expected.items()):
            raise RuntimeError("runtime package identity does not match the pinned baseline")
        if (device == "mps" and not torch.backends.mps.is_available()) or (device == "cuda" and not torch.cuda.is_available()):
            raise RuntimeError("the configured GPU is unavailable; CPU fallback is forbidden")
        self.model = SentenceTransformer(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
            device=device, model_kwargs={"torch_dtype": torch.float32},
            config_kwargs={"vision_config": None, "audio_config": None},
            trust_remote_code=False,
        )
        self.model.float().eval()
        if self.model.get_sentence_embedding_dimension() != DIMENSIONS or self.model.max_seq_length != MAX_ITEM_TOKENS:
            raise RuntimeError("cached model does not match the fixed text contract")
        if self.model.prompts.get("query") != SEARCH_PREFIX or self.model.prompts.get("CodeRetrieval") != CODE_PREFIX or self.model.prompts.get("document") != "title: none | text: ":
            raise RuntimeError("cached model prompt contract differs")
        if not self.model[1].include_prompt:
            raise RuntimeError("upstream prompt-inclusive pooling is required")
        if not self.ready():
            raise RuntimeError("model parameters are not on the configured FP32 GPU")
        self._observed_forward = False
        # Observe the actual lower transformer entry; upstream ST may legitimately
        # move final vectors to CPU for serialization after GPU inference.
        self._hook = self.model[0].auto_model.register_forward_pre_hook(self._observe_forward, with_kwargs=True)

    def _observe_forward(self, _module, args, kwargs):
        tensors = []
        def collect(value):
            if isinstance(value, self.torch.Tensor):
                tensors.append(value)
            elif isinstance(value, dict):
                for part in value.values(): collect(part)
            elif isinstance(value, (tuple, list)):
                for part in value: collect(part)
        collect(args)
        collect(kwargs)
        if not tensors or any(tensor.device.type != self.device for tensor in tensors):
            raise InferenceFailure()
        self._observed_forward = True

    def ready(self) -> bool:
        parameters = list(self.model.parameters())
        return bool(parameters) and all(parameter.device.type == self.device and parameter.dtype == self.torch.float32 for parameter in parameters)

    def capabilities(self) -> dict:
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "processingId": PROCESSING_ID, "dimensions": DIMENSIONS, "normalized": True,
            "execution": {"device": self.device, "dtype": "float32"},
            "runtime": self.runtime, "purposes": list(PURPOSES),
            "limits": {"maxItems": MAX_ITEMS, "maxContentBytes": MAX_CONTENT_BYTES,
                       "maxItemTokens": MAX_ITEM_TOKENS, "maxBatchTokens": MAX_BATCH_TOKENS},
        }

    def _count(self, text: str) -> int:
        # Use the same upstream processor as encode, including special tokens.
        features = self.model.preprocess(
            [text], prompt="", processing_kwargs={"text": {"truncation": False, "padding": False}},
        )
        return int(features["input_ids"].shape[-1])

    def encode(self, batch: Batch) -> dict:
        rendered, counts = prepare(batch, self._count)
        try:
            if not self.ready():
                raise InferenceFailure()
            self._observed_forward = False
            with self.torch.inference_mode():
                vectors = self.model.encode(
                    rendered, prompt="", batch_size=2, device=self.device,
                    precision="float32", normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False, processing_kwargs={"text": {"truncation": False}},
                )
            if not self._observed_forward or str(vectors.dtype) != "float32":
                raise InferenceFailure()
            values = validate_vectors(vectors.tolist(), len(batch.items))
            caps = self.capabilities()
            return {
                **{key: caps[key] for key in ("model", "processingId", "dimensions", "normalized", "execution", "runtime")},
                "requestId": str(batch.request_id),
                "items": [{"index": index, "purpose": item.purpose, "vector": vector, "tokenCount": count}
                          for index, (item, vector, count) in enumerate(zip(batch.items, values, counts, strict=True))],
                "usage": {"inputTokens": sum(counts)},
            }
        except Exception:
            # Upstream exception strings may contain input/tensors. Never expose them.
            raise InferenceFailure() from None
