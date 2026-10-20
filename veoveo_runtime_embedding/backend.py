"""Pinned full-model native processing and inference on an explicit FP32 GPU."""
import copy
import importlib.metadata
import os
import platform
from tempfile import TemporaryDirectory
from . import __version__
from .contract import *
from .media import NativeMedia, resolve, suppress_transfer_logs


def package_identity(source_revision):
    from torchcodec._core import get_ffmpeg_library_versions
    native = get_ffmpeg_library_versions()
    ffmpeg = {key: native[key] for key in ("libavfilter", "libavutil", "libavcodec", "libavformat")}
    ffmpeg["version"] = native["ffmpeg_version"]
    return {"package": "veoveo-runtime-embedding", "version": __version__, "sourceRevision": source_revision,
            "python": platform.python_version(), "sentenceTransformers": importlib.metadata.version("sentence-transformers"),
            "transformers": importlib.metadata.version("transformers"), "torch": importlib.metadata.version("torch"),
            "torchvision": importlib.metadata.version("torchvision"), "pillow": importlib.metadata.version("Pillow"),
            "torchcodec": importlib.metadata.version("torchcodec"), "ffmpeg": ffmpeg}


class ModelBackend:
    def __init__(self, device, source_revision):
        if (platform.system(), platform.machine()) not in (("Darwin", "arm64"), ("Linux", "x86_64")):
            raise RuntimeError("unsupported GPU runtime platform")
        if device not in ("mps", "cuda") or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
            raise RuntimeError("an explicit GPU with CPU fallback disabled is required")
        self.storage_origin = configured_origin(os.environ.get("VEOVEO_EMBEDDING_STORAGE_ORIGIN"))
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # Native HTTP clients can log request URLs at INFO/DEBUG. Signed URLs never enter logs.
        suppress_transfer_logs()
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch, self.device = torch, device
        self.runtime = package_identity(source_revision)
        expected = {"python": "3.14.6", "sentenceTransformers": "6.1.0", "transformers": "5.20.2", "pillow": "13.0.0",
                    "torch": "2.14.1+cu130" if device == "cuda" else "2.14.1",
                    "torchvision": "0.29.1+cu130" if device == "cuda" else "0.29.1",
                    "torchcodec": "0.17.0+cpu" if device == "cuda" else "0.17.0"}
        if any(self.runtime[key] != value for key, value in expected.items()):
            raise RuntimeError("runtime package identity does not match the pinned baseline")
        if (device == "mps" and not torch.backends.mps.is_available()) or (device == "cuda" and not torch.cuda.is_available()):
            raise RuntimeError("the configured GPU is unavailable; CPU fallback is forbidden")
        self.model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, local_files_only=True, device=device,
                                        model_kwargs={"dtype": torch.float32}, trust_remote_code=False)
        self.model.float().eval()
        if self.model.get_sentence_embedding_dimension() != DIMENSIONS or any(self.model.prompts.get(k) != v for k,v in PROMPTS.items()):
            raise RuntimeError("cached model differs from the pinned embedding contract")
        if not self.model[1].include_prompt:
            raise RuntimeError("upstream prompt-inclusive pooling is required")
        backbone = self.model[0].auto_model
        if backbone.vision_tower is None or backbone.audio_tower is None:
            raise RuntimeError("the full native multimodal model is required")
        if self.model[0].processor.feature_extractor.sampling_rate != 16000:
            raise RuntimeError("native audio sampling rate differs")
        self.media = NativeMedia(self.model[0].processor)
        if not self.ready(): raise RuntimeError("model parameters are not on the configured FP32 GPU")

    def ready(self):
        parameters = list(self.model.parameters())
        return bool(parameters) and all(p.device.type == self.device and p.dtype == self.torch.float32 for p in parameters)

    def capabilities(self):
        return {"model": {"id": MODEL_ID, "revision": MODEL_REVISION}, "dimensions": DIMENSIONS,
                "supportedDimensions": list(SUPPORTED_DIMENSIONS), "normalized": True,
                "modalities": ["text", "image", "audio", "video"], "promptNames": list(PROMPTS),
                "defaultPrompt": "", "defaults": copy.deepcopy(DEFAULT_PREPROCESSING),
                "execution": {"device": self.device, "dtype": "float32", "batchSize": 1}, "runtime": self.runtime,
                "limits": {"maxItems": MAX_ITEMS, "maxPartsPerItem": MAX_PARTS_PER_ITEM, "maxParts": MAX_PARTS,
                           "maxContentBytes": MAX_CONTENT_BYTES, "maxItemTokens": MAX_ITEM_TOKENS,
                           "maxBatchTokens": MAX_BATCH_TOKENS, "maxArtifactBytes": MAX_ARTIFACT_BYTES,
                           "maxTransferBytes": MAX_TRANSFER_BYTES, "maxPixels": MAX_PIXELS, "maxTotalPixels": MAX_TOTAL_PIXELS,
                           "maxSourceSeconds": MAX_SOURCE_SECONDS, "maxAudioSeconds": MAX_AUDIO_SECONDS,
                           "maxVideoFps": MAX_FPS, "maxVideoFrames": MAX_FRAMES},
                "processingSemantics": {"limits": "Veoveo service admission bounds, not Google model capabilities",
                    "preprocessing": "Native CPU decoding, resampling, image patches and video sampling",
                    "dimensions": "Native Sentence Transformers truncation followed by normalization",
                    "videoTimestamps": "Native prompt uses frame index/source FPS; coverage also records actual decoded PTS",
                    "audioFromVideo": "Native soundtrack append; incompatible with separate audio parts",
                    "imageFrames": "Native first image frame only", "textPlaceholders": "Reserved media markers are refused"}}

    def _preflight(self, conversations, media, batch, kwargs):
        counts = []
        for conversation, rows in zip(conversations, media, strict=True):
            # HF audio-from-video appends an audio marker in-place. Give each native
            # invocation a fresh copy and always an explicit outer batch.
            features = self.model.preprocess(copy.deepcopy([conversation]), prompt=batch.prompt, processing_kwargs=kwargs)
            ids, mask = features["input_ids"], features["attention_mask"]
            if len(ids.shape) != 2 or ids.shape[0] != 1 or tuple(mask.shape) != tuple(ids.shape):
                raise Rejected("preprocessing_cardinality")
            counts.append(int(mask.sum().item()))
            expected_frames = [len(row["coverage"]["frameIndices"]) for row in rows if row["type"] == "video"]
            observed_frames = features["num_frames_per_video"].tolist() if "num_frames_per_video" in features else []
            if observed_frames != expected_frames: raise Rejected("preprocessing_coverage")
            expected_images = sum(row["type"] == "image" for row in rows)
            observed_images = features["pixel_values"].shape[0] if "pixel_values" in features else 0
            if expected_images != observed_images: raise Rejected("preprocessing_coverage")
            expected_audio = sum(row["type"] == "audio" or (row["type"] == "video" and row["coverage"]["audioIncluded"]) for row in rows)
            observed_audio = features["input_features_mask"].shape[0] if "input_features_mask" in features else 0
            if expected_audio != observed_audio: raise Rejected("preprocessing_coverage")
        validate_counts(counts)
        return counts

    def encode(self, batch):
        # This context and the GPU admission slot belong to the actual worker, not
        # the HTTP caller. Download/preprocess failures are confirmed not_started.
        with TemporaryDirectory(prefix="veoveo-embedding-") as directory:
            try:
                conversations, media = resolve(batch, directory, self.storage_origin, self.media)
                kwargs = batch.native_kwargs()
                kwargs["audio"].update({"sampling_rate": 16000, "load_audio_backend": "torchcodec"})
                counts = self._preflight(conversations, media, batch, kwargs)
                if not self.ready(): raise Rejected("not_ready", 503)
            except Rejected:
                raise
            except Exception:
                raise Rejected("preprocessing_failed") from None
            try:
                with self.torch.inference_mode():
                    vectors = self.model.encode(copy.deepcopy(conversations), prompt=batch.prompt, batch_size=1,
                        device=self.device, precision="float32", normalize_embeddings=True, convert_to_numpy=True,
                        truncate_dim=batch.dimensions, show_progress_bar=False, processing_kwargs=kwargs)
                if str(vectors.dtype) != "float32": raise InferenceFailure()
                values = validate_vectors(vectors.tolist(), len(batch.items), batch.dimensions)
                return {"model": {"id": MODEL_ID, "revision": MODEL_REVISION}, "processingId": batch.processing_id(),
                        "dimensions": batch.dimensions, "normalized": True, "execution": {"device": self.device, "dtype": "float32"},
                        "runtime": self.runtime, "requestId": str(batch.request_id),
                        "processing": {**({"promptName": batch.prompt_name} if batch.prompt_name else {}), "promptSha256": "sha256:" + hashlib.sha256(batch.prompt.encode()).hexdigest(),
                                       "preprocessing": batch.preprocessing},
                        "items": [{"index": index, "vector": vector, "tokenCount": count, "media": rows}
                                  for index, (vector, count, rows) in enumerate(zip(values, counts, media, strict=True))],
                        "usage": {"inputTokens": sum(counts)}}
            except Exception:
                raise InferenceFailure() from None
