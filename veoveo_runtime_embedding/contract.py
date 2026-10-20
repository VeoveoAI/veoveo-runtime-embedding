"""Closed private multimodal contract; model-free validation and processing identity."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from uuid import UUID
from urllib.parse import urlsplit

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIMENSIONS = 768
SUPPORTED_DIMENSIONS = (128, 256, 512, 768)
MAX_ITEMS, MAX_PARTS_PER_ITEM, MAX_PARTS = 8, 16, 32
MAX_CONTENT_BYTES = 65536
MAX_ITEM_TOKENS, MAX_BATCH_TOKENS = 8192, 16384
MAX_ARTIFACT_BYTES, MAX_TRANSFER_BYTES = 64 * 1024**2, 128 * 1024**2
MAX_PIXELS, MAX_TOTAL_PIXELS = 16_000_000, 64_000_000
MAX_SOURCE_SECONDS, MAX_AUDIO_SECONDS = 300, 600
MAX_FPS, MAX_FRAMES = 120, 256
MAX_JSON_BYTES, MAX_RESPONSE_BYTES, REQUEST_SECONDS = 524288, 1048576, 60
SOFT_TOKENS = (70, 140, 280, 560, 1120)
PROMPTS = {
    "SearchQuery": "task: search result | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Document": "title: none | text: ",
    "QuestionAnswering": "task: question answering | query: ",
    "FactChecking": "task: fact checking | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
}
DEFAULT_PREPROCESSING = {
    "image": {"maxSoftTokens": 280},
    "audio": {"truncation": False},
    "video": {"fps": 1, "maxFrames": 32, "overflowStrategy": "uniform", "maxSoftTokens": 140,
              "addTimestamps": False, "loadAudioFromVideo": False},
}

class Rejected(Exception):
    """Closed pre-inference error, without inputs or upstream exception bodies."""
    def __init__(self, code: str, status: int = 422):
        self.code, self.status = code, status
        super().__init__(code)

class InferenceFailure(Exception):
    pass


def closed(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) <= set(required) | set(optional):
        raise Rejected("invalid_request")
    return value


def positive(value, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise Rejected("invalid_option")
    return value


def uuid_text(value, *, version=None):
    try:
        identifier = UUID(value)
        if str(identifier) != value or (version is not None and identifier.version != version):
            raise ValueError()
        return identifier
    except (ValueError, TypeError, AttributeError):
        raise Rejected("invalid_request") from None


def configured_origin(value):
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password
                or parsed.path or parsed.query or parsed.fragment or not parsed.port):
            raise ValueError()
        return (parsed.scheme, parsed.hostname.lower(), parsed.port)
    except (ValueError, TypeError, AttributeError):
        raise RuntimeError("an explicit private storage origin with port and no path is required") from None


@dataclass(frozen=True, slots=True)
class Transfer:
    url: str = field(repr=False)
    headers: dict[str, str] = field(repr=False)
    expires_at: datetime

@dataclass(frozen=True, slots=True)
class Source:
    artifact_id: UUID
    media_type: str
    size: int
    sha256: str
    get: Transfer = field(repr=False)

    def identity(self):
        return {"artifactId": str(self.artifact_id), "mediaType": self.media_type, "size": self.size, "sha256": self.sha256}

@dataclass(frozen=True, slots=True)
class Part:
    type: str
    text: str | None = field(default=None, repr=False)
    source: Source | None = None

@dataclass(frozen=True, slots=True)
class Item:
    content: tuple[Part, ...]

@dataclass(frozen=True, slots=True)
class Batch:
    request_id: UUID
    items: tuple[Item, ...]
    dimensions: int = DIMENSIONS
    prompt: str = field(default="", repr=False)
    prompt_name: str | None = None
    preprocessing: dict = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_PREPROCESSING)))

    def native_kwargs(self):
        names = {"maxSoftTokens": "max_soft_tokens", "maxLength": "max_length", "maxFrames": "max_frames",
                 "overflowStrategy": "overflow_strategy", "addTimestamps": "add_timestamps", "loadAudioFromVideo": "load_audio_from_video"}
        return {"text": {"truncation": False}, **{group: {names.get(key, key): value for key, value in values.items()}
                for group, values in self.preprocessing.items()}}

    def processing(self):
        return {"modelId": MODEL_ID, "modelRevision": MODEL_REVISION, "dimensions": self.dimensions,
                "normalized": True, "pooling": "upstream_mean_projection_normalize", "prompt": self.prompt,
                "preprocessing": self.preprocessing, "layout": "ordered_structured_conversations", "audioSamplingRate": 16000}

    def processing_id(self):
        import rfc8785
        return "sha256:" + hashlib.sha256(rfc8785.dumps(self.processing())).hexdigest()


def parse_options(value):
    raw = closed(value, (), ("image", "audio", "video"))
    result = json.loads(json.dumps(DEFAULT_PREPROCESSING))
    for group in result:
        options = closed(raw.get(group, {}), (), ("maxLength", *result[group])) if group == "audio" else closed(raw.get(group, {}), (), result[group])
        result[group].update(options)
    for group in ("image", "video"):
        if type(result[group]["maxSoftTokens"]) is not int or result[group]["maxSoftTokens"] not in SOFT_TOKENS:
            raise Rejected("invalid_option")
    audio = result["audio"]
    if type(audio["truncation"]) is not bool or ("maxLength" in audio and not audio["truncation"]):
        raise Rejected("invalid_option")
    if audio["truncation"]:
        audio["maxLength"] = positive(audio.get("maxLength", 480000), MAX_SOURCE_SECONDS * 16000)
    video = result["video"]
    if type(video["fps"]) not in (int, float) or not math.isfinite(video["fps"]) or not 0 < video["fps"] <= MAX_FPS:
        raise Rejected("invalid_option")
    positive(video["maxFrames"], MAX_FRAMES)
    if video["overflowStrategy"] not in ("uniform", "truncate") or any(type(video[key]) is not bool for key in ("addTimestamps", "loadAudioFromVideo")):
        raise Rejected("invalid_option")
    return result


def parse_source(value, kind):
    source = closed(value, ("artifactId", "mediaType", "size", "sha256", "get"))
    media_type = source["mediaType"]
    if (not isinstance(media_type, str) or len(media_type) > 255 or not re.fullmatch(r"[a-zA-Z0-9!#$&^_.+-]+/[a-zA-Z0-9!#$&^_.+-]+", media_type)
            or not media_type.lower().startswith(("audio/", "video/") if kind == "audio" else kind + "/")):
        raise Rejected("invalid_media_type")
    size = positive(source["size"], MAX_ARTIFACT_BYTES)
    if not isinstance(source["sha256"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", source["sha256"]):
        raise Rejected("invalid_source")
    transfer = closed(source["get"], ("method", "url", "expiresAt"), ("headers",))
    url = transfer["url"]
    try:
        parsed = urlsplit(url)
        if (transfer["method"] != "GET" or not isinstance(url, str) or len(url) > 8192
                or parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password
                or parsed.fragment or any(ord(c) < 32 or ord(c) == 127 for c in url)):
            raise ValueError()
        parsed.port
        expiry = transfer["expiresAt"]
        if not isinstance(expiry, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", expiry):
            raise ValueError()
        expires = datetime.fromisoformat(expiry)
        if expires.tzinfo is None:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise Rejected("invalid_source") from None
    headers = transfer.get("headers", {})
    if not isinstance(headers, dict) or len(headers) > 32:
        raise Rejected("invalid_source")
    for key, val in headers.items():
        if (not isinstance(key, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key)
                or not isinstance(val, str) or len(val) > 8192 or any(ord(c) < 32 or ord(c) == 127 for c in val)):
            raise Rejected("invalid_source")
    return Source(uuid_text(source["artifactId"]), media_type, size, source["sha256"], Transfer(url, headers, expires))


def parse_batch(value):
    value = closed(value, ("requestId", "items"), ("dimensions", "promptName", "prompt", "preprocessing"))
    identifier = uuid_text(value["requestId"], version=7)
    dimensions = value.get("dimensions", DIMENSIONS)
    if type(dimensions) is not int or dimensions not in SUPPORTED_DIMENSIONS:
        raise Rejected("invalid_dimensions")
    if "prompt" in value and "promptName" in value:
        raise Rejected("invalid_prompt")
    prompt_name = value.get("promptName")
    if "promptName" in value and (not isinstance(prompt_name, str) or prompt_name not in PROMPTS):
        raise Rejected("invalid_prompt")
    prompt = value.get("prompt", PROMPTS[prompt_name] if prompt_name else "")
    if not isinstance(prompt, str):
        raise Rejected("invalid_prompt")
    preprocessing = parse_options(value.get("preprocessing", {}))
    raw = value["items"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_ITEMS:
        raise Rejected("invalid_batch")
    items, parts, text_bytes, transfer_bytes, has_audio = [], 0, 0, 0, False
    try:
        text_bytes = len(prompt.encode())
        for item in raw:
            content = closed(item, ("content",))["content"]
            if not isinstance(content, list) or not 1 <= len(content) <= MAX_PARTS_PER_ITEM:
                raise Rejected("invalid_item")
            typed = []
            for part in content:
                if not isinstance(part, dict):
                    raise Rejected("invalid_part")
                kind = part.get("type")
                parts += 1
                if kind == "text":
                    closed(part, ("type", "text"))
                    text = part["text"]
                    if not isinstance(text, str) or not text.strip():
                        raise Rejected("empty_content")
                    # Template interprets manual media markers; public parts own layout.
                    if any(marker in text for marker in ("<|image|>", "<|audio|>", "<|video|>")):
                        raise Rejected("reserved_placeholder")
                    text_bytes += len(text.encode())
                    typed.append(Part(kind, text=text))
                elif kind in ("image", "audio", "video"):
                    closed(part, ("type", "source"))
                    source = parse_source(part["source"], kind)
                    transfer_bytes += source.size
                    has_audio |= kind == "audio"
                    typed.append(Part(kind, source=source))
                else:
                    raise Rejected("invalid_part")
            items.append(Item(tuple(typed)))
    except UnicodeEncodeError:
        raise Rejected("invalid_unicode") from None
    if any(marker in prompt for marker in ("<|image|>", "<|audio|>", "<|video|>")):
        raise Rejected("reserved_placeholder")
    if parts > MAX_PARTS or text_bytes > MAX_CONTENT_BYTES or transfer_bytes > MAX_TRANSFER_BYTES:
        raise Rejected("content_limit", 413)
    if preprocessing["video"]["loadAudioFromVideo"] and has_audio:
        raise Rejected("incompatible_audio_options")
    return Batch(identifier, tuple(items), dimensions, prompt, prompt_name, preprocessing)


def validate_counts(counts):
    if any(type(count) is not int or count <= 0 or count > MAX_ITEM_TOKENS for count in counts):
        raise Rejected("item_token_limit")
    if sum(counts) > MAX_BATCH_TOKENS:
        raise Rejected("batch_token_limit")


def validate_vectors(vectors, expected, dimensions=DIMENSIONS):
    if not isinstance(vectors, list) or len(vectors) != expected:
        raise InferenceFailure()
    for row in vectors:
        if not isinstance(row, list) or len(row) != dimensions:
            raise InferenceFailure()
        if any(type(n) not in (float, int) or not math.isfinite(n) for n in row):
            raise InferenceFailure()
        if abs(math.sqrt(sum(n*n for n in row)) - 1) > .001:
            raise InferenceFailure()
    return vectors
