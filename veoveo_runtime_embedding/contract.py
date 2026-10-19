"""Fixed text-only processing contract. No model packages are required to validate inputs."""
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable
from uuid import UUID

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIMENSIONS = 768
MAX_ITEMS = 8
MAX_CONTENT_BYTES = 65536
MAX_ITEM_TOKENS = 8192
MAX_BATCH_TOKENS = 16384
MAX_JSON_BYTES = 524288
MAX_RESPONSE_BYTES = 1048576
REQUEST_SECONDS = 60
PURPOSES = ("search_query", "code_query", "document")
SEARCH_PREFIX = "task: search result | query: "
CODE_PREFIX = "task: code retrieval | query: "
PROCESSING = {
    "modelId": MODEL_ID, "modelRevision": MODEL_REVISION,
    "dimensions": DIMENSIONS, "normalized": True,
    "pooling": "upstream_mean_projection_normalize",
    "searchQueryPrefix": SEARCH_PREFIX, "codeQueryPrefix": CODE_PREFIX,
    "documentTemplate": "title: {title_or_none} | text: {content}", "textOnly": True,
}
PROCESSING_ID = "sha256:" + hashlib.sha256(json.dumps(PROCESSING, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

class Rejected(Exception):
    """Bounded public rejection code; never embeds user content or upstream exceptions."""
    def __init__(self, code: str, status: int = 422):
        self.code = code
        self.status = status
        super().__init__(code)

class InferenceFailure(Exception):
    pass

@dataclass(frozen=True, slots=True)
class Item:
    purpose: str
    content: str
    title: str | None = None

    def render(self) -> str:
        if self.purpose == "search_query":
            return SEARCH_PREFIX + self.content
        if self.purpose == "code_query":
            return CODE_PREFIX + self.content
        return f"title: {self.title if self.title is not None else 'none'} | text: {self.content}"

@dataclass(frozen=True, slots=True)
class Batch:
    request_id: UUID
    items: tuple[Item, ...]


def parse_batch(value: object) -> Batch:
    if not isinstance(value, dict) or set(value) != {"requestId", "items"}:
        raise Rejected("invalid_request")
    try:
        identifier = UUID(value["requestId"])
        if identifier.version != 7 or str(identifier) != value["requestId"]:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise Rejected("invalid_request") from None
    raw = value["items"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_ITEMS:
        raise Rejected("invalid_batch")
    items = []
    byte_count = 0
    for item in raw:
        if not isinstance(item, dict) or item.get("purpose") not in PURPOSES:
            raise Rejected("invalid_purpose")
        purpose = item["purpose"]
        allowed = {"purpose", "content", "title"} if purpose == "document" else {"purpose", "text"}
        required = {"purpose", "content"} if purpose == "document" else allowed
        if not required <= set(item) <= allowed:
            raise Rejected("invalid_item")
        content = item["content" if purpose == "document" else "text"]
        title = item.get("title")
        if not isinstance(content, str) or not content.strip():
            raise Rejected("empty_content")
        if title is not None and (not isinstance(title, str) or not title.strip()):
            raise Rejected("invalid_title")
        try:
            byte_count += len(content.encode("utf-8")) + (len(title.encode("utf-8")) if title is not None else 0)
        except UnicodeEncodeError:
            raise Rejected("invalid_unicode") from None
        if byte_count > MAX_CONTENT_BYTES:
            raise Rejected("content_limit", 413)
        items.append(Item(purpose, content, title))
    return Batch(identifier, tuple(items))


def prepare(batch: Batch, count_tokens: Callable[[str], int]) -> tuple[list[str], list[int]]:
    rendered = [item.render() for item in batch.items]
    counts = [count_tokens(text) for text in rendered]
    if any(type(count) is not int or count <= 0 or count > MAX_ITEM_TOKENS for count in counts):
        raise Rejected("item_token_limit")
    if sum(counts) > MAX_BATCH_TOKENS:
        raise Rejected("batch_token_limit")
    return rendered, counts


def validate_vectors(vectors: object, expected: int) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != expected:
        raise InferenceFailure()
    for row in vectors:
        if not isinstance(row, list) or len(row) != DIMENSIONS:
            raise InferenceFailure()
        if any(type(number) not in (float, int) or not math.isfinite(number) for number in row):
            raise InferenceFailure()
        if abs(math.sqrt(sum(number * number for number in row)) - 1.0) > 0.001:
            raise InferenceFailure()
    return vectors
