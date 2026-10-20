"""Model-free closed contract tests; no tokenizer, decoder or inference."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import unittest
from uuid import uuid7
from veoveo_runtime_embedding.contract import *


def wire(items=None, **options):
    return {"requestId": str(uuid7()), "items": items or [{"content": [{"type": "text", "text": "hello again"}]}], **options}


def source(kind="image", **overrides):
    return {"artifactId": str(uuid7()), "mediaType": kind + "/fixture", "size": 3, "sha256": "sha256:" + "0"*64,
            "get": {"method": "GET", "url": "http://storage.test:9000/object?signature=synthetic-only",
                    "expiresAt": (datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat()}, **overrides}


def media(kind="image", **overrides):
    return {"type": kind, "source": source(kind, **overrides)}


class ContractTests(unittest.TestCase):
    def test_ordered_content_and_unprompted_default(self):
        batch = parse_batch(wire([{"content": [{"type":"text", "text":"first"}, media(), {"type":"text", "text":"last"}]}]))
        self.assertEqual([p.type for p in batch.items[0].content], ["text","image","text"])
        self.assertEqual(batch.prompt, "")
        self.assertEqual(batch.dimensions, 768)
        self.assertFalse(batch.preprocessing["audio"]["truncation"])
        self.assertNotIn("signature", repr(batch))
        self.assertNotIn("first", repr(batch))

    def test_dimensions_prompts_and_effective_processing_identity(self):
        base = parse_batch(wire())
        explicit = parse_batch(wire(dimensions=768, prompt="", preprocessing=copy.deepcopy(DEFAULT_PREPROCESSING)))
        self.assertEqual(base.processing_id(), explicit.processing_id())
        for name, prefix in PROMPTS.items():
            named = parse_batch(wire(promptName=name))
            self.assertEqual(named.prompt, prefix)
            self.assertEqual(named.processing_id(), parse_batch(wire(prompt=prefix)).processing_id())
            self.assertNotEqual(base.processing_id(), named.processing_id())
        for dimension in SUPPORTED_DIMENSIONS:
            self.assertEqual(parse_batch(wire(dimensions=dimension)).dimensions, dimension)
        for options in ({"dimensions":129}, {"dimensions":True}, {"prompt":None}, {"promptName":"query"}, {"promptName":"SearchQuery","prompt":""}):
            with self.assertRaises(Rejected): parse_batch(wire(**options))

    def test_frozen_jcs_processing_identity_for_rust_mirror(self):
        import rfc8785
        fixture = json.loads((Path(__file__).parent / "fixtures/processing-jcs.json").read_text(encoding="utf-8"))
        identities = []
        for case in fixture["cases"]:
            with self.subTest(case=case["name"]):
                canonical = case["canonicalUtf8"].encode("utf-8")
                self.assertEqual(rfc8785.dumps(case["descriptor"]), canonical)
                self.assertEqual("sha256:" + hashlib.sha256(canonical).hexdigest(), case["sha256"])
                descriptor = case["descriptor"]
                batch = parse_batch(wire(dimensions=descriptor["dimensions"], prompt=descriptor["prompt"],
                                         preprocessing=descriptor["preprocessing"]))
                self.assertEqual(batch.processing(), descriptor)
                self.assertEqual(batch.processing_id(), case["sha256"])
                identities.append(batch.processing_id())
        self.assertEqual(identities[0], identities[1])
        self.assertNotEqual(identities[0], identities[2])

    def test_closed_native_controls_and_explicit_truncation(self):
        options = {"image":{"maxSoftTokens":70}, "audio":{"truncation":True},
                   "video":{"fps":2,"maxFrames":8,"overflowStrategy":"truncate","maxSoftTokens":560,"addTimestamps":True}}
        batch = parse_batch(wire(preprocessing=options))
        self.assertEqual(batch.preprocessing["audio"]["maxLength"],480000)
        self.assertEqual(batch.native_kwargs()["image"]["max_soft_tokens"],70)
        self.assertFalse(batch.native_kwargs()["text"]["truncation"])
        for controls in ({"banana":{}}, {"image":{"max_soft_tokens":70}}, {"image":{"maxSoftTokens":71}},
                         {"audio":{"maxLength":1}}, {"video":{"fps":float("inf")}}, {"video":{"fps":121}},
                         {"video":{"maxFrames":257}}, {"video":{"addTimestamps":1}}, {"audio":{"truncation":None}}):
            with self.assertRaises(Rejected): parse_batch(wire(preprocessing=controls))
        with self.assertRaises(Rejected):
            parse_batch(wire([{"content":[media("audio")]}], preprocessing={"video":{"loadAudioFromVideo":True}}))

    def test_all_service_request_bounds_and_unknown_fields(self):
        invalid = [wire(items=[{"content":[]}]), wire(items=[{"content":[{"type":"text","text":"x"}]*17}]),
                   wire(items=[{"content":[media()]*16}]*3), wire(items=[{"content":[{"type":"text","text":"x"}]}]*9),
                   wire(items=[{"content":[{"type":"text","text":"é"*(MAX_CONTENT_BYTES//2+1)}]}]),
                   wire(items=[{"content":[{"type":"text","text":"\ud800"}]}]),
                   wire(items=[{"content":[{"type":"text","text":"<|image|>"}]}]), wire(prompt="<|video|>"),
                   wire(items=[{"content":[{"type":"image","artifactId":str(uuid7())}]}]),
                   wire(items=[{"content":[media(size=MAX_ARTIFACT_BYTES+1)]}]),
                   wire(items=[{"content":[media(size=MAX_ARTIFACT_BYTES)]}]*3), wire(unknown=True)]
        invalid.append({"requestId":str(uuid7()),"items":[]})
        for value in invalid:
            with self.subTest(value=str(value)[:100]):
                with self.assertRaises(Rejected): parse_batch(value)
        raw = wire()
        for change in ({"method":"POST"}, {"url":"http://user:pass@storage.test:9000/a"},
                       {"url":"http://storage.test:9000/a#secret"}, {"headers":{"X-Test":"x\r\nsecret"}}, {"expiresAt":"2026-01-01"}):
            part = media()
            part["source"]["get"].update(change)
            raw["items"] = [{"content":[part]}]
            with self.assertRaises(Rejected): parse_batch(raw)

    def test_count_and_vector_validation_are_whole_batch_and_dimension_aware(self):
        validate_counts([8192,8192])
        for counts in ([8193],[8192,8192,1],[0]):
            with self.assertRaises(Rejected): validate_counts(counts)
        for dimension in SUPPORTED_DIMENSIONS:
            row = [1.0]+[0.0]*(dimension-1)
            validate_vectors([row],1,dimension)
            for result in ([], [row[:-1]], [[0.0]*dimension], [[float("nan")]+row[1:]], [row,row]):
                with self.assertRaises(InferenceFailure): validate_vectors(result,1,dimension)

if __name__ == "__main__": unittest.main()
