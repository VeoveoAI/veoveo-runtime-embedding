"""Synthetic preprocessing fixtures; these do not run a tokenizer or model."""
import math
import unittest
from uuid import uuid7
from veoveo_runtime_embedding.contract import *


def wire(items):
    return {"requestId": str(uuid7()), "items": items}

class ContractTests(unittest.TestCase):
    def test_exact_upstream_prompts_and_mixed_order(self):
        batch = parse_batch(wire([
            {"purpose": "search_query", "text": "Mars"},
            {"purpose": "code_query", "text": "def add(a, b): return a+b"},
            {"purpose": "document", "content": "untitled"},
            {"purpose": "document", "content": "body", "title": "Heading"},
        ]))
        rendered, counts = prepare(batch, lambda text: len(text))
        self.assertEqual(rendered, [SEARCH_PREFIX + "Mars", CODE_PREFIX + "def add(a, b): return a+b", "title: none | text: untitled", "title: Heading | text: body"])
        self.assertEqual(counts, [len(text) for text in rendered])
        self.assertEqual(PROCESSING_ID, "sha256:e570e2bd489dbf11cf9da6de0e21b5ba06123d6a205c69aa1312495c3ac6e4a4")

    def test_byte_limits_count_utf8_and_titles(self):
        self.assertEqual(len(parse_batch(wire([{"purpose":"document", "content":"a" * MAX_CONTENT_BYTES}])).items), 1)
        for items in [[], [{"purpose":"search_query","text":"a"}]*9,
                      [{"purpose":"document","content":"a" * MAX_CONTENT_BYTES,"title":"b"}],
                      [{"purpose":"document","content":"é" * (MAX_CONTENT_BYTES//2+1)}],
                      [{"purpose":"document","content":"body","title":" "}],
                      [{"purpose":"search_query","text":"a","model":"override"}],
                      [{"purpose":"search_query","text":"\ud800"}]]:
            with self.assertRaises(Rejected): parse_batch(wire(items))

    def test_token_boundaries_reject_without_truncating_or_partial_batches(self):
        two = parse_batch(wire([{"purpose":"search_query","text":"synthetic"}]*2))
        rendered, counts = prepare(two, lambda _text: 8192)
        self.assertEqual(counts, [8192,8192])
        with self.assertRaises(Rejected): prepare(two, lambda _text: 8193)
        three = parse_batch(wire([{"purpose":"document","content":"synthetic"}]*3))
        with self.assertRaises(Rejected): prepare(three, lambda _text: 6000)
        self.assertEqual(len(rendered), 2)

    def test_vectors_require_complete_finite_normalized_768d_output(self):
        unit = [1.0]+[0.0]*767
        self.assertEqual(validate_vectors([unit],1), [unit])
        for rows in [[], [unit[:-1]], [[0.0]*768], [[float("nan")]+unit[1:]], [[float("inf")]+unit[1:]], [unit,unit]]:
            with self.assertRaises(InferenceFailure): validate_vectors(rows,1)

if __name__ == "__main__": unittest.main()
