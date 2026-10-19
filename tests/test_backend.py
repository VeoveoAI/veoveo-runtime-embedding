"""Synthetic upstream model seam: no torch import, tokenizer or real inference."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from uuid import uuid7
from veoveo_runtime_embedding.backend import ModelBackend
from veoveo_runtime_embedding.contract import *

class BackendTests(unittest.TestCase):
    def synthetic(self, counts):
        backend = ModelBackend.__new__(ModelBackend)
        backend.device, backend.runtime = "mps", {"syntheticFixture": True}
        backend.torch = SimpleNamespace(inference_mode=nullcontext)
        backend.ready = lambda: True
        counted, calls = [], []
        def count(text):
            counted.append(text)
            return counts[len(counted) - 1]
        def encode(rendered, **kwargs):
            # Every item must have passed token checks before the first model call.
            self.assertEqual(len(counted), len(rendered))
            calls.append((rendered, kwargs))
            backend._observed_forward = True
            rows = []
            for index, _ in enumerate(rendered):
                row = [0.0] * DIMENSIONS
                row[index] = 1.0
                rows.append(row)
            return SimpleNamespace(dtype="float32", tolist=lambda: rows)
        backend.model = SimpleNamespace(encode=encode)
        backend._count = count
        return backend, calls

    def test_mixed_lengths_use_native_two_item_microbatches_and_keep_order(self):
        counts = [8192] + [1] * 7  # synthetic token counts, not real tokenizer evidence
        backend, calls = self.synthetic(counts)
        items = (Item("search_query", "synthetic long"),) + tuple(Item("document", f"synthetic short {index}") for index in range(7))
        result = backend.encode(Batch(uuid7(), items))
        self.assertEqual(len(calls), 1)
        rendered, options = calls[0]
        self.assertEqual(options["batch_size"], 2)
        self.assertEqual(options["prompt"], "")
        self.assertFalse(options["processing_kwargs"]["text"]["truncation"])
        self.assertEqual(rendered, [item.render() for item in items])
        self.assertEqual([item["index"] for item in result["items"]], list(range(8)))
        self.assertEqual([item["tokenCount"] for item in result["items"]], counts)
        for index, item in enumerate(result["items"]):
            self.assertEqual(item["vector"][index], 1.0)
        self.assertEqual(result["usage"]["inputTokens"], 8199)

    def test_late_overflow_rejects_whole_batch_before_any_model_call(self):
        for counts in ([1, 8193], [8192, 8192, 1]):
            backend, calls = self.synthetic(counts)
            items = tuple(Item("code_query", f"synthetic {index}") for index in range(len(counts)))
            with self.assertRaises(Rejected): backend.encode(Batch(uuid7(), items))
            self.assertEqual(calls, [])

if __name__ == "__main__": unittest.main()
