"""Synthetic upstream model seam: no torch import, tokenizer or real inference."""
from contextlib import nullcontext
import sys
from unittest.mock import patch
import veoveo_runtime_embedding.backend as backend_module
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

    def test_startup_accepts_native_unspecified_length_without_changing_it(self):
        # Synthetic loader/GPU only: this proves startup contract handling, not inference.
        sentinel = 1000000000000000019884624838656
        dtype = object()
        class SyntheticModel:
            max_seq_length = sentinel
            tokenizer = SimpleNamespace(model_max_length=sentinel)
            prompts = {"query": SEARCH_PREFIX, "CodeRetrieval": CODE_PREFIX, "document": "title: none | text: "}
            def float(self): return self
            def eval(self): return self
            def get_sentence_embedding_dimension(self): return self.dimension
            def parameters(self): return [SimpleNamespace(device=SimpleNamespace(type="mps"), dtype=dtype)]
            def __getitem__(self, index):
                if index == 1: return SimpleNamespace(include_prompt=True)
                return SimpleNamespace(auto_model=SimpleNamespace(register_forward_pre_hook=lambda *args, **kwargs: object()))
        packages = {"python":"3.14.6", "sentenceTransformers":"6.1.0", "transformers":"5.20.2",
                    "pillow":"13.0.0", "torch":"2.14.1", "torchvision":"0.29.1"}
        torch = SimpleNamespace(float32=dtype, backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        for dimension in (DIMENSIONS, DIMENSIONS - 1):
            model = SyntheticModel()
            model.dimension = dimension
            with patch.dict(sys.modules, {"torch": torch, "sentence_transformers": SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: model)}), \
                    patch.object(backend_module.platform, "system", return_value="Darwin"), \
                    patch.object(backend_module.platform, "machine", return_value="arm64"), \
                    patch.object(backend_module, "package_identity", return_value=packages), \
                    patch.dict(backend_module.os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK":"0"}):
                if dimension == DIMENSIONS:
                    loaded = ModelBackend("mps", "synthetic-source")
                    self.assertTrue(loaded.ready())
                    self.assertEqual(loaded.model.max_seq_length, sentinel)
                    self.assertEqual(loaded.model.tokenizer.model_max_length, sentinel)
                else:
                    with self.assertRaisesRegex(RuntimeError, "fixed text contract"):
                        ModelBackend("mps", "synthetic-source")

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
