"""Synthetic native model seam; no torch, decoder or real embeddings."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys
import asyncio
import threading
from uuid import uuid7
import veoveo_runtime_embedding.backend as module
from veoveo_runtime_embedding.backend import ModelBackend
from veoveo_runtime_embedding.contract import *
from test_contract import wire


class Tensor:
    def __init__(self, count): self.shape, self.count = (1,count), count
    def sum(self): return SimpleNamespace(item=lambda:self.count)


class BackendTests(unittest.TestCase):
    def test_startup_keeps_full_towers_native_lengths_and_pinned_gpu(self):
        dtype=object()
        sentinel=1000000000000000019884624838656
        class Model:
            max_seq_length=sentinel
            prompts=PROMPTS.copy()
            def float(self):return self
            def eval(self):return self
            def get_sentence_embedding_dimension(self):return 768
            def parameters(self):return [SimpleNamespace(device=SimpleNamespace(type="mps"),dtype=dtype)]
            def __getitem__(self,index):
                if index==1:return SimpleNamespace(include_prompt=True)
                return SimpleNamespace(auto_model=SimpleNamespace(vision_tower=object(),audio_tower=object()),processor=SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000)))
        calls=[]
        def load(*args,**kwargs):
            calls.append(kwargs)
            return Model()
        packages={"python":"3.14.6","sentenceTransformers":"6.1.0","transformers":"5.20.2","pillow":"13.0.0","torch":"2.14.1","torchvision":"0.29.1","torchcodec":"0.17.0"}
        torch=SimpleNamespace(float32=dtype,backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:True)))
        with patch.dict(sys.modules,{"torch":torch,"sentence_transformers":SimpleNamespace(SentenceTransformer=load)}), patch.object(module.platform,"system",return_value="Darwin"), patch.object(module.platform,"machine",return_value="arm64"), patch.object(module,"package_identity",return_value=packages), patch.dict(module.os.environ,{"VEOVEO_EMBEDDING_STORAGE_ORIGIN":"http://storage.test:9000","PYTORCH_ENABLE_MPS_FALLBACK":"0"}):
            loaded=ModelBackend("mps","synthetic-source")
        self.assertEqual(loaded.model.max_seq_length,sentinel)
        self.assertNotIn("config_kwargs",calls[0])
        self.assertTrue(calls[0]["local_files_only"])
        self.assertEqual(calls[0]["revision"],MODEL_REVISION)
        self.assertEqual(calls[0]["model_kwargs"],{"dtype":dtype})

    def synthetic(self, counts, dtype="float32", mutate=False):
        backend = ModelBackend.__new__(ModelBackend)
        backend.device, backend.runtime = "mps", {"syntheticFixture":True}
        backend.storage_origin, backend.media = ("http","storage.test",9000), object()
        backend.torch = SimpleNamespace(inference_mode=nullcontext)
        backend.ready = lambda:True
        calls, preflight = [], []
        def preprocess(conversations, **kwargs):
            # Explicit [conversation] must have one row, regardless of N=1.
            self.assertEqual(len(conversations),1)
            self.assertIsInstance(conversations[0],list)
            self.assertEqual(conversations[0][0]["role"],"user")
            self.assertFalse(kwargs["processing_kwargs"]["text"]["truncation"])
            preflight.append(conversations)
            if mutate: conversations[0][0]["content"].append({"type":"audio"})
            tensor = Tensor(counts[len(preflight)-1])
            return {"input_ids":tensor,"attention_mask":tensor}
        def encode(conversations, **kwargs):
            self.assertEqual(len(preflight),len(conversations))
            self.assertTrue(all(len(c)==1 and c[0]["role"]=="user" for c in conversations))
            self.assertTrue(all(len(c[0]["content"])==1 for c in conversations))
            calls.append((conversations,kwargs))
            rows=[]
            for index in range(len(conversations)):
                row=[0.0]*kwargs["truncate_dim"]
                row[index]=1.0
                rows.append(row)
            return SimpleNamespace(dtype=dtype,tolist=lambda:rows)
        backend.model=SimpleNamespace(encode=encode,preprocess=preprocess)
        return backend,calls

    def test_all_preflight_before_native_batch_one_and_no_message_mutation(self):
        backend,calls=self.synthetic([8192,8192],mutate=True)
        batch=parse_batch(wire([{"content":[{"type":"text","text":"first"}]},{"content":[{"type":"text","text":"second"}]}],dimensions=256))
        output=backend.encode(batch)
        self.assertEqual(len(calls),1)
        conversations,kwargs=calls[0]
        self.assertEqual([c[0]["content"][0]["text"] for c in conversations],["first","second"])
        self.assertEqual(kwargs["batch_size"],1)
        self.assertEqual(kwargs["device"],"mps")
        self.assertEqual(kwargs["prompt"],"")
        self.assertEqual(output["dimensions"],256)
        self.assertEqual([row["index"] for row in output["items"]],[0,1])
        self.assertEqual([row["vector"][i] for i,row in enumerate(output["items"])],[1,1])
        self.assertEqual(output["usage"]["inputTokens"],16384)
        self.assertNotIn("promptName",output["processing"])

    def test_single_item_is_also_a_batch_of_conversations(self):
        backend,calls=self.synthetic([10])
        backend.encode(parse_batch(wire()))
        self.assertEqual(len(calls[0][0]),1)
        self.assertIsInstance(calls[0][0][0],list)

    def test_late_overflow_rejects_before_any_inference(self):
        for counts in ([1,8193],[8192,8192,1]):
            backend,calls=self.synthetic(counts)
            batch=parse_batch(wire([{"content":[{"type":"text","text":"synthetic"}]}]*len(counts)))
            with self.assertRaises(Rejected): backend.encode(batch)
            self.assertEqual(calls,[])

    def test_failures_preserve_execution_boundary_and_cleanup_files(self):
        backend,calls=self.synthetic([1],dtype="float16")
        captured=[]
        original=module.resolve
        def resolved(batch,directory,origin,native):
            captured.append(Path(directory)/"temporary")
            captured[-1].write_bytes(b"synthetic transient media")
            return original(batch,directory,origin,native)
        with patch.object(module,"resolve",side_effect=resolved):
            with self.assertRaises(InferenceFailure): backend.encode(parse_batch(wire()))
        self.assertEqual(len(calls),1)
        self.assertFalse(captured[0].exists())
        with patch.object(module,"resolve",side_effect=Rejected("source_digest")):
            with self.assertRaisesRegex(Rejected,"source_digest"): backend.encode(parse_batch(wire()))
        with patch.object(module,"resolve",side_effect=ValueError("private details must not echo")):
            with self.assertRaisesRegex(Rejected,"preprocessing_failed"): backend.encode(parse_batch(wire()))

    def test_preflight_cardinality_failure_is_confirmed_not_started(self):
        backend,calls=self.synthetic([1])
        backend.model.preprocess=lambda *a,**kw:{"input_ids":SimpleNamespace(shape=(2,1)),"attention_mask":Tensor(1)}
        with self.assertRaisesRegex(Rejected,"preprocessing_cardinality"):backend.encode(parse_batch(wire()))
        self.assertEqual(calls,[])

    def test_readiness_requires_all_full_model_parameters_on_fp32_gpu(self):
        backend=ModelBackend.__new__(ModelBackend)
        backend.device="mps"
        backend.torch=SimpleNamespace(float32="fp32")
        def parameter(device="mps",dtype="fp32"):return SimpleNamespace(device=SimpleNamespace(type=device),dtype=dtype)
        for values,want in [([parameter(),parameter()],True),([],False),([parameter(),parameter("cpu")],False),([parameter(dtype="fp16")],False)]:
            backend.model=SimpleNamespace(parameters=lambda:values)
            self.assertEqual(backend.ready(),want)

class WorkerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_temporary_media_and_slot_survive_caller_timeout_and_cancel(self):
        from veoveo_runtime_embedding.app import Admission
        backend,_=BackendTests.synthetic(self,[1])
        started,release=threading.Event(),threading.Event()
        original_encode=backend.model.encode
        original_resolve=module.resolve
        paths=[]
        def resolved(batch,directory,origin,native):
            paths.append(Path(directory)/"synthetic-transient-media")
            paths[-1].write_bytes(b"synthetic fixture")
            return original_resolve(batch,directory,origin,native)
        def blocked(*args,**kwargs):
            started.set()
            release.wait(2)
            return original_encode(*args,**kwargs)
        backend.model.encode=blocked
        admission=Admission(backend)
        batch=parse_batch(wire())
        with patch.object(module,"resolve",side_effect=resolved):
            worker=admission.submit(batch)
            try:
                await asyncio.to_thread(started.wait,1)
                with self.assertRaises(TimeoutError):await asyncio.wait_for(asyncio.shield(worker),.01)
                self.assertTrue(paths[0].exists())
                with self.assertRaises(Rejected):admission.submit(batch)
                async def caller():return await asyncio.shield(worker)
                pending=asyncio.create_task(caller())
                await asyncio.sleep(0)
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):await pending
                self.assertTrue(paths[0].exists())
            finally:
                release.set()
                await worker
        self.assertFalse(paths[0].exists())

if __name__ == "__main__":unittest.main()
