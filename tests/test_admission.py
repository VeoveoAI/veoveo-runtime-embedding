"""Synthetic backend/real threading and ASGI; no GPU or model inference."""
import asyncio
import json
import threading
import unittest
from uuid import uuid7
from veoveo_runtime_embedding.app import Admission, create_app
from veoveo_runtime_embedding.contract import *

class SyntheticBackend:
    def __init__(self, block=False, reject=False, failure=False):
        self.started = threading.Event()
        self.release = threading.Event()
        self.block, self.reject, self.failure = block, reject, failure
        self.calls = 0
    def ready(self): return True
    def capabilities(self): return {"syntheticFixture": True}
    def encode(self, batch):
        self.calls += 1
        self.started.set()
        if self.block: self.release.wait(2)
        if self.reject: raise Rejected("item_token_limit")
        if self.failure: raise RuntimeError("private text and vectors MUST NOT ECHO")
        return {"requestId":str(batch.request_id),"syntheticFixture":True}

async def request(app, body, token="synthetic-only-bearer", path="/embeddings", method="POST"):
    sent = []
    first = True
    async def receive():
        nonlocal first
        if first:
            first = False
            return {"type":"http.request", "body":body, "more_body":False}
        await asyncio.Event().wait()
    async def send(message): sent.append(message)
    headers = [(b"content-type",b"application/json"),(b"content-length",str(len(body)).encode())]
    if token: headers.append((b"authorization",("Bearer "+token).encode()))
    scope = {"type":"http", "asgi":{"version":"3.0","spec_version":"2.4"}, "http_version":"1.1",
             "method":method,"scheme":"http","path":path,"raw_path":path.encode(),"query_string":b"",
             "root_path":"","server":("localhost",8804),"client":("127.0.0.1",1234),"headers":headers}
    await asyncio.wait_for(app(scope,receive,send),1)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    raw = b"".join(message.get("body",b"") for message in sent if message["type"] == "http.response.body")
    return status,raw

def raw_request():
    return json.dumps({"requestId":str(uuid7()),"items":[{"content":[{"type":"text","text":"synthetic"}]}]}).encode()

class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_and_cancellation_do_not_release_live_gpu_slot(self):
        backend = SyntheticBackend(block=True)
        admission = Admission(backend)
        batch = parse_batch(json.loads(raw_request()))
        worker = admission.submit(batch)
        await asyncio.to_thread(backend.started.wait,1)
        try:
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(worker),.01)
            with self.assertRaises(Rejected): admission.submit(batch)
            async def caller(): return await asyncio.shield(worker)
            caller_task = asyncio.create_task(caller())
            await asyncio.sleep(0)
            caller_task.cancel()
            with self.assertRaises(asyncio.CancelledError): await caller_task
            self.assertFalse(worker.done())
            with self.assertRaises(Rejected): admission.submit(batch)
        finally:
            backend.release.set()
            await worker
        await admission.submit(batch)
        self.assertEqual(backend.calls,2)

    async def test_auth_rejections_are_safe_and_token_overflow_is_not_started(self):
        backend = SyntheticBackend(reject=True)
        app = create_app(lambda: backend,"synthetic-only-bearer")
        async with app.router.lifespan_context(app):
            status,raw = await request(app,raw_request(),token="wrong")
            self.assertEqual(status,401)
            self.assertEqual(backend.calls,0)
            status,raw = await request(app,raw_request())
            self.assertEqual(status,422)
            self.assertEqual(json.loads(raw)["execution"],"not_started")
            private = b'{"requestId":"private text MUST NOT ECHO","items":[]}'
            status,raw = await request(app,private)
            self.assertNotIn(b"private text",raw)
            self.assertEqual(json.loads(raw)["execution"],"not_started")
            status,raw = await request(app,b"x" * (MAX_JSON_BYTES+1))
            self.assertEqual(status,413)

    async def test_runtime_failure_and_timeout_remain_ambiguous_without_error_echo(self):
        for backend,deadline,expected in [(SyntheticBackend(failure=True),1,500),(SyntheticBackend(block=True),.01,504)]:
            app = create_app(lambda:backend,"synthetic-only-bearer",deadline)
            try:
                async with app.router.lifespan_context(app):
                    status,raw = await request(app,raw_request())
                    self.assertEqual(status,expected)
                    self.assertEqual(json.loads(raw)["execution"],"unknown")
                    self.assertNotIn(b"private text",raw)
                    if backend.block:
                        status,_ = await request(app,raw_request())
                        self.assertEqual(status,429)
            finally:
                backend.release.set()
                await asyncio.sleep(.02)

if __name__ == "__main__": unittest.main()
