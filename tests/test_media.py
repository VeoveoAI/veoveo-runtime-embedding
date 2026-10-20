"""Real loopback transfer tests and synthetic metadata/decoder seams; no inference."""
from contextlib import contextmanager
from datetime import datetime,timedelta,timezone
import hashlib
import io
import logging
from pathlib import Path
import threading
import sys
from tempfile import TemporaryDirectory
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from uuid import uuid7
from veoveo_runtime_embedding.contract import *
from veoveo_runtime_embedding.media import NativeMedia,fetch,resolve,pixels,duration
from test_contract import wire,media

BODY=b"synthetic transport fixture bytes; not valid encoded media"

class Handler(BaseHTTPRequestHandler):
    calls=[]
    def log_message(self,*args): pass
    def do_GET(self):
        self.calls.append(self.path)
        case=self.path.split("?")[0]
        try:
            if case=="/headerdrip":
                for byte in b"HTTP/1.0 200 OK\r\nContent-Length: 100\r\n\r\n":
                    self.wfile.write(bytes([byte]));self.wfile.flush();time.sleep(.02)
                return
            self.send_response(302 if case=="/redirect" else 200)
            if case=="/redirect":self.send_header("Location","http://forbidden.invalid/private")
            if case=="/encoding":self.send_header("Content-Encoding","gzip")
            if case!="/overflow":self.send_header("Content-Length",str(len(BODY)-1 if case=="/length" else len(BODY)))
            self.end_headers()
            data=BODY+(b"extra" if case=="/overflow" else b"")
            if case=="/slow":
                for byte in data:
                    self.wfile.write(bytes([byte]));self.wfile.flush();time.sleep(.02)
            else:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.server.daemon_threads=True
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.port=cls.server.server_port
        cls.origin=("http","127.0.0.1",cls.port)
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def source(self,case="good",**changes):
        transfer=Transfer(f"http://127.0.0.1:{self.port}/{case}?X-Amz-Signature=synthetic-private",{},datetime.now(timezone.utc)+timedelta(minutes=5))
        return Source(uuid7(),"image/fixture",len(BODY),"sha256:"+hashlib.sha256(BODY).hexdigest(),transfer,**changes)
    def test_actual_bytes_digest_redirect_encoding_size_and_expiry(self):
        with TemporaryDirectory() as directory:
            source=self.source()
            self.assertEqual(fetch(source,directory,0,self.origin,time.monotonic()+3).read_bytes(),BODY)
            for index,(case,code) in enumerate((("redirect","source_redirect"),("length","source_size"),("overflow","source_size"),("encoding","source_encoding")),1):
                with self.subTest(case=case):
                    with self.assertRaisesRegex(Rejected,code):fetch(self.source(case),directory,index,self.origin,time.monotonic()+3)
            wrong=Source(source.artifact_id,source.media_type,source.size,"sha256:"+"0"*64,source.get)
            with self.assertRaisesRegex(Rejected,"source_digest"):fetch(wrong,directory,8,self.origin,time.monotonic()+3)
            expired=Source(source.artifact_id,source.media_type,source.size,source.sha256,Transfer(source.get.url,{},datetime.now(timezone.utc)-timedelta(seconds=1)))
            with self.assertRaisesRegex(Rejected,"source_expired"):fetch(expired,directory,9,self.origin,time.monotonic()+3)
            badhost=Source(source.artifact_id,source.media_type,source.size,source.sha256,Transfer(source.get.url,{"Host":"forbidden.invalid"},source.get.expires_at))
            with self.assertRaisesRegex(Rejected,"source_origin"):fetch(badhost,directory,10,self.origin,time.monotonic()+3)
            with self.assertRaisesRegex(Rejected,"source_origin"):fetch(source,directory,11,("http","other.invalid",80),time.monotonic()+3)
    def test_total_deadline_stops_slow_body_and_slow_headers(self):
        with TemporaryDirectory() as directory:
            for index,case in enumerate(("slow","headerdrip")):
                start=time.monotonic()
                with self.assertRaisesRegex(Rejected,"source_timeout"):fetch(self.source(case),directory,index,self.origin,start+.12)
                self.assertLess(time.monotonic()-start,1)
    def test_proxy_environment_is_ignored_and_dependency_logs_are_suppressed(self):
        stream=io.StringIO();handler=logging.StreamHandler(stream);root=logging.getLogger()
        previous=root.level;root.setLevel(logging.DEBUG);root.addHandler(handler)
        logging.getLogger("httpcore.http11").setLevel(logging.DEBUG)
        try:
            with TemporaryDirectory() as directory, patch.dict("os.environ",{"HTTP_PROXY":"http://127.0.0.1:1","HTTPS_PROXY":"http://127.0.0.1:1","ALL_PROXY":"http://127.0.0.1:1","NO_PROXY":""}):
                self.assertEqual(fetch(self.source(),directory,0,self.origin,time.monotonic()+3).read_bytes(),BODY)
                with self.assertRaises(Rejected):fetch(self.source("redirect"),directory,1,self.origin,time.monotonic()+3)
            logging.getLogger("httpx").info("synthetic-private")
            logging.getLogger("httpcore.http11").debug("synthetic-private")
            self.assertNotIn("synthetic-private",stream.getvalue())
            self.assertNotIn("X-Amz",stream.getvalue())
        finally:root.removeHandler(handler);root.setLevel(previous)


class MediaAdmissionTests(unittest.TestCase):
    def test_video_coverage_uses_native_sampler_and_distinguishes_actual_pts(self):
        calls=[]
        meta=SimpleNamespace(duration_seconds=2.0,width=192,height=128,num_frames=8,average_fps=4.0)
        def get_frames_at(*,indices):
            calls.append(("decode",indices))
            return SimpleNamespace(data=SimpleNamespace(shape=(2,3,128,192)),pts_seconds=SimpleNamespace(tolist=lambda:[0.0,1.77]))
        def sample(metadata,**kwargs):
            calls.append(("sample",metadata,kwargs))
            return [0,7]
        native=NativeMedia(SimpleNamespace(video_processor=SimpleNamespace(sample_frames=sample)))
        modules={"torchcodec.decoders":SimpleNamespace(VideoDecoder=lambda *args,**kwargs:SimpleNamespace(metadata=meta,get_frames_at=get_frames_at)),
                 "transformers.video_utils":SimpleNamespace(VideoMetadata=lambda **kwargs:SimpleNamespace(**kwargs))}
        batch=parse_batch(wire([{"content":[media("video")]}]))
        with patch.dict(sys.modules,modules):
            plan=native.inspect(batch.items[0].content[0],Path("synthetic.mp4"),batch.preprocessing)
            self.assertEqual(len(calls),1)
            self.assertEqual(plan.pixels,2*192*128)
            native.validate_decoded(plan,batch.preprocessing)
        self.assertEqual(calls[0][2],{"fps":1,"max_frames":32,"overflow_strategy":"uniform"})
        self.assertEqual(plan.coverage["frameIndices"],[0,7])
        self.assertEqual(plan.coverage["frameTimestampsSeconds"],[0.0,1.77])
        self.assertEqual(plan.coverage["modelFrameTimestampsSeconds"],[0.0,1.75])
        self.assertIsNone(plan.decoder)

    def test_all_metadata_limits_precede_every_decode(self):
        batch=parse_batch(wire([{"content":[media(),media()]}]))
        events=[]
        class Native:
            def inspect(self,part,path,options):
                events.append("header")
                return SimpleNamespace(kind=part.type,pixels=MAX_TOTAL_PIXELS//2+1,audio_seconds=0,path=path,coverage={})
            def validate_decoded(self,*args):events.append("decode")
        with TemporaryDirectory() as directory,patch("veoveo_runtime_embedding.media.fetch",side_effect=lambda s,d,i,o,t:Path(d)/str(i)):
            with self.assertRaisesRegex(Rejected,"media_pixels"):resolve(batch,directory,("http","fixture",80),Native())
        self.assertEqual(events,["header","header"])
        for value in (0,-1,float("nan"),301,None):
            with self.assertRaises(Rejected):duration(value)
        for size in ((0,3),(True,2),(4001,4000),(None,1)):
            with self.assertRaises(Rejected):pixels(*size)
    def test_resolved_messages_have_explicit_batch_and_safe_source_metadata(self):
        batch=parse_batch(wire([{"content":[{"type":"text","text":"before"},media(),{"type":"text","text":"after"}]}]))
        class Native:
            def inspect(self,part,path,options):return SimpleNamespace(kind=part.type,pixels=1,audio_seconds=0,coverage={"width":1,"height":1,"frameIndex":0})
            def validate_decoded(self,*args):pass
        with TemporaryDirectory() as directory,patch("veoveo_runtime_embedding.media.fetch",return_value=Path("/tmp/synthetic-media")):
            conversations,rows=resolve(batch,directory,("http","fixture",80),Native())
        self.assertEqual(len(conversations),1)
        self.assertIsInstance(conversations[0],list)
        self.assertEqual([p["type"] for p in conversations[0][0]["content"]],["text","image","text"])
        self.assertEqual(rows[0][0]["partIndex"],1)
        self.assertNotIn("get",rows[0][0]["source"])
        self.assertNotIn("synthetic-only",str(rows))

    def test_actual_audio_aggregate_stops_decode_before_preflight_or_forward(self):
        """Synthetic underreported metadata; actual sample counts, no buffers/GPU."""
        from veoveo_runtime_embedding.backend import ModelBackend
        for kind in ("audio","video"):
            for exceed in (False,True):
                with self.subTest(kind=kind,exceed=exceed):
                    counts=[300*16000,300*16000]+([1,1] if exceed else [])
                    events=[]
                    class Native:
                        def inspect(self,part,path,options):
                            coverage={"sourceDurationSeconds":1.0}
                            if kind=="video":coverage={"audioIncluded":True,"audio":coverage}
                            return SimpleNamespace(kind=kind,pixels=0,audio_seconds=1.0,coverage=coverage)
                        def validate_decoded(self,plan,options):
                            events.append("decode")
                            coverage=plan.coverage if kind=="audio" else plan.coverage["audio"]
                            coverage["decodedSamples"]=counts[len(events)-1]
                    batch=parse_batch(wire([{"content":[media(kind) for _ in counts]}],
                        preprocessing={"video":{"loadAudioFromVideo":kind=="video"}}))
                    native=Native()
                    with patch("veoveo_runtime_embedding.media.fetch",side_effect=lambda s,d,i,o,t:Path(d)/str(i)):
                        if exceed:
                            backend=ModelBackend.__new__(ModelBackend)
                            backend.storage_origin=("http","fixture",80);backend.media=native
                            backend.model=SimpleNamespace(preprocess=Mock(),encode=Mock())
                            with self.assertRaisesRegex(Rejected,"media_duration"):backend.encode(batch)
                            self.assertEqual(events,["decode"]*3,"the fourth source must not decode")
                            backend.model.preprocess.assert_not_called()
                            backend.model.encode.assert_not_called()
                        else:
                            with TemporaryDirectory() as directory:
                                _,rows=resolve(batch,directory,("http","fixture",80),native)
                            self.assertEqual(events,["decode"]*2)
                            audio=[row["coverage"] if kind=="audio" else row["coverage"]["audio"] for row in rows[0]]
                            self.assertEqual(sum(value["decodedSamples"] for value in audio),600*16000)

if __name__=="__main__":unittest.main()
