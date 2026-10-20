"""Private authorized transfers and native media admission. No MCP/Hub identity."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from urllib.parse import urlsplit
from .contract import *


def suppress_transfer_logs():
    # Children otherwise propagate past a disabled parent. Existing children are
    # disabled too; future children inherit the above-CRITICAL effective level.
    names = {"httpx", "httpcore", *(name for name in logging.Logger.manager.loggerDict if name.startswith(("httpx.", "httpcore.")))}
    for name in names:
        logger = logging.getLogger(name)
        logger.disabled = True
        logger.setLevel(logging.CRITICAL + 1)


def fetch(source, directory, index, origin, deadline):
    """Bounded async native HTTP; worker owns this loop and its temporary files."""
    import asyncio
    import httpx
    import time
    suppress_transfer_logs()
    parsed = urlsplit(source.get.url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if (parsed.scheme, parsed.hostname.lower(), port) != origin:
        raise Rejected("source_origin")
    for name, value in source.get.headers.items():
        if name.lower() == "host" and value.lower() != parsed.netloc.lower():
            raise Rejected("source_origin")
    if source.get.expires_at <= datetime.now(timezone.utc):
        raise Rejected("source_expired")
    async def transfer():
        remaining = deadline - time.monotonic()
        if remaining <= 0: raise Rejected("source_timeout")
        path = Path(directory) / f"media-{index}"
        async with asyncio.timeout(remaining):
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=min(15, remaining)) as client:
                async with client.stream("GET", source.get.url, headers=source.get.headers) as response:
                    if response.status_code != 200:
                        raise Rejected("source_redirect" if 300 <= response.status_code < 400 else "source_refused")
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise Rejected("source_encoding")
                    length = response.headers.get("content-length")
                    if length is not None and (not length.isdecimal() or int(length) != source.size):
                        raise Rejected("source_size")
                    read = 0
                    with path.open("xb") as output:
                        async for chunk in response.aiter_raw():
                            if time.monotonic() >= deadline: raise Rejected("source_timeout")
                            read += len(chunk)
                            if read > source.size: raise Rejected("source_size")
                            output.write(chunk)
                    if read != source.size: raise Rejected("source_size")
        return path
    try:
        return asyncio.run(transfer())
    except Rejected:
        raise
    except (TimeoutError, httpx.TimeoutException):
        raise Rejected("source_timeout") from None
    except Exception:
        raise Rejected("source_unavailable") from None


def duration(value):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= MAX_SOURCE_SECONDS:
        raise Rejected("media_duration")
    return float(value)


def pixels(width, height):
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0 or width * height > MAX_PIXELS:
        raise Rejected("media_pixels")
    return width * height


@dataclass
class MediaPlan:
    source: Source
    kind: str
    path: Path = field(repr=False)
    coverage: dict
    pixels: int = 0
    audio_seconds: float = 0
    decoder: object = field(default=None, repr=False)
    indices: object = field(default=None, repr=False)
    audio_decoder: object = field(default=None, repr=False)


class NativeMedia:
    """Use the pinned processor's sampler and TorchCodec metadata before decoding."""
    def __init__(self, processor):
        self.processor = processor

    def _audio(self, path):
        from torchcodec.decoders import AudioDecoder
        try:
            decoder = AudioDecoder(str(path), sample_rate=16000, num_channels=1)
        except Exception:
            raise Rejected("missing_audio_stream") from None
        meta = decoder.metadata
        seconds = duration(meta.duration_seconds)
        if type(meta.sample_rate) is not int or meta.sample_rate <= 0 or type(meta.num_channels) is not int or meta.num_channels <= 0:
            raise Rejected("media_metadata")
        return decoder, {"sourceDurationSeconds": seconds, "sourceSamplingRate": meta.sample_rate,
                         "sourceChannels": meta.num_channels, "samplingRate": 16000}

    def inspect(self, part, path, options):
        source = part.source
        if part.type == "image":
            from PIL import Image
            with Image.open(path) as image:
                width, height = image.size
                count = pixels(width, height)
            return MediaPlan(source, part.type, path, {"width": width, "height": height, "frameIndex": 0}, pixels=count)
        if part.type == "audio":
            decoder, coverage = self._audio(path)
            return MediaPlan(source, part.type, path, coverage, audio_seconds=coverage["sourceDurationSeconds"], audio_decoder=decoder)
        from torchcodec.decoders import VideoDecoder
        from transformers.video_utils import VideoMetadata
        decoder = VideoDecoder(str(path), seek_mode="exact", device="cpu", num_ffmpeg_threads=0)
        meta = decoder.metadata
        seconds = duration(meta.duration_seconds)
        per_frame = pixels(meta.width, meta.height)
        if (type(meta.num_frames) is not int or meta.num_frames <= 0 or type(meta.average_fps) not in (float,int)
                or not math.isfinite(meta.average_fps) or meta.average_fps <= 0):
            raise Rejected("media_metadata")
        metadata = VideoMetadata(total_num_frames=meta.num_frames, fps=meta.average_fps, duration=seconds,
                                 width=meta.width, height=meta.height, video_backend="torchcodec")
        video_options = options["video"]
        # This is the native pinned sampler, not a replacement implementation.
        indices = self.processor.video_processor.sample_frames(metadata, fps=video_options["fps"],
                    max_frames=video_options["maxFrames"], overflow_strategy=video_options["overflowStrategy"])
        if not 1 <= len(indices) <= video_options["maxFrames"] or any(not 0 <= int(i) < meta.num_frames for i in indices):
            raise Rejected("media_metadata")
        coverage = {"sourceDurationSeconds": seconds, "sourceFrameCount": meta.num_frames, "sourceFps": float(meta.average_fps),
                    "width": meta.width, "height": meta.height, "frameIndices": [int(i) for i in indices],
                    "modelFrameTimestampsSeconds": [float(i)/meta.average_fps for i in indices],
                    "audioIncluded": video_options["loadAudioFromVideo"]}
        plan = MediaPlan(source, part.type, path, coverage, pixels=per_frame * len(indices), decoder=decoder, indices=indices)
        if video_options["loadAudioFromVideo"]:
            plan.audio_decoder, coverage["audio"] = self._audio(path)
            plan.audio_seconds = coverage["audio"]["sourceDurationSeconds"]
        return plan

    def validate_decoded(self, plan, options):
        """Bounded native decode records actual PTS/samples. ST later decodes paths again."""
        if plan.kind == "image":
            from PIL import Image
            with Image.open(plan.path) as image:
                image.load()
                if pixels(*image.size) != plan.pixels: raise Rejected("media_metadata")
        elif plan.kind == "video":
            frames = plan.decoder.get_frames_at(indices=plan.indices)
            shape = frames.data.shape
            if (len(shape) != 4 or shape[0] != len(plan.indices) or shape[1] != 3
                    or shape[0] * pixels(shape[3], shape[2]) != plan.pixels):
                raise Rejected("media_metadata")
            pts = frames.pts_seconds.tolist()
            if len(pts) != len(plan.indices) or any(not math.isfinite(t) for t in pts):
                raise Rejected("media_metadata")
            plan.coverage["frameTimestampsSeconds"] = pts
        if plan.audio_decoder is not None:
            samples = plan.audio_decoder.get_all_samples()
            shape = samples.data.shape
            if len(shape) != 2 or shape[0] != 1 or not 0 < shape[1] <= MAX_SOURCE_SECONDS * 16000 or samples.sample_rate != 16000:
                raise Rejected("media_duration")
            if not samples.data.isfinite().all().item(): raise Rejected("invalid_media")
            retained = min(shape[1], options["audio"]["maxLength"]) if options["audio"]["truncation"] else shape[1]
            coverage = plan.coverage if plan.kind == "audio" else plan.coverage["audio"]
            coverage.update({"decodedSamples": int(shape[1]), "retainedSamples": int(retained),
                             "truncated": retained < shape[1]})
        # Decoder objects may retain buffers. The model gets verified paths only.
        plan.decoder = plan.audio_decoder = plan.indices = None


def resolve(batch, directory, origin, native):
    """Authorize already happened at MCP. Check exact sizes, admit all headers, then decode."""
    import time
    deadline = time.monotonic() + REQUEST_SECONDS
    conversations, media, plans = [], [], []
    for item in batch.items:
        content, item_media = [], []
        for index, part in enumerate(item.content):
            if part.type == "text":
                content.append({"type": "text", "text": part.text})
            else:
                path = fetch(part.source, directory, len(plans), origin, deadline)
                plan = native.inspect(part, path, batch.preprocessing)
                plans.append(plan)
                content.append({"type": part.type, part.type: str(path)})
                item_media.append({"partIndex": index, "type": part.type, "source": part.source.identity(), "coverage": plan.coverage})
        # The explicit outer list is mandatory even for one item.
        conversations.append([{"role": "user", "content": content}])
        media.append(item_media)
    if sum(plan.pixels for plan in plans) > MAX_TOTAL_PIXELS: raise Rejected("media_pixels")
    if sum(plan.audio_seconds for plan in plans) > MAX_AUDIO_SECONDS: raise Rejected("media_duration")
    decoded_samples = 0
    for plan in plans:
        native.validate_decoded(plan, batch.preprocessing)
        # Source metadata admits decoding early; actual decoded samples must also
        # obey the aggregate bound when a container underreports its duration.
        audio = plan.coverage if plan.kind == "audio" else plan.coverage.get("audio")
        if audio is not None:
            decoded_samples += audio["decodedSamples"]
            if decoded_samples > MAX_AUDIO_SECONDS * 16000:
                raise Rejected("media_duration")
    return conversations, media
