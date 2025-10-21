# app.py
import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from urllib.request import urlopen, Request

# -------- Config --------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
BUCKET_DEFAULT = os.getenv("AUDIO_BUCKET", "audio-files")
CHUNK_SECONDS_DEFAULT = int(os.getenv("CHUNK_SECONDS", "600"))  # 10 minutes
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "64k")               # audio bitrate
AUDIO_CODEC = os.getenv("AUDIO_CODEC", "aac")                   # m4a = audio/mp4
AUDIO_RATE = os.getenv("AUDIO_RATE", "16000")                   # 16kHz
AUDIO_CHANNELS = os.getenv("AUDIO_CHANNELS", "1")               # mono

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
app = FastAPI(title="Audio Splitter Service")

# CORS (اختياري - يسمح بالوصول من أي مصدر)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class SplitRequest(BaseModel):
    transcriptionId: str
    storagePath: str                  # "uploads/meeting.m4a" أو URL موقّع/عام
    bucket: Optional[str] = None      # default audio-files
    chunkSeconds: Optional[int] = None
    outputPrefix: Optional[str] = None  # e.g., "audio-chunks"

class SplitResponse(BaseModel):
    parts: List[str]
    bucket: str
    chunkSeconds: int

def run_ffmpeg_split(in_file: Path, out_dir: Path, chunk_seconds: int) -> List[Path]:
    """
    Transcode to consistent CBR AAC/m4a and split by duration into valid .m4a chunks.
    """
    out_pattern = str(out_dir / "part_%03d.m4a")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_file),
        "-ac", AUDIO_CHANNELS,
        "-ar", AUDIO_RATE,
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1",
        out_pattern,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        tail = e.stderr.decode(errors="ignore")[-2000:]
        raise RuntimeError(f"ffmpeg failed: {tail}")

    parts = sorted(out_dir.glob("part_*.m4a"))
    if not parts:
        raise RuntimeError("No chunks produced; check input format or ffmpeg install.")
    return parts

def _normalize_path(p: str) -> str:
    return p.lstrip("/")

def _download_to_bytes(bucket: str, storage_path: str) -> bytes:
    """
    - يدعم URL (عام/موقّع) عبر urllib
    - أو مسار داخلي في Supabase Storage عبر supabase-py
    """
    storage_path = storage_path.strip()

    # تنزيل عبر URL مباشر
    if storage_path.startswith("http://") or storage_path.startswith("https://"):
        try:
            req = Request(storage_path, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=120) as resp:
                if getattr(resp, "status", 200) != 200:
                    raise HTTPException(404, f"HTTP download failed: {getattr(resp, 'status', 'unknown')}")
                return resp.read()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(404, f"HTTP download error: {str(e)}")

    # تنزيل من Supabase Storage
    storage_path = _normalize_path(storage_path)
    try:
        data = sb.storage.from_(bucket).download(storage_path)
    except Exception as e:
        raise HTTPException(404, f"Supabase download threw exception: {getattr(e, 'message', str(e))}")
    if data is None:
        raise HTTPException(404, "Supabase download returned None (object not found?)")
    return data

@app.post("/split", response_model=SplitResponse)
def split_audio(req: SplitRequest):
    bucket = req.bucket or BUCKET_DEFAULT
    chunk_seconds = req.chunkSeconds or CHUNK_SECONDS_DEFAULT
    output_prefix = (req.outputPrefix or "audio-chunks").strip().strip("/")

    # 1) تنزيل الملف
    try:
        blob = _download_to_bytes(bucket, req.storagePath)
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        raise HTTPException(500, f"Unexpected error during download: {str(e)}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_file = td_path / "input_any"
        in_file.write_bytes(blob)

        # 2) تقسيم
        out_dir = td_path / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)
        parts = run_ffmpeg_split(in_file, out_dir, chunk_seconds)

        # 3) رفع الناتج
        storage_paths: List[str] = []
        base_dir = f"{output_prefix}/{req.transcriptionId}"
        for p in parts:
            rel_name = p.name
            remote_path = f"{base_dir}/{rel_name}"
            with p.open("rb")as fh:
                res = sb.storage.from_(bucket).upload(remote_path, fh, {"content-type": "audio/mp4"})
            if res is None:
                raise HTTPException(500, f"Failed to upload chunk {rel_name} to {bucket}/{remote_path}")
            storage_paths.append(remote_path)

    return SplitResponse(parts=storage_paths, bucket=bucket, chunkSeconds=chunk_seconds)
