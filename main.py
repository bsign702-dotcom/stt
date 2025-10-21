# app.py
import os
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# -------- Config --------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
BUCKET_DEFAULT = os.getenv("AUDIO_BUCKET", "audio-files")
CHUNK_SECONDS_DEFAULT = int(os.getenv("CHUNK_SECONDS", "600"))  # 10 minutes
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "64k")    # audio bitrate
AUDIO_CODEC = os.getenv("AUDIO_CODEC", "aac")        # codec for m4a container
AUDIO_RATE = os.getenv("AUDIO_RATE", "16000")        # 16kHz
AUDIO_CHANNELS = os.getenv("AUDIO_CHANNELS", "1")    # mono
FFMPEG_PATH_ENV = os.getenv("FFMPEG_PATH", "")       # optional custom path

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
app = FastAPI(title="Audio Splitter Service")

# ---- CORS (مهم لاستدعاء الخدمة من أي مكان) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class SplitRequest(BaseModel):
    transcriptionId: str
    storagePath: str                  # e.g., "uploads/meeting.m4a"
    bucket: Optional[str] = None      # default audio-files
    chunkSeconds: Optional[int] = None
    # Optional prefix to place the chunks
    outputPrefix: Optional[str] = None  # e.g., "audio-chunks"

class SplitResponse(BaseModel):
    parts: List[str]                  # list of storage paths for chunks (in same bucket)
    bucket: str
    chunkSeconds: int
    ffmpeg: Optional[str] = None      # info for debugging

def find_ffmpeg() -> Tuple[str, str]:
    """
    Returns (path, version_string). Raises if not found.
    """
    # Priority: env override -> which -> common paths
    candidates = []
    if FFMPEG_PATH_ENV:
        candidates.append(FFMPEG_PATH_ENV)
    which = shutil.which("ffmpeg")
    if which:
        candidates.append(which)
    candidates += ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/bin/ffmpeg"]

    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            try:
                ver = subprocess.run([c, "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True, timeout=5)
                return c, ver.stdout.strip().splitlines()[0]
            except Exception as e:
                continue
    raise RuntimeError("ffmpeg not found. Set FFMPEG_PATH or install ffmpeg on the server.")

def run_ffmpeg_split(ffmpeg_path: str, in_file: Path, out_dir: Path, chunk_seconds: int) -> List[Path]:
    """
    Transcode to consistent CBR and split by duration into valid .m4a chunks.
    """
    out_pattern = str(out_dir / "part_%03d.m4a")
    cmd = [
        ffmpeg_path,
        "-y",
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
        # اجمع stdout/stderr لتشخيص سريع
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode(errors="ignore")[-4000:]
        raise RuntimeError(f"ffmpeg failed: {tail}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg timeout (600s). Try shorter chunkSeconds or check input format.")

    parts = sorted(out_dir.glob("part_*.m4a"))
    if not parts:
        raise RuntimeError("No chunks produced; check input format or ffmpeg install.")
    return parts

@app.get("/health")
def health():
    try:
        path, ver = find_ffmpeg()
        return {"status": "ok", "ffmpeg": {"path": path, "version": ver}}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@app.get("/ffmpeg-info")
def ffmpeg_info():
    try:
        path, ver = find_ffmpeg()
        return {"path": path, "version": ver}
    except Exception as e:
        raise HTTPException(500, f"ffmpeg not available: {e}")

@app.post("/split", response_model=SplitResponse)
def split_audio(req: SplitRequest):
    bucket = req.bucket or BUCKET_DEFAULT
    chunk_seconds = req.chunkSeconds or CHUNK_SECONDS_DEFAULT
    output_prefix = req.outputPrefix or "audio-chunks"

    # 0) تأكد من ffmpeg
    ffmpeg_path, ffmpeg_ver = find_ffmpeg()

    # 1) Download from Supabase
    try:
        dl = sb.storage.from_(bucket).download(req.storagePath)
    except Exception as e:
        raise HTTPException(500, f"Supabase download threw exception: {e}")

    if dl is None:
        raise HTTPException(404, f"File not found or access denied: bucket={bucket}, path={req.storagePath}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_file = td_path / "input.any"
        try:
            in_file.write_bytes(dl)
        except Exception as e:
            raise HTTPException(500, f"Failed to write temp input: {e}")

        # 2) Split with ffmpeg
        out_dir = td_path / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            parts = run_ffmpeg_split(ffmpeg_path, in_file, out_dir, chunk_seconds)
        except Exception as e:
            raise HTTPException(500, f"{e}")

        # 3) Upload chunks
        storage_paths: List[str] = []
        base_dir = f"{output_prefix}/{req.transcriptionId}"
        for p in parts:
            rel_name = p.name  # part_000.m4a
            remote_path = f"{base_dir}/{rel_name}"
            try:
                with p.open("rb") as fh:
                    # upsert=True مهم إذا أعدت التشغيل
                    res = sb.storage.from_(bucket).upload(
                        remote_path, fh,
                        {"content-type": "audio/mp4", "upsert": "true"}
                    )
            except Exception as e:
                raise HTTPException(500, f"Supabase upload failed for {remote_path}: {e}")

            if res is None:
                raise HTTPException(500, f"Upload returned None for {remote_path}")

            storage_paths.append(remote_path)

    return SplitResponse(
        parts=storage_paths,
        bucket=bucket,
        chunkSeconds=chunk_seconds,
        ffmpeg=ffmpeg_ver
    )
