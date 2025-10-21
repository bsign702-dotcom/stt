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
AUDIO_CODEC = os.getenv("AUDIO_CODEC", "aac")                   # m4a container (audio/mp4)
AUDIO_RATE = os.getenv("AUDIO_RATE", "16000")                   # 16kHz
AUDIO_CHANNELS = os.getenv("AUDIO_CHANNELS", "1")               # mono

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# -------- FastAPI app --------
app = FastAPI(title="Audio Splitter Service")

# CORS (allow all)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------- Models --------
class SplitRequest(BaseModel):
    transcriptionId: str
    storagePath: str                  # "uploads/meeting.m4a" or signed/public URL
    bucket: Optional[str] = None      # default audio-files
    chunkSeconds: Optional[int] = None
    outputPrefix: Optional[str] = None  # e.g., "audio-chunks"

class SplitResponse(BaseModel):
    parts: List[str]
    bucket: str
    chunkSeconds: int

# -------- Helpers --------
def run_ffmpeg_split(in_file: Path, out_dir: Path, chunk_seconds: int) -> List[Path]:
    """Transcode to consistent CBR AAC/m4a and split by duration."""
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
    """Supports Supabase path or full public/signed URL."""
    storage_path = storage_path.strip()

    # 1) Direct URL
    if storage_path.startswith("http://") or storage_path.startswith("https://"):
        try:
            req = Request(storage_path, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=120) as resp:
                if getattr(resp, "status", 200) != 200:
                    raise HTTPException(404, f"HTTP download failed: {resp.status}")
                return resp.read()
        except Exception as e:
            raise HTTPException(404, f"Download error: {str(e)}")

    # 2) Supabase Storage
    storage_path = _normalize_path(storage_path)
    try:
        data = sb.storage.from_(bucket).download(storage_path)
    except Exception as e:
        raise HTTPException(404, f"Supabase download exception: {getattr(e, 'message', str(e))}")
    if data is None:
        raise HTTPException(404, "File not found in Supabase storage.")
    return data

# -------- Routes --------
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/routes")
def list_routes():
    return {
        "routes": sorted(
            [
                {"path": r.path, "methods": sorted(list(getattr(r, "methods", [])))}
                for r in app.routes
            ],
            key=lambda x: x["path"],
        ),
    }

@app.post("/split", response_model=SplitResponse)
def split_audio(req: SplitRequest):
    bucket = req.bucket or BUCKET_DEFAULT
    chunk_seconds = req.chunkSeconds or CHUNK_SECONDS_DEFAULT
    output_prefix = (req.outputPrefix or "audio-chunks").strip().strip("/")

    # 1) Download source file
    try:
        blob = _download_to_bytes(bucket, req.storagePath)
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        raise HTTPException(500, f"Unexpected download error: {str(e)}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_file = td_path / "input.m4a"
        in_file.write_bytes(blob)

        # 2) Split file
        out_dir = td_path / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)
        parts = run_ffmpeg_split(in_file, out_dir, chunk_seconds)

        # 3) Upload chunks to Supabase
        storage_paths: List[str] = []
        base_dir = f"{output_prefix}/{req.transcriptionId}"
        for p in parts:
            rel_name = p.name
            remote_path = f"{base_dir}/{rel_name}"
            with p.open("rb") as fh:
                res = sb.storage.from_(bucket).upload(remote_path, fh, {"content-type": "audio/mp4"})
            if res is None:
                raise HTTPException(500, f"Failed to upload chunk {rel_name}")
            storage_paths.append(remote_path)

    return SplitResponse(parts=storage_paths, bucket=bucket, chunkSeconds=chunk_seconds)
