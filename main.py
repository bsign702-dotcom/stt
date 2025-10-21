# app.py
import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

# -------- Config --------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
BUCKET_DEFAULT = os.getenv("AUDIO_BUCKET", "audio-files")
CHUNK_SECONDS_DEFAULT = int(os.getenv("CHUNK_SECONDS", "600"))  # 10 minutes
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "64k")  # audio bitrate
AUDIO_CODEC = os.getenv("AUDIO_CODEC", "aac")      # codec for m4a container
AUDIO_RATE = os.getenv("AUDIO_RATE", "16000")      # 16kHz
AUDIO_CHANNELS = os.getenv("AUDIO_CHANNELS", "1")  # mono

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
app = FastAPI(title="Audio Splitter Service")

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


def run_ffmpeg_split(in_file: Path, out_dir: Path, chunk_seconds: int) -> List[Path]:
    """
    Transcode to consistent CBR and split by duration into valid .m4a chunks.
    """
    out_pattern = str(out_dir / "part_%03d.m4a")
    cmd = [
        "ffmpeg",
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
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # attach a short stderr
        tail = e.stderr.decode(errors="ignore")[-2000:]
        raise RuntimeError(f"ffmpeg failed: {tail}")

    parts = sorted(out_dir.glob("part_*.m4a"))
    if not parts:
        raise RuntimeError("No chunks produced; check input format or ffmpeg install.")
    return parts


@app.post("/split", response_model=SplitResponse)
def split_audio(req: SplitRequest):
    bucket = req.bucket or BUCKET_DEFAULT
    chunk_seconds = req.chunkSeconds or CHUNK_SECONDS_DEFAULT
    output_prefix = req.outputPrefix or "audio-chunks"

    # 1) Download from Supabase
    dl = sb.storage.from_(bucket).download(req.storagePath)
    if dl is None:
        raise HTTPException(404, "File not found or access denied in Supabase.")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_file = td_path / "input"
        in_file.write_bytes(dl)

        # 2) Split with ffmpeg
        out_dir = td_path / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)
        parts = run_ffmpeg_split(in_file, out_dir, chunk_seconds)

        # 3) Upload chunks
        storage_paths: List[str] = []
        # target dir in bucket: audio-chunks/{transcriptionId}/part_XXX.m4a
        base_dir = f"{output_prefix}/{req.transcriptionId}"
        for p in parts:
            rel_name = p.name  # part_000.m4a
            remote_path = f"{base_dir}/{rel_name}"
            with p.open("rb") as fh:
                res = sb.storage.from_(bucket).upload(remote_path, fh, {"content-type": "audio/mp4"})
            if res is None:
                raise HTTPException(500, f"Failed to upload chunk {rel_name} to {bucket}/{remote_path}")
            storage_paths.append(remote_path)

    return SplitResponse(parts=storage_paths, bucket=bucket, chunkSeconds=chunk_seconds)
