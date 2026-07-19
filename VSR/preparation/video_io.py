import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torchvision


def _find_ffmpeg():
    ffmpeg = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "未找到 ffmpeg；请安装 ffmpeg/imageio-ffmpeg "
            "或设置 FFMPEG_BINARY"
        ) from error


def write_video(filename, frames, fps=25):
    output_path = Path(filename)
    ffmpeg = _find_ffmpeg()
    with tempfile.TemporaryDirectory(dir=output_path.parent) as temporary_directory:
        encoded_path = Path(temporary_directory) / "encoded.mp4"
        remuxed_path = Path(temporary_directory) / "remuxed.mp4"
        torchvision.io.write_video(str(encoded_path), frames, fps=fps)
        # PyAV 10 records the MP4 track duration at the final PTS, hiding the last frame on reopen.
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(encoded_path),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-y",
                str(remuxed_path),
            ],
            check=True,
        )
        os.replace(remuxed_path, output_path)
