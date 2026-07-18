import sys
import tempfile
from pathlib import Path

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preparation.video_io import write_video


def main():
    torch.manual_seed(7)
    frames = torch.randint(0, 256, (8, 96, 96, 3), dtype=torch.uint8)
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "roundtrip.mp4"
        write_video(output_path, frames, fps=25)
        decoded, _, metadata = torchvision.io.read_video(
            str(output_path), pts_unit="sec", output_format="THWC"
        )
        assert decoded.shape == frames.shape
        assert metadata["video_fps"] == 25.0
    print("视频写入 smoke 通过")


if __name__ == "__main__":
    main()
