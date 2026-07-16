import json
from dataclasses import dataclass
from pathlib import Path

import torchvision

from datamodule.transforms import VideoTransform


@dataclass(frozen=True)
class StreamItem:
    uid: str
    video_path: Path
    target_tokens: tuple[int, ...]
    target_text: str | None
    domain: str
    feedback: bool


def iter_stream_manifest(manifest_path, data_root, text_transform):
    manifest_path = Path(manifest_path)
    data_root = Path(data_root)
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            video_path = Path(row["video"])
            if not video_path.is_absolute():
                video_path = data_root / video_path
            tokens = tuple(int(token) for token in row.get("target_tokens", []))
            target_text = row.get("target_text")
            if target_text is None and tokens:
                target_text = text_transform.post_process(tokens)
            yield StreamItem(
                uid=str(row.get("uid", f"sample-{line_number:08d}")),
                video_path=video_path,
                target_tokens=tokens,
                target_text=target_text,
                domain=str(row.get("domain", "unknown")),
                feedback=bool(row.get("feedback", False)),
            )


def load_stream_video(item):
    if not item.video_path.is_file():
        raise FileNotFoundError(f"找不到流式样本：{item.video_path}")
    video = torchvision.io.read_video(
        str(item.video_path), pts_unit="sec", output_format="THWC"
    )[0]
    video = video.permute(0, 3, 1, 2).contiguous()
    return VideoTransform("test")(video)
