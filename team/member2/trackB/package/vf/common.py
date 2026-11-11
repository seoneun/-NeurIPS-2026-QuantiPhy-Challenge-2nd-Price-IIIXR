from pathlib import Path

ROOT = Path(".")


def video_file(video_id, split="val"):
    d = ROOT / ("data/val/validation_videos" if split == "val" else "data/test")
    p = d / f"{video_id}.mp4"
    if p.exists():
        return p
    hits = [q for q in d.glob("*.mp4") if q.stem.strip() == video_id]
    return hits[0] if hits else p
