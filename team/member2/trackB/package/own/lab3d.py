import re

FW = 0.596


def norm(s):
    s = s.lower().replace("_", " ")
    for a, b in (("soccor", "soccer"), ("baskteball", "basketball"), ("pingpong", "ping pong"), ("note book", "notebook")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def parse_depth(info):
    out = []
    for line in str(info).split("\n"):
        m = re.search(r"(?:t\s*=\s*([\d.]+)\s*s?[^,]*,\s*)?distance_(.+?)_camera\s*=\s*([\d.]+)\s*m", line)
        if m:
            out.append((float(m.group(1)) if m.group(1) else None, norm(m.group(2)), float(m.group(3))))
    return out
