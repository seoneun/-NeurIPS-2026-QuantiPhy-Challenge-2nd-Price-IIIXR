import re

UNIT_SI = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "km": 1e3,
           "mm/s": 1e-3, "cm/s": 1e-2, "m/s": 1.0, "km/h": 1 / 3.6,
           "mm/s^2": 1e-3, "cm/s^2": 1e-2, "m/s^2": 1.0}
UNIT_K = {"mm": 0, "cm": 0, "m": 0, "km": 0, "mm/s": 1, "cm/s": 1, "m/s": 1, "km/h": 1,
          "mm/s^2": 2, "cm/s^2": 2, "m/s^2": 2}

TYPOS = {
    "bycicle": "bicycle", "conrner": "corner", "balck": "black", "ballon the": "ball on the",
    "velolicty": "velocity", "velocoty": "velocity", "diatance": "distance", "soccor": "soccer",
    "diasplacement": "displacement", "centeral": "central", "baskteball": "basketball",
    "lenth": "length", "rebounce": "rebound", "soccerball": "soccer ball", "bill board": "billboard",
    "the the ": "the ", "shoppingbag": "shopping bag", "conveyer": "conveyor", "trashbin": "trash bin",
    "pingpong": "ping pong",
}
NUM = r"(\d*\.?\d+)"




UNIT_RE = r"(m/s\^2|m/s2|cm/s\^2|cm/s2|mm/s\^2|m/s|cm/s|mm/s|km/h|meters?|metres?|centimeters?|millimeters?|cm|mm|km|m)"


def norm_unit(u):
    u = u.lower().strip()
    u = {"meter": "m", "meters": "m", "metre": "m", "metres": "m", "centimeter": "cm", "centimeters": "cm",
         "millimeter": "mm", "millimeters": "mm", "m/s2": "m/s^2", "cm/s2": "cm/s^2"}.get(u, u)
    return u if u in UNIT_SI else None






SIZE_DIMS = [("wingspan", "wingspan"), ("wing span", "wingspan"), ("diameter", "diameter"), ("radius", "radius"),
             ("thickness", "thickness"), ("height", "height"), ("tall", "height"), ("length", "length"),
             ("long", "length"), ("width", "width"), ("wide", "width"), ("breadth", "width"), ("size", "size"),
             ("depth", "thickness")]










def parse_depth(depth_info):
    out = []
    if depth_info is None or str(depth_info).strip() in ("", "nan", "None"):
        return out
    for line in str(depth_info).split("\n"):
        l = line.strip()
        if not l:
            continue
        t = None
        mt = re.match(r"^\s*t\s*=\s*" + NUM + r"\s*s?\s*,\s*", l)
        if mt:
            t = float(mt.group(1)); l = l[mt.end():]
        m = re.search(r"distance_?(.+?)\s*=\s*" + NUM, l)
        if not m:
            continue
        key = m.group(1).strip()
        key = re.sub(r"(?:_and)?_camera\s*$", "", key)
        out.append(dict(key=key, t=t, d=float(m.group(2))))
    return out


SYN = {"human": "person", "people": "person", "pedestrian": "person", "man": "person", "woman": "person",
       "boy": "person", "girl": "person", "player": "person", "walker": "person", "lady": "person",
       "bike": "bicycle", "cyclist": "bicycle", "pingpong": "ping", "ping-pong": "ping", "pong": "ping",
       "notebook": "note", "trashbin": "trash", "bin": "trash", "automobile": "car", "vehicle": "car",
       "aeroplane": "plane", "airplane": "plane", "jet": "plane", "kid": "person", "child": "person",
       "astronaut": "person", "swimmer": "person", "boatman": "person", "runner": "person",
       "centeral": "central", "seagull": "bird", "eagle": "eagle", "ship": "boat", "sailboat": "boat",
       "basketball": "basketball", "football": "soccer", "mug": "cup", "glass": "cup"}
COLORS = {"white", "black", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "gray", "silver", "golden", "gold"}
STOP = {"the", "a", "an", "of", "and", "on", "in", "at", "to", "camera", "distance", "one", "with", "is", "that", "this"}






