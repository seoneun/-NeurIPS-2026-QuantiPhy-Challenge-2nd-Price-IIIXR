import os, re
import numpy as np
import pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
VAL_CSV = f"{R}/data/val/validation_dataset.csv"
VAL_VID = f"{R}/data/val/validation_videos"
TEST_PQ = f"{R}/data/test/test_dataset.parquet"
TEST_VID = os.environ.get("QP_TEST_VID", f"{R}/data/test_480p")

SYSTEM_PROMPT = (
    "You are an expert video analyst specializing in physics measurements.\n"
    "Analyze the video frames carefully and provide ONLY the numerical answer with units. No explanation or reasoning needed.\n"
    "Format your response as: [value] [unit]\n"
    "Example: 2.5 cm\n"
    "Be as accurate as possible with measurements and calculations. Please give me an estimated answer even if you are not sure."
)

OUT_COLS = ["Unnamed: 0", "video_id", "video_source", "video_type", "fps", "inference_type", "question",
            "ground_truth_prior", "depth_info", "ground_truth_posterior", "parsed_value"]


def load_split(split):
    if split == "val":
        d = pd.read_csv(VAL_CSV)
        d = d[[c for c in d.columns if not c.startswith("Unnamed") or c == "Unnamed: 0"]].copy()
        files = {f.strip()[:-4]: os.path.join(VAL_VID, f) for f in os.listdir(VAL_VID) if f.endswith(".mp4")}
        d["video_path"] = d.video_id.map(lambda v: files.get(str(v).strip()))
    elif split == "test":
        d = pd.read_parquet(TEST_PQ).reset_index(drop=True)
        d.insert(0, "Unnamed: 0", np.arange(len(d)))
        d = d.rename(columns={"prior": "ground_truth_prior"})
        d["ground_truth_posterior"] = np.nan
        d["video_path"] = d.video_id.map(lambda v: os.path.join(TEST_VID, f"{v}.mp4"))
    else:
        raise ValueError(split)
    assert d.video_path.notna().all()
    return d


def build_context_prefix(prior=None, depth_info=None):
    parts = []
    if prior is not None and not pd.isna(prior):
        prior_text = str(prior).strip()
        if prior_text:
            parts.append(f"Given that {prior_text}.")
    if depth_info is not None and not pd.isna(depth_info):
        depth_text = str(depth_info).strip()
        if depth_text:
            parts.append(
                "Additionally, you have the following information about the distance between the objects in the video and the shooting camera: "
                f"{depth_text}"
            )
    if not parts:
        return ""
    context = " ".join(parts).strip()
    if context[-1] not in ".!?":
        context += "."
    return context + " "


def depth_for(row):
    vt = str(row.get("video_type", ""))
    return row.get("depth_info") if (len(vt) >= 2 and vt[1] == "3") else None


num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
EXACT_NUMBER_WITH_UNIT_RE = re.compile(
    r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*(?:"
    r"meters?|meter|m|kilometers?|kilometer|km|centimeters?|centimeter|cm|millimeters?|millimeter|mm|"
    r"inches?|inch|in|feet|foot|ft|yards?|yard|yd|"
    r"seconds?|second|s|minutes?|minute|min|hours?|hour|h|"
    r"m/s|m/s²|m/s^2|m/s2|km/h|kph|mph|ft/s|cm/s|mm/s|"
    r"m/s/s|m/s²|m/s^2|m/s2|ft/s²|ft/s^2|ft/s2|cm/s²|cm/s^2|cm/s2|mm/s²|mm/s^2|mm/s2|g"
    r")?\s*$",
    re.IGNORECASE,
)


def parse_number(text):
    if not text or pd.isna(text):
        return np.nan
    original_text = str(text).strip()
    if EXACT_NUMBER_WITH_UNIT_RE.match(original_text):
        num_match = num_re.search(original_text)
        if num_match:
            try:
                return abs(float(num_match.group(0)))
            except ValueError:
                pass
    text_to_parse = original_text
    delimiters = ["Final Answer:", "Answer:", "=>", "=", ":", "is:", "The answer is:", "Example:"]
    last_delimiter_pos, last_delimiter = -1, None
    for delimiter in delimiters:
        positions = [m.start() for m in re.finditer(re.escape(delimiter), text_to_parse, re.IGNORECASE)]
        if positions:
            last_pos = max(positions)
            if last_pos > last_delimiter_pos:
                last_delimiter_pos, last_delimiter = last_pos, delimiter
    if last_delimiter_pos >= 0 and last_delimiter:
        parts = text_to_parse.rsplit(last_delimiter, 1)
        if len(parts) > 1:
            text_to_parse = parts[-1].strip()
    text_cleaned = text_to_parse.lower()
    units_to_remove = ['kilometers', 'kilometer', 'km', 'meters', 'meter', 'm', 'centimeters', 'centimeter', 'cm',
                       'millimeters', 'millimeter', 'mm', 'inches', 'inch', 'in', 'feet', 'foot', 'ft', 'yards', 'yard', 'yd',
                       'hours', 'hour', 'h', 'minutes', 'minute', 'min', 'seconds', 'second', 's',
                       'km/h', 'kph', 'mph', 'ft/s', 'm/s', 'cm/s', 'mm/s',
                       'm/s/s', 'm/s²', 'm/s^2', 'm/s2', 'ft/s²', 'ft/s^2', 'ft/s2', 'cm/s²', 'cm/s^2', 'cm/s2',
                       'mm/s²', 'mm/s^2', 'mm/s2', 'g', 'm^2', 'm²']
    for unit in units_to_remove:
        text_cleaned = text_cleaned.replace(unit, '')
    all_numbers = num_re.findall(text_cleaned)
    if all_numbers:
        try:
            return abs(float(all_numbers[-1]))
        except (ValueError, IndexError):
            pass
    all_numbers_original = num_re.findall(original_text)
    if all_numbers_original:
        try:
            return abs(float(all_numbers_original[-1]))
        except (ValueError, IndexError):
            pass
    return np.nan


UNIT_TABLE = {
    "mm": (1e-3, 0), "millimeters": (1e-3, 0), "millimeter": (1e-3, 0),
    "cm": (1e-2, 0), "centimeters": (1e-2, 0), "centimeter": (1e-2, 0),
    "m": (1.0, 0), "meters": (1.0, 0), "meter": (1.0, 0), "metres": (1.0, 0), "km": (1e3, 0),
    "mm/s": (1e-3, 1), "cm/s": (1e-2, 1), "m/s": (1.0, 1), "km/h": (1 / 3.6, 1), "kph": (1 / 3.6, 1),
    "mm/s^2": (1e-3, 2), "cm/s^2": (1e-2, 2), "m/s^2": (1.0, 2),
}


def norm_unit(u):
    if u is None:
        return None
    u = u.strip().lower().replace("²", "^2").replace("s2", "s^2").rstrip(".?)")
    u = u.replace("m/s/s", "m/s^2")
    return u if u in UNIT_TABLE else None


def target_unit(question, inference_type):
    q = question.replace("？", "?").strip()
    m = re.findall(r"\bin\s*\(?\s*((?:mm|cm|m|km)(?:/s(?:\^2|²|2)?)?|meters?|metres?|centimeters?|millimeters?|km/h)\b(?:\^2|²)?", q, re.I)
    if m:
        u = m[-1]
        idx = q.lower().rfind(u.lower())
        tail = q[idx + len(u): idx + len(u) + 2]
        if "/s" in u.lower() and ("^2" in tail or "²" in tail) and "^2" not in u:
            u = u + "^2"
        nu = norm_unit(u)
        if nu:
            return nu
    return "m" if inference_type[1] == "S" else "m/s"


def prior_value(prior):
    s = str(prior).split("\n")[0]
    for m in re.finditer(r"([-+]?\d*\.?\d+)\s*(meters?|metres?|centimeters?|millimeters?|km/h|(?:mm|cm|m|km)(?:/s(?:\^2|²|2)?)?)(?![a-z])", s, re.I):
        nu = norm_unit(m.group(2))
        if nu:
            f, k = UNIT_TABLE[nu]
            return float(m.group(1)) * f, k
    return None, None
