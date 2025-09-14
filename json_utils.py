import json, re, ast
from json import JSONDecodeError

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.S)

def safe_parse_json(s: str):
    if s is None:
        raise ValueError("No JSON text")
    s = s.strip()
    if not s:
        raise ValueError("Empty JSON text")

    # Strip ```json fences if present
    if s.startswith("```"):
        s = _CODE_FENCE_RE.sub("", s).strip()

    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass

    # Fallback: extract first JSON object/array from mixed text
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.S)
    if m:
        return json.loads(m.group(1))

    raise ValueError("Could not parse LLM output as JSON")