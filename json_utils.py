import json, re, ast
from json import JSONDecodeError

def safe_parse_json(maybe_json: str):
    s = maybe_json.strip()

    # 1) Strip code fences if present
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)

    # 2) Try direct JSON
    try:
        return json.loads(s)
    except JSONDecodeError:
        pass

    # 3) Try to extract the first JSON array/object substring
    match = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            return json.loads(candidate)
        except JSONDecodeError:
            # 4) Last resort: tolerate single quotes via ast.literal_eval (Python-ish)
            try:
                return ast.literal_eval(candidate)
            except Exception:
                pass

    # 5) Another last resort: ast on full string
    try:
        return ast.literal_eval(s)
    except Exception:
        raise ValueError("Could not parse LLM output as JSON")
