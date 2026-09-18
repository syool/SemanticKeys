import json, urllib.request

BASE = "http://localhost:8000/v1"
model = json.load(urllib.request.urlopen(f"{BASE}/models"))["data"][0]["id"]
print(f"model: {model}\n")

def call(payload):
    payload["model"] = model
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=90))

# 1. Thinking suppression
r = call({
    "max_tokens": 200, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [
        {"role": "system", "content": "Answer with the number only."},
        {"role": "user", "content": "What is 17 times 23?"},
    ],
})
msg = r["choices"][0]["message"]
n = r["usage"]["completion_tokens"]
fin = r["choices"][0]["finish_reason"]
print(f"[thinking] tokens={n} finish={fin} reasoning={msg.get('reasoning')!r}")
print(f"           content={msg['content'][:80]!r}")
print("  OK" if n < 40 and fin == "stop" else "  PROBLEM: verbose or truncated")

# 2. Structured output
r = call({
    "max_tokens": 128, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [
        {"role": "system", "content": "Emit a verdict."},
        {"role": "user", "content": "Is 2+2=5?"},
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "v", "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verified": {"type": "boolean"},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["verified", "reasons"],
                "additionalProperties": False,
            },
        },
    },
})
c = r["choices"][0]["message"]["content"]
try:
    obj = json.loads(c)
    ok = isinstance(obj.get("verified"), bool)
except Exception:
    ok = False
print(f"\n[schema] {c[:100]!r}")
print("  OK" if ok else "  PROBLEM: schema not honoured")

# 3. Multi-field JSON, the SK-6 case that failed 51% of turns on Qwen
r = call({
    "max_tokens": 400, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [
        {"role": "system", "content":
         "Respond with a JSON object with fields: topic, who, when, where, "
         "what, how, why, evidence, explanation. No markdown fences."},
        {"role": "user", "content": "Marie Curie was born in Warsaw in 1867. Where was she born?"},
    ],
})
c = r["choices"][0]["message"]["content"]
try:
    json.loads(c.strip()); valid = True
except Exception:
    valid = False
print(f"\n[6-field JSON] valid={valid}  tokens={r['usage']['completion_tokens']}")
print(f"  {c[:150]!r}")
