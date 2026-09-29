#!/usr/bin/env python3
"""A tool call with an array parameter must come back with that argument as a JSON array (patch 0001).

Usage: tools/toolcheck.py      (PORT env var, default 8888). Exit code 1 on failure.
"""
import json
import os
import sys
import urllib.request

URL = f"http://127.0.0.1:{os.environ.get('PORT', '8888')}/v1/chat/completions"
TOOLS = [{"type": "function", "function": {
    "name": "add_tags", "description": "Attach tags to a document.",
    "parameters": {"type": "object", "required": ["doc_id", "tags"], "properties": {
        "doc_id": {"type": "integer", "description": "The document id."},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "The tags to attach."}}}}}]


def main() -> None:
    body = {"model": "Qwen3.8-Flash-Next", "max_tokens": 1024, "temperature": 0, "tools": TOOLS,
            "messages": [{"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool."}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    msg = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    ok = False
    for c in calls:
        args = json.loads(c["function"]["arguments"])
        print(f"tool call {c['function']['name']}({json.dumps(args)})")
        ok |= c["function"]["name"] == "add_tags" and isinstance(args.get("tags"), list) and len(args["tags"]) >= 3 \
            and isinstance(args.get("doc_id"), int)
    print("toolcheck:", "OK: tags is a JSON array" if ok else f"FAILED: {msg}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
