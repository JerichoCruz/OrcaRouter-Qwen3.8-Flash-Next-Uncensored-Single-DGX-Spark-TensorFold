#!/usr/bin/env python3
"""Needle in a haystack at ~195k tokens: a passphrase hidden at ~60% depth of random prose, asked for greedily.

Usage: tools/needle.py [label] [approx words param]     (PORT env var, default 8888). Exit code 1 if the answer is wrong.
"""
import json
import os
import sys
import time
import urllib.request

from bench import prose

URL = f"http://127.0.0.1:{os.environ.get('PORT', '8888')}/v1/chat/completions"
SECRET = "violet-harbor-7291"


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "needle"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 248_000
    hay = prose(size, 777).split(". ")
    at = int(len(hay) * 0.6)
    hay.insert(at, f"Remember this: the secret passphrase is {SECRET}")
    prompt = ". ".join(hay) + "\n\nWhat is the secret passphrase mentioned in the text above? Reply with the passphrase only."
    body = {"model": "Qwen3.8-Flash-Next", "max_tokens": 512, "temperature": 0, "seed": 1234,
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time()
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(req, timeout=3600))
    content = reply["choices"][0]["message"].get("content") or ""
    ok = SECRET in content
    print(f"{label}: needle {reply['usage']['prompt_tokens']} tok, prefill {reply['tensorfold'].get('prefill_s')} s, "
          f"total {time.time() - t0:.1f} s, answer {content.strip()[-80:]!r}: {'CORRECT' if ok else 'WRONG'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
