"""The reporting half of scripts/demo.sh.

Reads the page the service produced, corrects it the way a person would,
uploads the identical page again, and prints what changed. The point is the
second pass: the same image, the same recogniser, the same mistakes coming out
of the API -- and a system that now knows how this handwriting fails.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8810")

# What the mock provider is reading from. In real use this is whatever you
# wrote, and the corrections come from you fixing the text in the editor.
TRUTH = [
    "the noisy channel model treats recognition output as an observation",
    "every correction gives an aligned pair of predicted and corrected text",
    "accumulating those alignments builds a personal confusion matrix",
    "the matrix shows my own failure modes rather than an average writer's",
    "low confidence tokens get candidates from inverse confusions",
    "the decoder substitutes only when the margin is decisive",
]

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=30) as response:
        return json.load(response)


def post(path: str, payload: dict, base: str = BASE):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def wait_for_document(document_id: int, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = get(f"/api/documents/{document_id}")
        if payload["document"]["status"] in ("done", "failed"):
            return payload
        time.sleep(0.25)
    raise SystemExit(f"document {document_id} did not finish")


def diff_words(observed: str) -> int:
    wrong = 0
    for line, truth in zip(observed.split("\n"), TRUTH):
        for got, want in zip(line.split(), truth.split()):
            if got != want:
                wrong += 1
    return wrong


def show(text: str) -> None:
    for line, truth in zip(text.split("\n"), TRUTH):
        rendered = []
        for got, want in zip(line.split(), truth.split()):
            rendered.append(f"{RED}{got}{RESET}" if got != want else got)
        print("    " + " ".join(rendered))


def main() -> int:
    document = wait_for_document(1)
    page_id = document["pages"][0]["id"]
    page = get(f"/api/pages/{page_id}")
    first = page["page"]["text"]

    print(f"\n{BOLD}6. What came back{RESET}")
    show(first)

    tiers = {"red": 0, "amber": 0, "none": 0}
    for token in page["tokens"]:
        tiers[token["tier"]] = tiers.get(token["tier"], 0) + 1
    before = diff_words(first)
    print(f"\n    {before} wrong words. Underlined: "
          f"{tiers['red']} red, {tiers['amber']} amber, {tiers['none']} left clean.")

    flagged = [t for t in page["tokens"] if t["tier"] != "none"]
    for token in flagged[:5]:
        suggestion = f"  suggests {token['suggestion']!r}" if token.get("suggestion") else ""
        print(f"      {DIM}{token['tier']:5s}{RESET} {token['original']!r:14s} "
              f"{token['confidence']:.0%}  {token.get('reason','')}{suggestion}")

    print(f"\n{BOLD}7. Correcting it, the way you would in the editor{RESET}")
    result = post(f"/api/pages/{page_id}/corrections", {"text": "\n".join(TRUTH)})
    print(f"    {result['pairs']} corrected line(s) sent, {result['learned']} learned")

    rescorer = os.environ.get("RESCORER", "http://127.0.0.1:8811")
    with urllib.request.urlopen(rescorer + "/stats", timeout=10) as response:
        stats = json.load(response)
    print(f"    the confusion matrix now holds {stats['confusion_pairs']} character pairs "
          f"from {stats['corrections']} correction(s):")
    for row in stats["top_confusions"][:5]:
        print(f"      {row['written']!r} read as {row['read_as']!r}: "
              f"{row['count']} times ({row['rate']:.0%} of the time)")

    print(f"\n{BOLD}8. The same page again{RESET}")
    print(f"    {DIM}identical image, identical recogniser output -- only the "
          f"post-processor has changed{RESET}")

    import subprocess

    work = os.path.dirname(os.environ.get("DB", "/tmp/rosetta-demo/rosetta.db"))
    subprocess.run(
        ["curl", "-sf", "-X", "POST", "-F", f"file=@{work}/page.png",
         f"{BASE}/api/documents"],
        check=True, stdout=subprocess.DEVNULL,
    )

    second_document = wait_for_document(2)
    second_page = get(f"/api/pages/{second_document['pages'][0]['id']}")
    second = second_page["page"]["text"]
    show(second)

    after = diff_words(second)
    rewritten = [t for t in second_page["tokens"] if t["text"].lower() != t["original"].lower()]

    print(f"\n    {GREEN}{before} wrong words became {after}{RESET}, "
          f"with {len(rewritten)} rewritten by the rescorer:")
    for token in rewritten[:8]:
        print(f"      {token['original']!r} -> {token['text']!r}")

    print(f"\n{BOLD}That is the loop.{RESET} Every correction teaches the error model, "
          f"the lexicon and\nthe language prior at once, and the next page is read better "
          f"for it.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
