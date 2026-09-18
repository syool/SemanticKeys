"""
verify_models.py — Preflight check for HuggingFace repositories.

Run this BEFORE starting a ~100 GB download. It queries the HuggingFace API
for each configured repository and reports:

  - whether the repo exists at all (catches typos and renamed repos)
  - whether it is gated and whether your token has access (Gemma requires
    accepting a license on the model page first)
  - the actual total size of the weight files, so you can check disk space
    and catch cases where you are about to pull a much larger variant than
    you intended
  - the current revision SHA, which belongs in the paper's reproducibility
    statement

Model catalogs move. Repository IDs that were correct when this framework
was written may have been renamed, deprecated, or superseded. This script
exists so that a stale ID fails in five seconds rather than after an hour
of downloading.

Models and datasets live behind separate API paths on HuggingFace, and this
script routes each entry accordingly. The convention in models.conf is that
keys ending in `_HF_REPO` are datasets and other `_REPO` keys are models.
That matters because querying a dataset against the models endpoint returns
401, which reads like a licensing problem and points you at a model page
that was never going to exist.

Usage:
    python verify_models.py                      # check everything in models.conf
    python verify_models.py --repo Qwen/Qwen3.5-27B
    python verify_models.py --repo hotpotqa/hotpot_qa --repo-type dataset
    python verify_models.py --conf ../models.conf
    python verify_models.py --search qwen3                      # find models
    python verify_models.py --search hotpot --search-type dataset
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

HF_API = "https://huggingface.co/api"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


def _get_json(url: str, token: str | None = None, timeout: int = 20):
    """GET a JSON endpoint. Returns (status_code, parsed_or_None, error_text)."""
    headers = {"User-Agent": "sk-verify-models/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read()), ""
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, None, body[:400]
    except urllib.error.URLError as e:
        return 0, None, f"network error: {e.reason}"


def parse_conf(conf_path: Path) -> dict[str, str]:
    """
    Extract KEY="value" pairs from a shell-style conf file, ignoring comments
    and commented-out alternatives.
    """
    out: dict[str, str] = {}
    if not conf_path.exists():
        return out
    pattern = re.compile(r'^\s*([A-Z0-9_]+)\s*=\s*"([^"]*)"\s*$')
    for line in conf_path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = pattern.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def human_gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.1f} GB"


# HuggingFace serves models and datasets from separate API paths. Querying a
# dataset ID against the models path returns 401/404, which looks like a
# gating problem but is really a wrong-endpoint problem — so repo_type must
# be routed correctly rather than guessed from the error.
REPO_TYPE_PATHS = {"model": "models", "dataset": "datasets"}

# File extensions that count toward "how big is this download".
PAYLOAD_EXTS = {
    "model": (".safetensors", ".bin", ".gguf", ".pt", ".onnx"),
    "dataset": (".parquet", ".arrow", ".json", ".jsonl", ".csv", ".tsv",
                ".zip", ".gz", ".tar"),
}


def verify_repo(repo_id: str, token: str | None,
                repo_type: str = "model") -> dict:
    """
    Query the HF API for one repository.

    repo_type is "model" or "dataset"; it selects the API path and the set of
    file extensions counted toward the download size. Passing the wrong type
    produces a 404 or 401 that is easy to misread as a permissions problem,
    so callers should set it explicitly.

    Returns a dict with: repo, repo_type, ok, status, gated, sha,
    total_bytes, n_payload_files, message.
    """
    result = {
        "repo": repo_id, "repo_type": repo_type, "ok": False, "status": None,
        "gated": None, "sha": None, "total_bytes": 0, "n_payload_files": 0,
        "message": "",
    }

    api_path = REPO_TYPE_PATHS.get(repo_type)
    if api_path is None:
        result["message"] = (
            f"unknown repo_type {repo_type!r} "
            f"(expected one of {sorted(REPO_TYPE_PATHS)})"
        )
        return result

    # files_metadata=1 asks the API to include per-file sizes.
    status, data, err = _get_json(
        f"{HF_API}/{api_path}/{repo_id}?files_metadata=1", token=token
    )
    result["status"] = status

    other_type = "dataset" if repo_type == "model" else "model"

    if status == 404:
        result["message"] = (
            f"not found under /{api_path} — check for a typo or a rename. "
            f"If this is actually a {other_type}, it belongs in a "
            f"{other_type} entry rather than a {repo_type} one."
        )
        return result
    if status in (401, 403):
        result["message"] = (
            f"access denied under /{api_path}. Two common causes: the repo is "
            f"gated (accept the license at https://huggingface.co/{repo_id} "
            f"and set HF_TOKEN), or it is really a {other_type} and was "
            f"queried against the wrong endpoint."
        )
        result["gated"] = True
        return result
    if status == 0 or data is None:
        result["message"] = err or "no response"
        return result
    if status != 200:
        result["message"] = f"unexpected status {status}: {err}"
        return result

    result["sha"] = data.get("sha")
    result["gated"] = bool(data.get("gated", False))

    exts = PAYLOAD_EXTS[repo_type]
    total = 0
    count = 0
    for sib in data.get("siblings", []) or []:
        name = sib.get("rfilename", "")
        size = sib.get("size") or 0
        if name.endswith(exts):
            total += size
            count += 1
    result["total_bytes"] = total
    result["n_payload_files"] = count

    if count == 0:
        label = "weight" if repo_type == "model" else "data"
        result["message"] = (
            f"no {label} files found — the API may not have returned file "
            f"metadata, or this repo hosts only configs and loading scripts"
        )
        result["ok"] = True  # repo exists; size just unknown
        return result

    result["ok"] = True
    return result


def search_repos(query: str, token: str | None, limit: int = 15,
                 repo_type: str = "model") -> None:
    """
    List candidate repos matching a query — useful when an ID has gone stale.

    Searches models by default. Pass repo_type="dataset" to search datasets,
    which is what you want when a dataset ID stops resolving.
    """
    import urllib.parse

    api_path = REPO_TYPE_PATHS.get(repo_type, "models")
    status, data, err = _get_json(
        f"{HF_API}/{api_path}?search={urllib.parse.quote(query)}"
        f"&limit={limit}&sort=downloads&direction=-1",
        token=token,
    )
    if status != 200 or data is None:
        print(f"{RED}Search failed ({status}): {err}{RESET}")
        return
    if not data:
        print(f"\nNo {repo_type} matches for {query!r}.")
        other = "dataset" if repo_type == "model" else "model"
        print(f"If you expected a {other}, retry with --search-type {other}.\n")
        return
    print(f"\nTop {len(data)} {repo_type} matches for {query!r} (by downloads):\n")
    for m in data:
        mid = m.get("modelId") or m.get("id", "")
        dl = m.get("downloads", 0)
        gated = " [gated]" if m.get("gated") else ""
        print(f"  {mid:<55} {dl:>10,} downloads{gated}")
    print()


def check_disk_space(path: Path, needed_bytes: int) -> None:
    """Warn if the target filesystem cannot hold the planned download."""
    import shutil

    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
    except OSError as e:
        print(f"  {YELLOW}Could not check disk space: {e}{RESET}")
        return

    free = usage.free
    # Leave 10% headroom — filling a filesystem completely causes odd failures.
    needed_with_headroom = int(needed_bytes * 1.1)
    print(f"\nDisk space at {path}:")
    print(f"  Free:              {human_gb(free)}")
    print(f"  Needed (+10%):     {human_gb(needed_with_headroom)}")
    if free < needed_with_headroom:
        short = needed_with_headroom - free
        print(f"  {RED}INSUFFICIENT — short by {human_gb(short)}{RESET}")
    else:
        print(f"  {GREEN}Sufficient{RESET}")


def classify_conf_repos(conf: dict[str, str]) -> list[tuple[str, str, str]]:
    """
    Split conf entries into (key, repo_id, repo_type) triples.

    Convention in models.conf:
        *_HF_REPO  → dataset  (e.g. HOTPOTQA_HF_REPO)
        *_REPO     → model    (e.g. QWEN_REPO, EMBEDDER_REPO)

    Getting this wrong is the failure this function exists to prevent: a
    dataset ID queried against the models endpoint returns 401, which reads
    like a licensing problem and sends you to a model page that does not
    exist.
    """
    out: list[tuple[str, str, str]] = []
    for key, value in conf.items():
        if not value or not key.endswith("_REPO"):
            continue
        repo_type = "dataset" if key.endswith("_HF_REPO") else "model"
        out.append((key, value, repo_type))
    # Models first so the size-heavy entries lead the report.
    out.sort(key=lambda t: (t[2] != "model", t[0]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conf", default=None,
                   help="Path to models.conf (default: ../models.conf relative to this file)")
    p.add_argument("--repo", action="append", default=[],
                   help="Check a specific repo instead of the conf. Repeatable.")
    p.add_argument("--repo-type", choices=["model", "dataset"], default="model",
                   help="Repo type for --repo entries (default: model)")
    p.add_argument("--search", default=None,
                   help="Search HF for repos matching a query, then exit.")
    p.add_argument("--search-type", choices=["model", "dataset"], default="model",
                   help="What to search when using --search (default: model)")
    p.add_argument("--token", default=None,
                   help="HF token (default: HF_TOKEN or HUGGING_FACE_HUB_TOKEN env)")
    p.add_argument("--target-dir", default=None,
                   help="Directory the download will go to, for a disk-space check")
    args = p.parse_args()

    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    if args.search:
        search_repos(args.search, token, repo_type=args.search_type)
        return

    # --- Assemble the list of repos to check ---
    if args.repo:
        entries = [("--repo", r, args.repo_type) for r in args.repo]
        source = "command line"
    else:
        conf_path = Path(args.conf) if args.conf else \
            Path(__file__).resolve().parent.parent / "models.conf"
        conf = parse_conf(conf_path)
        if not conf:
            print(f"{RED}Could not read {conf_path}{RESET}", file=sys.stderr)
            sys.exit(2)
        entries = classify_conf_repos(conf)
        source = str(conf_path)

    n_models = sum(1 for _, _, t in entries if t == "model")
    n_datasets = len(entries) - n_models

    print("=" * 70)
    print("HuggingFace repository verification")
    print("=" * 70)
    print(f"Source:  {source}")
    print(f"Token:   {'set' if token else 'NOT SET (gated repos will fail)'}")
    print(f"Checking {n_models} model(s) and {n_datasets} dataset(s)")
    print()

    results = []
    for key, repo, repo_type in entries:
        print(f"[{repo_type:<7}] {repo} ...", end=" ", flush=True)
        r = verify_repo(repo, token, repo_type=repo_type)
        r["conf_key"] = key
        results.append(r)
        if r["ok"] and r["n_payload_files"]:
            print(f"{GREEN}OK{RESET}  "
                  f"{human_gb(r['total_bytes'])} in {r['n_payload_files']} files")
        elif r["ok"]:
            print(f"{YELLOW}exists, size unknown{RESET}")
        else:
            print(f"{RED}FAIL{RESET}")

    # --- Detail on anything that isn't clean ---
    problems = [r for r in results if not r["ok"] or r["message"]]
    if problems:
        print("\n" + "-" * 70)
        print("Details")
        print("-" * 70)
        for r in problems:
            colour = RED if not r["ok"] else YELLOW
            print(f"\n{colour}{r['repo']}{RESET}  "
                  f"({r['repo_type']}, HTTP {r['status']}, "
                  f"from {r.get('conf_key', '?')})")
            print(f"  {r['message']}")
            if not r["ok"]:
                short = r["repo"].split("/")[-1].split("-")[0]
                print(f"  {DIM}Try: python verify_models.py --search {short} "
                      f"--search-type {r['repo_type']}{RESET}")

    # --- Revision SHAs for the reproducibility statement ---
    ok_results = [r for r in results if r["ok"] and r["sha"]]
    if ok_results:
        print("\n" + "-" * 70)
        print("Revision SHAs — record these in the paper")
        print("-" * 70)
        for r in ok_results:
            print(f"  [{r['repo_type']:<7}] {r['repo']:<48} {r['sha']}")

    # --- Size accounting, models and datasets separately ---
    model_bytes = sum(r["total_bytes"] for r in results if r["repo_type"] == "model")
    data_bytes = sum(r["total_bytes"] for r in results if r["repo_type"] == "dataset")
    total = model_bytes + data_bytes
    if total:
        print()
        if model_bytes:
            print(f"Model weights:  {human_gb(model_bytes)}")
        if data_bytes:
            # Dataset repos often host several formats of the same data, so the
            # figure below is an upper bound on what you actually need.
            print(f"Dataset files:  {human_gb(data_bytes)}  "
                  f"{DIM}(upper bound — repos often host multiple formats){RESET}")
        print(f"Total:          {human_gb(total)}")
        if args.target_dir:
            check_disk_space(Path(args.target_dir), total)

    n_fail = sum(1 for r in results if not r["ok"])
    print("\n" + "=" * 70)
    if n_fail:
        print(f"{RED}{n_fail} of {len(results)} repositories failed verification.{RESET}")
        print("Fix models.conf before downloading.")
        sys.exit(1)
    print(f"{GREEN}All {len(results)} repositories verified.{RESET}")


if __name__ == "__main__":
    main()