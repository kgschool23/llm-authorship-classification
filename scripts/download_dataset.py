"""Download a pinned MixInstruct snapshot without API keys or extra packages."""
import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c7f77a4ef0515a99d1752c6387c22366d6922da6"
BASE = f"https://huggingface.co/datasets/llm-blender/mix-instruct/resolve/{REVISION}"
FILES = ("val_data_prepared.jsonl", "test_data_prepared.jsonl", "README.md")
EXPECTED_SHA256 = {
    "val_data_prepared.jsonl": "2114785a88bbd31fd8f7416ed131949cf5b054bcd70041bc4aae1e5c63931b7f",
    "test_data_prepared.jsonl": "eda4a320757876a37dfcecb21ec9a53a55bd4e3ad620216c3c777f5b542dfc78",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "data/raw/mixinstruct")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name in FILES:
        target = args.destination / name
        url = f"{BASE}/{name}"
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": "CMPSC448-research/1.0"})
            print(f"Downloading {name}...", flush=True)
            try:
                with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                    total = int(response.headers.get("Content-Length", 0))
                    received = 0
                    last_report = -1
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        received += len(chunk)
                        if received // (10 * 1024 * 1024) != last_report:
                            print(f"  {received / 1024**2:.1f} MiB", flush=True)
                            last_report = received // (10 * 1024 * 1024)
                    if total and received != total:
                        raise RuntimeError(f"Incomplete download: {received} of {total} bytes")
                os.replace(partial, target)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        print(f"Available: {name} ({target.stat().st_size / 1024**2:.1f} MiB)")
        checksum = sha256(target)
        if name in EXPECTED_SHA256 and checksum != EXPECTED_SHA256[name]:
            raise RuntimeError(f"Checksum mismatch for {name}; remove the local file and retry")
        records.append({"filename": name, "url": url, "bytes": target.stat().st_size, "sha256": checksum})
    manifest = {"dataset": "llm-blender/mix-instruct", "revision": REVISION, "files": records}
    (args.destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Download complete.")


if __name__ == "__main__":
    main()
