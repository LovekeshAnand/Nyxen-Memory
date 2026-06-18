#!/usr/bin/env python3
"""Download official LoCoMo and LongMemEval benchmark datasets."""
import argparse
import os
import urllib.request

DATASETS = {
    "locomo10": {
        "url": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        "path": "data/locomo10.json",
    },
    "longmemeval_oracle": {
        "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
        "path": "data/longmemeval_oracle.json",
    },
    "longmemeval_s": {
        "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
        "path": "data/longmemeval_s_cleaned.json",
    },
}


def download(name: str, force: bool = False):
    if name not in DATASETS:
        print(f"Unknown dataset: {name}. Available: {', '.join(DATASETS)}")
        return False
    info = DATASETS[name]
    if os.path.exists(info["path"]) and not force:
        print(f"[skip] {info['path']} already exists")
        return True
    os.makedirs(os.path.dirname(info["path"]), exist_ok=True)
    print(f"Downloading {name} -> {info['path']}...")
    urllib.request.urlretrieve(info["url"], info["path"])
    print(f"Done ({os.path.getsize(info['path']) / 1024 / 1024:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--dataset", default="all", help="locomo10, longmemeval_oracle, longmemeval_s, or all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        download(name, force=args.force)


if __name__ == "__main__":
    main()
