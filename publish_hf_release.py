from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import get_token


DEFAULT_RELEASE_ROOT = Path("/home/foivos/data/glossapi_work/hf_release")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update the Hugging Face dataset repo for the staged release.")
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-report-every", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.private and args.public:
        raise SystemExit("choose either --private or --public, not both")
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    private = True if args.private else False if args.public else True
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.release_root,
        num_workers=args.num_workers,
        print_report=True,
        print_report_every=args.print_report_every,
    )
    print(f"https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
