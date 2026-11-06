"""Build merged ADS/OpenADS-derived feature tables from subject directories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medtabx.features.ads_features import (
    build_ads_feature_table,
    build_ads_feature_table_from_root,
    discover_ads_subject_roots,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build ADS feature tables")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Dataset root containing one or more sub-* subject directories.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        action="append",
        default=None,
        help="Explicit subject directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.subject:
        subject_roots = [Path(p) for p in args.subject]
        df = build_ads_feature_table(subject_roots)
    else:
        if args.root is None:
            raise SystemExit("Either --root or at least one --subject must be provided.")
        df = build_ads_feature_table_from_root(args.root)
        subject_roots = discover_ads_subject_roots(args.root)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"saved_csv={output_path}")

    print(f"subjects={len(subject_roots)}")
    print(f"rows={len(df)}")
    print(f"columns={len(df.columns)}")
    print(f"subject_ids={list(df['subject_id'])}")


if __name__ == "__main__":
    main()
