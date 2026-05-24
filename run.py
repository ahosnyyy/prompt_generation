"""
run.py - Entry point for car cabin prompt generation pipeline.

Usage:
    python run.py [--manifest MANIFEST] [--sample-size N] [--output-dir DIR]
"""

import argparse

from src.generate_prompts import generate_prompts
from src.utils import save_prompts_to_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured prompts for synthetic car-cabin image generation."
    )
    parser.add_argument(
        "--manifest",
        default="refs/manifest.yaml",
        help="Path to the reference manifest YAML (default: refs/manifest.yaml)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of prompts PER REF image (default: from config.py SAMPLE_SIZE)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Base output directory (default: data/)",
    )
    args = parser.parse_args()

    print("Generating prompts...")
    records = generate_prompts(
        manifest_path=args.manifest,
        sample_size=args.sample_size,
    )

    print(f"Generated {len(records)} prompt records.")
    save_prompts_to_file(records, base_dir=args.output_dir)


if __name__ == "__main__":
    main()
