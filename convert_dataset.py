#!/usr/bin/env python3
"""
Convert shapenet_airplane_500_models directory structure to the format expected by dataset.py

Expected structure:
    root/
    ├── ShapeNetRendering/
    │   └── 02691156/              # airplane category ID
    │       └── <model_id>/
    │           └── rendering/
    │               └── *.png
    └── ShapeNetVox32/
        └── 02691156/
            └── <model_id>/
                └── model.binvox
"""

import os
import shutil
from pathlib import Path
import argparse


def convert_dataset(source_dir, output_dir, category_id="02691156"):
    """
    Convert shapenet_airplane_500_models to ShapeNetRendering/ShapeNetVox32 structure

    Args:
        source_dir: Path to shapenet_airplane_500_models
        output_dir: Path where ShapeNetRendering and ShapeNetVox32 will be created
        category_id: ShapeNet category ID for airplanes (default: 02691156)
    """

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Create output directories
    rendering_root = output_path / "ShapeNetRendering" / category_id
    voxel_root = output_path / "ShapeNetVox32" / category_id

    rendering_root.mkdir(parents=True, exist_ok=True)
    voxel_root.mkdir(parents=True, exist_ok=True)

    model_dirs = [
        d for d in source_path.iterdir() if d.is_dir() and d.name != "__pycache__"
    ]
    total_models = len(model_dirs)

    converted_count = 0
    skipped_count = 0

    print(f"Found {total_models} model directories")
    print(f"Converting to: {output_path}")

    for idx, model_dir in enumerate(model_dirs):
        model_id = model_dir.name

        # Check for screenshots
        screenshots_dir = model_dir / "screenshots"
        binvox_file = None

        # Look for binvox file (could be .solid.binvox or .surface.binvox)
        models_dir = model_dir / "models"
        if models_dir.exists():
            for f in models_dir.iterdir():
                if f.suffix == ".binvox":
                    binvox_file = f
                    break

        if not screenshots_dir.exists():
            skipped_count += 1
            print(f"[{idx+1}/{total_models}] SKIP {model_id}: no screenshots directory")
            continue

        if not binvox_file:
            skipped_count += 1
            print(f"[{idx+1}/{total_models}] SKIP {model_id}: no binvox file")
            continue

        # Create model directories in output structure
        model_rendering_dir = rendering_root / model_id / "rendering"
        model_voxel_dir = voxel_root / model_id

        model_rendering_dir.mkdir(parents=True, exist_ok=True)
        model_voxel_dir.mkdir(parents=True, exist_ok=True)

        # Copy/link screenshots
        for img_file in screenshots_dir.glob("*.png"):
            src_img = img_file
            dst_img = model_rendering_dir / img_file.name

            # Use symlink if on same filesystem, otherwise copy
            try:
                if dst_img.exists():
                    dst_img.unlink()
                os.symlink(src_img, dst_img)
            except OSError:
                # Fall back to copy if symlink fails (e.g., different filesystems)
                shutil.copy2(src_img, dst_img)

        # Copy/link binvox file as model.binvox
        dst_binvox = model_voxel_dir / "model.binvox"
        try:
            if dst_binvox.exists():
                dst_binvox.unlink()
            os.symlink(binvox_file, dst_binvox)
        except OSError:
            shutil.copy2(binvox_file, dst_binvox)

        converted_count += 1
        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{total_models}] Converted {converted_count} models...")

    print("\n✓ Conversion complete!")
    print(f"  Converted: {converted_count} models")
    print(f"  Skipped: {skipped_count} models")
    print("\nOutput structure created at:")
    print(f"  - {rendering_root}")
    print(f"  - {voxel_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert shapenet_airplane_500_models to expected format"
    )
    parser.add_argument(
        "--source",
        default="/Users/Omega/dev/syde577-project/shapenet_airplane_500_models",
        help="Path to shapenet_airplane_500_models directory",
    )
    parser.add_argument(
        "--output",
        default="/Users/Omega/dev/syde577-project/shapenet_airplane_500_models",
        help="Output directory (will create ShapeNetRendering and ShapeNetVox32 subdirs)",
    )
    parser.add_argument(
        "--category",
        default="02691156",
        help="ShapeNet category ID (default: 02691156 for airplanes)",
    )

    args = parser.parse_args()

    convert_dataset(args.source, args.output, args.category)
