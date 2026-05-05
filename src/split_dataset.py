import os
import shutil
import random
from pathlib import Path

SOURCE_DIR = Path("PokemonData")
TARGET_DIR = Path("data/pokemon")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(file_path):
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def split_dataset():
    random.seed(42)

    if not SOURCE_DIR.exists():
        print(f"Source folder not found: {SOURCE_DIR}")
        return

    class_folders = [
        folder for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    ]

    print(f"Found {len(class_folders)} classes.")

    for class_folder in class_folders:
        class_name = class_folder.name

        image_files = [
            file for file in class_folder.iterdir()
            if file.is_file() and is_image_file(file)
        ]

        if len(image_files) == 0:
            print(f"Warning: no images in {class_name}")
            continue

        random.shuffle(image_files)

        total_count = len(image_files)
        train_count = int(total_count * TRAIN_RATIO)
        val_count = int(total_count * VAL_RATIO)

        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]

        split_files = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        for split_name, files in split_files.items():
            target_class_dir = TARGET_DIR / split_name / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            for file in files:
                target_path = target_class_dir / file.name
                shutil.copy2(file, target_path)

        print(
            f"{class_name}: "
            f"train={len(train_files)}, "
            f"val={len(val_files)}, "
            f"test={len(test_files)}"
        )

    print("\nDataset split completed.")
    print(f"Saved to: {TARGET_DIR}")


if __name__ == "__main__":
    split_dataset()