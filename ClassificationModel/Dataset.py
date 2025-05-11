import shutil
from pathlib import Path

def flatten_dataset(source_root: str, target_root: str, splits=('train','val','test')):
    source_root = Path(source_root)
    target_root = Path(target_root)

    # Create the top-level split folders
    for split in splits:
        (target_root / split).mkdir(parents=True, exist_ok=True)

    # Iterate each class folder
    for class_folder in source_root.iterdir():
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name

        # For each split, copy images into target_root/split/class_name/
        for split in splits:
            src_split_dir = class_folder / split
            if not src_split_dir.exists():
                continue

            dst_class_dir = target_root / split / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in src_split_dir.iterdir():
                if not img_path.is_file():
                    continue

                dst_path = dst_class_dir / img_path.name
                # avoid name collisions
                i = 1
                while dst_path.exists():
                    dst_path = dst_class_dir / f"{img_path.stem}_{i}{img_path.suffix}"
                    i += 1

                shutil.copy2(img_path, dst_path)
                # if you want to move instead of copy:
                # shutil.move(img_path, dst_path)

    print(f"Done! Flattened dataset saved to: {target_root}")

# Example usage:
# flatten_dataset_by_split("food-101-resized", "food-101-ultralytics")


# gets labels and splits from dataset file
def writeYAML(file):
    pass
