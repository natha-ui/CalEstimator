import shutil
from pathlib import Path
import random
import zipfile

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

def extractSmall(input_dir, output_dir, num_classes, num_examples, create_zip=False):
    counts = [len([f for f in Path(input_dir+"/train/apple_pie").rglob("*") if f.is_file()]),
              len([f for f in Path(input_dir+"/val/apple_pie").rglob("*") if f.is_file()]),
              len([f for f in Path(input_dir+"/test/apple_pie").rglob("*") if f.is_file()])]
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    splits = ['train', 'val', 'test']
    

    class_names = random.sample([d.name for d in (input_dir / 'train').iterdir() if d.is_dir()], k=num_classes)
    print(class_names)

    i = 0
    for split in splits:
        for class_name in class_names:
            src_class_dir = input_dir / split / class_name
            dst_class_dir = output_dir / split / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)

            files = [f for f in src_class_dir.iterdir() if f.is_file()]
            count = (int)(num_examples*counts[i])
            sampled = random.sample(files, k=count)

            for file in sampled:
                shutil.copy2(file, dst_class_dir / file.name)

        i += 1
            # if create_zip: zip_path = shutil.make_archive(str(output_dir), 'zip', str(output_dir))

    # print(f"Extracted {num_classes} classes with {num_examples} examples each to: {output_dir}")
    # if create_zip: print(f"Zipped dataset saved to: {zip_path}")

# Example usage:
# extractSmall("food-101", "food-10-small", numClasses=10, numExamples=20)
