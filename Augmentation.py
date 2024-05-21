import argparse
from itertools import product
import os
import random
import shutil
import sys
from PIL import Image, ImageEnhance, ImageFilter
from matplotlib import pyplot as plt


DEBUG_PERCENTAGE = 2
FLIPS = list(Image.Transpose)


def flip(img):
    return img.transpose(random.choice(FLIPS))


def rotate(img):
    return img.rotate(random.randint(10, 350))


def zoom(img):
    percentage = random.randint(5, 25)
    horizontal_crop = img.width * percentage // 100
    vertical_crop = img.height * percentage // 100
    return img.crop(
        (
            horizontal_crop,
            vertical_crop,
            img.width - horizontal_crop,
            img.height - vertical_crop,
        )
    ).resize(img.size)


def random_dissimilar():
    return (
        random.uniform(0.4, 0.8) if random.random() < 0.5 else random.uniform(1.25, 2.5)
    )


def contrast(img):
    return ImageEnhance.Contrast(img).enhance(random_dissimilar())


def brightness(img):
    return ImageEnhance.Brightness(img).enhance(random_dissimilar())


def blur(img):
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.5)))


def original(img):
    return img


AUGMENTATIONS = [flip, rotate, zoom, contrast, brightness, blur]
AUGMENTED_PREFIX = "augmented_"


def save(img, path, augmentation, *, single_file):
    name, ext = os.path.splitext(path)
    path = f"{name}_{augmentation}{ext}"
    if not single_file:
        path = AUGMENTED_PREFIX + path
    img.save(path)
    print(f'Transformation saved to "{path}"')


def augment_file(path, augmentations):
    single_file = original not in augmentations
    subplots = []
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            for augmentation in augmentations:
                augmented = augmentation(img)
                save(augmented, path, augmentation.__name__, single_file=single_file)
                subplots.append((augmented, augmentation.__name__))
    except (IOError, SyntaxError) as e:
        print(f"Error loading image {path}: {e}")
        sys.exit(1)
    if single_file:
        _, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax, (subplot, title) in zip(axes.flat, subplots):
            ax.imshow(subplot)
            ax.set_title(title)
        plt.tight_layout()
        plt.show()


def augment_directory(args):
    files_count = dict()
    augmentations_count = dict()
    # count files
    for root, dirs, files in os.walk(args.path):
        assert dirs or files, f"{root} is empty"
        assert not dirs or not files, f"{root} contains both files and directories"
        if files:
            num_files = (
                round(len(files) * DEBUG_PERCENTAGE / 100) if args.debug else len(files)
            )
            files_count[root] = num_files
            augmentations_count[root] = num_files * (len(AUGMENTATIONS) + 1)
    # balance directories and augment files
    for root, dirs, files in os.walk(args.path):
        if files:
            files = [
                os.path.join(root, f) for f in random.sample(files, k=files_count[root])
            ]
            augmented_path = AUGMENTED_PREFIX + root
            shutil.rmtree(augmented_path, ignore_errors=True)
            os.makedirs(augmented_path, exist_ok=True)
            augmentations = {f: [original] for f in files}
            for f, augmentation in random.sample(
                list(product(files, AUGMENTATIONS)),
                k=augmentations_count[root] - files_count[root],
            ):
                augmentations[f].append(augmentation)
            for f, a in augmentations.items():
                augment_file(f, a)
        elif args.balanced:
            subdirs = [os.path.join(root, d) for d in dirs]
            all_files = all(sd in files_count for sd in subdirs)
            all_dirs = all(sd not in files_count for sd in subdirs)
            assert all_files or all_dirs, f"failed to balance {root}"
            if all_files:
                min_count = min(map(files_count.get, subdirs))
                for sd in subdirs:
                    augmentations_count[sd] = max(
                        files_count[sd], min_count * (len(AUGMENTATIONS) + 1)
                    )
    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Path of the file or directory to augment"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=f"Take only {DEBUG_PERCENTAGE}%% of images in each directory",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Create directories with the same image count",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.path):
        print(f"File not found: {args.path}")
        sys.exit(1)
    elif os.path.isfile(args.path):
        assert not args.debug, "--debug mode is incompatible with single file"
        assert not args.balanced, "--balanced mode is incompatible with single file"
        augment_file(args.path, AUGMENTATIONS)
    else:
        augment_directory(args)


if __name__ == "__main__":
    main()
