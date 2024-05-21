import argparse
from functools import wraps
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv


def get_mask(img):
    clahe = transformation_clahe(img, None)
    bin_img = pcv.threshold.dual_channels(
        rgb_img=clahe,
        x_channel="a",
        y_channel="b",
        points=[(70, 70), (135, 150)],
    )
    return pcv.fill_holes(pcv.fill(bin_img=bin_img, size=50))


def gen_mask(func):
    @wraps(func)
    def wrapper(img, mask):
        return func(img, get_mask(img) if mask is None else mask)

    return wrapper


@gen_mask
def transformation_mask(img, mask):
    return pcv.apply_mask(img, mask, "black")


@gen_mask
def transformation_canny(img, mask):
    return pcv.canny_edge_detect(pcv.apply_mask(img, mask, "black"), sigma=1.3)


@gen_mask
def transformation_background(img, mask):
    return pcv.apply_mask(img, 255 - mask, "black")


def transformation_analyze(img, mask):
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])
    labeled_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")
    return pcv.analyze.size(img, labeled_mask)


def transformation_original(img, _):
    return img


# TODO: better pseudolandmarks
@gen_mask
def transformation_pseudolandmarks(img, mask):
    def pseudolandmarks_place_dots(dots, color):
        for pos in dots:
            cv2.circle(
                cpy,
                tuple(map(int, pos[0])),
                3,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

    left, right, center = pcv.homology.y_axis_pseudolandmarks(img=img, mask=mask)
    cpy = np.copy(img)
    pseudolandmarks_place_dots(left, (255, 0, 0))
    pseudolandmarks_place_dots(right, (255, 0, 255))
    pseudolandmarks_place_dots(center, (0, 79, 255))
    return cpy


def transformation_equalization(img, _):
    return pcv.hist_equalization(pcv.rgb2gray(img))


def transformation_blur(img, _):
    return pcv.gaussian_blur(img, ksize=(11, 11))


def transformation_clahe(img, _):
    light, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    light = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


TRANSFORMATIONS = [
    (transformation_mask, "Mask"),
    (transformation_canny, "Canny edge detection"),
    (transformation_background, "Background"),
    (transformation_analyze, "Analyze object"),
    (transformation_original, "Original"),
    (transformation_pseudolandmarks, "Pseudolandmarks"),
    (transformation_equalization, "Histogram equalization"),
    (transformation_blur, "Gaussian blur"),
    (transformation_clahe, "CLAHE"),
]


def transformations(img):
    mask = get_mask(img)
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax, (transformation, title) in zip(axes.flat, TRANSFORMATIONS):
        ax.imshow(cv2.cvtColor(transformation(img, mask), cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    plt.tight_layout()


def plot_histogram(ax, img, colorspace, title, colors, channel_names):
    histograms = [
        cv2.calcHist(
            [cv2.cvtColor(img, colorspace) if colorspace else img],
            [i],
            None,
            [64],
            [0, 256],
        )
        for i in range(3)
    ]
    data = [h.flatten() / h.sum() for h in histograms]
    for i, color in enumerate(colors):
        ax.plot(data[i], color=color, label=channel_names[i])
    ax.set_title(title)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Proportion of Pixels (%)")
    ax.legend()


def histogram(img):
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_histogram(
        axes[0],
        img,
        cv2.COLOR_BGR2RGB,
        "RGB Histogram",
        ["red", "green", "blue"],
        ["Red", "Green", "Blue"],
    )
    plot_histogram(
        axes[1],
        img,
        cv2.COLOR_BGR2HSV,
        "HSV Histogram",
        ["purple", "cyan", "orange"],
        ["Hue", "Saturation", "Value"],
    )
    plot_histogram(
        axes[2],
        img,
        cv2.COLOR_BGR2LAB,
        "LAB Histogram",
        ["gray", "magenta", "yellow"],
        ["L*", "a*", "b*"],
    )
    plt.tight_layout()


def parse_args():
    parser = argparse.ArgumentParser(description="Imaging processing")
    parser.add_argument("--image", type=str, help="Path to a single image to process")
    parser.add_argument("--src", help="Source directory")
    parser.add_argument("--dst", help="Destination directory")
    parser.add_argument(
        "--transformation",
        type=str,
        choices=[
            f.__name__.removeprefix("transformation_") for f, _ in TRANSFORMATIONS
        ],
        help="Type of transformation to apply",
    )
    args = parser.parse_args()
    present = [
        args.image is not None,
        args.src is not None,
        args.dst is not None,
        args.transformation is not None,
    ]
    is_file = present == [True, False, False, False]
    is_directory = present == [False, True, True, True]
    assert (
        is_file or is_directory
    ), "you should provide --image, or --src/--dst/--transformation"
    assert args.src is None or "/" not in args.src
    assert args.dst is None or "/" not in args.dst
    return is_file, args


def handle_file(filename):
    img = cv2.imread(filename)
    transformations(img)
    histogram(img)
    plt.show()


def handle_directory(src, dst, transformation):
    transformation = eval(f"transformation_{transformation}")
    assert os.path.isdir(src)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    for root_src, dirs, files in os.walk(src):
        root_dst = root_src.replace(src, dst)
        for d in dirs:
            os.makedirs(os.path.join(root_dst, d), exist_ok=True)
        for f in files:
            img = cv2.imread(os.path.join(root_src, f))
            cv2.imwrite(os.path.join(root_dst, f), transformation(img, None))


def main():
    is_file, args = parse_args()
    if is_file:
        handle_file(args.image)
    else:
        handle_directory(args.src, args.dst, args.transformation)


if __name__ == "__main__":
    main()
