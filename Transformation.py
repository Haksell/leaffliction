import argparse
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv


def transformation_masked(img, mask):
    return pcv.apply_mask(img, mask, "black")


def transformation_canny(img, mask):
    return pcv.canny_edge_detect(pcv.apply_mask(img, mask, "black"), sigma=1.3)


def transformation_background(img, mask):
    return pcv.apply_mask(img, 255 - mask, "black")


def transformation_analyze(img, mask):
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])
    labeled_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")
    return pcv.analyze.size(img, labeled_mask)


def transformation_original(img, _):
    return img


# TODO: better pseudolandmarks
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


def transformation_clahe(img, _):
    light, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    light = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


def transformations(img):
    clahe = transformation_clahe(img, None)
    bin_img = pcv.threshold.dual_channels(
        rgb_img=clahe,
        x_channel="a",
        y_channel="b",
        points=[(75, 75), (130, 145)],
    )
    mask = pcv.fill_holes(pcv.fill(bin_img=bin_img, size=50))

    images = [
        (transformation_masked(img, mask), "Mask"),
        (transformation_canny(img, mask), "Canny edge detection"),
        (transformation_background(img, mask), "Background"),
        (transformation_analyze(img, mask), "Analyze object"),
        (transformation_original(img, mask), "Original"),
        (transformation_pseudolandmarks(img, mask), "Pseudolandmarks"),
        (pcv.hist_equalization(pcv.rgb2gray(img)), "Histogram equalization"),
        (pcv.gaussian_blur(img, ksize=(11, 11)), "Gaussian blur"),
        (clahe, "CLAHE"),
    ]
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax, (transformed, title) in zip(axes.flat, images):
        ax.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    plt.tight_layout()


def plot_histogram(ax, img, colorspace, title, colors, channel_names):
    histograms = [
        cv2.calcHist([cv2.cvtColor(img, colorspace)], [i], None, [64], [0, 256])
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
        cv2.COLOR_RGB2HSV,
        "HSV Histogram",
        ["purple", "cyan", "orange"],
        ["Hue", "Saturation", "Value"],
    )
    plot_histogram(
        axes[2],
        img,
        cv2.COLOR_RGB2LAB,
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
            "mask",
            "edge",
            "background",
            "analyze",
            "original",
            "pseudolandmarks",
            "equalization",
            "blur",
            "clahe",
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
    return is_file, args


def handle_file(filename):
    img = cv2.imread(filename)
    transformations(img)
    histogram(img)
    plt.show()


def handle_directory(src, dst, transformation):
    assert os.path.isdir(src)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    assert all(map(os.path.isfile, os.listdir(src)))
    for file in os.listdir(src):
        print(file)


def main():
    is_file, args = parse_args()
    if is_file:
        handle_file(args.image)
    else:
        handle_directory(args.src, args.dst, args.transformation)


if __name__ == "__main__":
    main()
