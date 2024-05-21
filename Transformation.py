import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv
import sys


def pseudolandmarks_place_dots(img, dots, color):
    for pos in dots:
        cv2.circle(
            img,
            tuple(map(int, pos[0])),
            3,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )


def pseudolandmarks(img, mask):
    left, right, center = pcv.homology.y_axis_pseudolandmarks(img=img, mask=mask)
    img = np.copy(img)
    pseudolandmarks_place_dots(img, left, (255, 0, 0))
    pseudolandmarks_place_dots(img, right, (255, 0, 255))
    pseudolandmarks_place_dots(img, center, (0, 79, 255))
    return img


def transformations(img):
    bin_img = pcv.threshold.dual_channels(
        rgb_img=img,
        x_channel="a",
        y_channel="b",
        points=[(80, 80), (125, 140)],
        above=True,
    )
    mask = pcv.fill_holes(pcv.fill(bin_img=bin_img, size=50))
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])
    labeled_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")

    images = [
        (img, "Original"),
        (pcv.gaussian_blur(img=img, ksize=(11, 11)), "Gaussian blur"),
        (pcv.apply_mask(img, mask, "black"), "Mask"),
        (img, "Original"),
        (pcv.analyze.size(img=img, labeled_mask=labeled_mask), "Analyze object"),
        (pseudolandmarks(img, mask), "Pseudolandmarks"),
    ]
    _, axes = plt.subplots(2, 3, figsize=(12, 8))
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


def main():
    # TODO: argparse
    img = cv2.imread(sys.argv[1])
    transformations(img)
    # histogram(img) TODO
    plt.show()


if __name__ == "__main__":
    main()
