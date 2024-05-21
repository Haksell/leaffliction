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


# TODO: better pseudolandmarks
def pseudolandmarks(img, mask):
    left, right, center = pcv.homology.y_axis_pseudolandmarks(img=img, mask=mask)
    img = np.copy(img)
    pseudolandmarks_place_dots(img, left, (255, 0, 0))
    pseudolandmarks_place_dots(img, right, (255, 0, 255))
    pseudolandmarks_place_dots(img, center, (0, 79, 255))
    return img


def contrast_limited_adaptive_histogram_equalization(img):
    light, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    light = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


def transformations(img):
    clahe = contrast_limited_adaptive_histogram_equalization(img)
    bin_img = pcv.threshold.dual_channels(
        rgb_img=clahe,
        x_channel="a",
        y_channel="b",
        points=[(75, 75), (130, 145)],
    )
    mask = pcv.fill_holes(pcv.fill(bin_img=bin_img, size=50))
    masked = pcv.apply_mask(img, mask, "black")
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])
    labeled_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")

    images = [
        (masked, "Mask"),
        (pcv.canny_edge_detect(masked, sigma=1.3), "Canny edge detection"),
        (pcv.apply_mask(img, 255 - mask, "black"), "Background"),
        (pcv.analyze.size(img, labeled_mask), "Analyze object"),
        (img, "Original"),
        (pseudolandmarks(img, mask), "Pseudolandmarks"),
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


def main():
    # TODO: argparse
    img = cv2.imread(sys.argv[1])
    transformations(img)
    # histogram(img) TODO
    plt.show()


if __name__ == "__main__":
    main()
