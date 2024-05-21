import cv2
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import sys


def transformation_analyze(image):
    bin_img = pcv.threshold.dual_channels(
        rgb_img=image,
        x_channel="a",
        y_channel="b",
        points=[(80, 80), (125, 140)],
        above=True,
    )
    mask = pcv.fill_holes(pcv.fill(bin_img=bin_img, size=50))
    roi = pcv.roi.rectangle(img=image, x=0, y=0, h=image.shape[0], w=image.shape[1])
    labeled_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")
    return pcv.analyze.size(img=image, labeled_mask=labeled_mask)


def transformations(image):
    images = [
        (image, "Original"),
        (pcv.gaussian_blur(img=image, ksize=(11, 11)), "Gaussian blur"),
        (image, "Original"),
        (image, "Original"),
        (transformation_analyze(image), "Analyze object"),
        (image, "Original"),
    ]
    _, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (transformed, title) in zip(axes.flat, images):
        ax.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    plt.tight_layout()


def plot_histogram(ax, image, colorspace, title, colors, channel_names):
    histograms = [
        cv2.calcHist([cv2.cvtColor(image, colorspace)], [i], None, [64], [0, 256])
        for i in range(3)
    ]
    data = [h.flatten() / h.sum() for h in histograms]
    for i, color in enumerate(colors):
        ax.plot(data[i], color=color, label=channel_names[i])
    ax.set_title(title)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Proportion of Pixels (%)")
    ax.legend()


def histogram(image):
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_histogram(
        axes[0],
        image,
        cv2.COLOR_BGR2RGB,
        "RGB Histogram",
        ["red", "green", "blue"],
        ["Red", "Green", "Blue"],
    )
    plot_histogram(
        axes[1],
        image,
        cv2.COLOR_RGB2HSV,
        "HSV Histogram",
        ["purple", "cyan", "orange"],
        ["Hue", "Saturation", "Value"],
    )
    plot_histogram(
        axes[2],
        image,
        cv2.COLOR_RGB2LAB,
        "LAB Histogram",
        ["gray", "magenta", "yellow"],
        ["L*", "a*", "b*"],
    )
    plt.tight_layout()


def main():
    # TODO: argparse
    image = cv2.imread(sys.argv[1])
    transformations(image)
    # histogram(image) TODO
    plt.show()


if __name__ == "__main__":
    main()
