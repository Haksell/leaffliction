import cv2
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import sys


def transformations(image):
    kernel_sizes = [1, 5, 9, 13, 17, 21]
    images = [
        pcv.gaussian_blur(img=image, ksize=(k, k), sigma_x=0, sigma_y=None)
        for k in kernel_sizes
    ]
    _, axes = plt.subplots(2, 3)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Gaussian Blur {kernel_sizes[i-1]}x{kernel_sizes[i-1]}")
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
