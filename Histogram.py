import sys
import cv2
import matplotlib.pyplot as plt


def plot_histogram(ax, data, title, colors, channel_names):
    for i, color in enumerate(colors):
        ax.plot(data[i], color=color, label=channel_names[i])
    ax.set_title(title)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Proportion of Pixels (%)")
    ax.legend()


def calculate_histogram(image):
    histograms = [cv2.calcHist([image], [i], None, [64], [0, 256]) for i in range(3)]
    return [hist.ravel() / hist.sum() for hist in histograms]


def main(filename):
    image = cv2.imread(filename)
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_histogram(
        axes[0],
        calculate_histogram(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
        "RGB Histogram",
        ["red", "green", "blue"],
        ["Red", "Green", "Blue"],
    )
    plot_histogram(
        axes[1],
        calculate_histogram(cv2.cvtColor(image, cv2.COLOR_RGB2HSV)),
        "HSV Histogram",
        ["purple", "cyan", "orange"],
        ["Hue", "Saturation", "Value"],
    )
    plot_histogram(
        axes[2],
        calculate_histogram(cv2.cvtColor(image, cv2.COLOR_RGB2LAB)),
        "LAB Histogram",
        ["gray", "magenta", "yellow"],
        ["L*", "a*", "b*"],
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # TODO: argparse
    main(sys.argv[1])
