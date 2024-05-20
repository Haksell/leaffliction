import sys
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

BINS = 64


def plot_histogram(ax, data, name):
    ax.plot(
        data["pixel intensity"],
        data["proportion of pixels (%)"],
        label=name.title(),
        color=name,
    )


def main(filename):
    img, _, _ = pcv.readimage(filename)
    _, hist_data = pcv.visualize.histogram(img=img, hist_data=True, bins=BINS)
    blue, green, red = (
        hist_data[:BINS],
        hist_data[BINS : 2 * BINS],
        hist_data[2 * BINS :],
    )

    _, ax = plt.subplots()

    plot_histogram(ax, blue, "blue")
    plot_histogram(ax, green, "green")
    plot_histogram(ax, red, "red")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Proportion of Pixels (%)")
    ax.set_title("Color Channel Intensity Histograms")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # TODO: argparse
    main(sys.argv[1])
