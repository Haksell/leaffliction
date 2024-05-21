import sys
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt


def process_and_analyze_image(filename):
    img, _, _ = pcv.readimage(filename=filename)

    bin_img = pcv.threshold.dual_channels(
        rgb_img=img,
        x_channel="a",
        y_channel="b",
        points=[(80, 80), (125, 140)],
        above=True,
    )

    mask = pcv.fill(bin_img=bin_img, size=50)
    mask = pcv.fill_holes(mask)
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])
    labeled_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")
    return pcv.analyze.size(img=img, labeled_mask=labeled_mask)


def main():
    _, axs = plt.subplots(2, 2)
    for i, filename in enumerate(sys.argv[1:5]):
        row = i // 2
        col = i % 2
        axs[row, col].axis("off")
        axs[row, col].imshow(process_and_analyze_image(filename))
        axs[row, col].set_title(f"Image {i + 1}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
