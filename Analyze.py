import sys
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

pcv.params.dpi = 100
pcv.params.text_size = 20
pcv.params.text_thickness = 20


def process_and_analyze_image(filename):
    img, _, _ = pcv.readimage(filename=filename)

    thresh1 = pcv.threshold.dual_channels(
        rgb_img=img,
        x_channel="a",
        y_channel="b",
        points=[(80, 80), (125, 140)],
        above=True,
    )

    mask = pcv.fill(bin_img=thresh1, size=50)
    mask = pcv.fill_holes(mask)
    roi1 = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])
    labeled_mask = pcv.roi.filter(mask=mask, roi=roi1, roi_type="partial")
    analysis_image = pcv.analyze.size(img=img, labeled_mask=labeled_mask)
    return analysis_image


filenames = sys.argv[1:5]
fig, axs = plt.subplots(2, 2)

for i, filename in enumerate(sys.argv[1:5]):
    row = i // 2
    col = i % 2
    axs[row, col].imshow(process_and_analyze_image(filename))
    axs[row, col].set_title(f"Image {i + 1}")

plt.tight_layout()
plt.show()