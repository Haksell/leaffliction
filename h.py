import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

BINS = 64

img, _, _ = pcv.readimage(filename="garbage.jpg")
_, hist_data = pcv.visualize.histogram(img=img, hist_data=True, bins=BINS)
blue, green, red = hist_data[:BINS], hist_data[BINS : 2 * BINS], hist_data[2 * BINS :]
print(blue)
print(green)
print(red)

fig, ax = plt.subplots()

# Plot histograms
ax.plot(
    blue["pixel intensity"],
    blue["proportion of pixels (%)"],
    label="Blue",
    color="blue",
)
ax.plot(
    green["pixel intensity"],
    green["proportion of pixels (%)"],
    label="Green",
    color="green",
)
ax.plot(
    red["pixel intensity"], red["proportion of pixels (%)"], label="Red", color="red"
)

# Labeling
ax.set_xlabel("Pixel Intensity")
ax.set_ylabel("Histogram Count")
ax.set_title("Color Channel Intensity Histograms")
ax.legend()

# Show plot
plt.tight_layout()
plt.show()
