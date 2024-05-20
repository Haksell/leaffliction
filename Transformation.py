import cv2
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

image_path = "garbage.jpg"
image, path, filename = pcv.readimage(filename=image_path)

images = [image]
kernel_sizes = [3, 5, 7, 9, 11]
for k in kernel_sizes:
    images.append(pcv.gaussian_blur(img=image, ksize=(k, k), sigma_x=0, sigma_y=None))

fig, axes = plt.subplots(2, 3)

for i, ax in enumerate(axes.flat):
    ax.axis("off")
    if i < len(images):
        ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax.set_title(
            "Original Image"
            if i == 0
            else f"Gaussian Blur {kernel_sizes[i-1]}x{kernel_sizes[i-1]}"
        )

plt.tight_layout()
plt.show()
