from numpy import ndarray
from plantcv import plantcv as pcv
import argparse
import cv2
import numpy as np
import os


top_left = (0, 0)
bottom_right = (250, 250)
roi_contours = [(top_left, bottom_right)]


def options():
    parser = argparse.ArgumentParser(description="Imaging processing")
    parser.add_argument("-src", help="Source directory", required=False)
    parser.add_argument("-dst", help="Destination directory", required=False)
    parser.add_argument(
        "-gau", help="Apply Gaussian blur to pictures", action="store_true"
    )
    parser.add_argument("-msk", help="Apply mask to pictures", action="store_true")
    parser.add_argument("-roi", help="Apply ROI to pictures", action="store_true")
    parser.add_argument("-anz", help="Analyze pictures", action="store_true")
    parser.add_argument("-psd", help="Generate Pseudolandmarks", action="store_true")
    parser.add_argument("file_path", help="Path to the image file", nargs="?")
    args = parser.parse_args()
    return args


def gaussian_blur(s_thresh: ndarray) -> ndarray:
    """
    Apply Gaussian blur to the picture
    """
    # Apply Gaussian blur
    transformed_img = pcv.gaussian_blur(
        img=s_thresh, ksize=(1, 1), sigma_x=0, sigma_y=None
    )
    return transformed_img


def mask(s_thresh: ndarray, img: ndarray) -> ndarray:
    """
    Mask the image
    """
    # Median blur
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    # Convert RGB to LAB and extract the blue (cyan) channel
    b = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    b_cnt = pcv.threshold.binary(gray_img=b, threshold=130, object_type="light")
    # Combine the blue and saturation images
    bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_cnt)
    # Apply mask (for VIS images, mask_color=white)
    transformed_img = pcv.apply_mask(img=img, mask=bs, mask_color="white")
    return transformed_img


def create_roi(img_arr: ndarray):
    """
    Apply region of interest to the image
    """
    img_copy = np.copy(img_arr)
    cv2.rectangle(img_copy, top_left, bottom_right, (255, 0, 0), 2)
    roi = pcv.roi.rectangle(img=img_arr, x=0, y=0, h=255, w=255)
    return img_copy, roi


def analize_obj(s_thresh: ndarray, img: ndarray) -> ndarray:
    """
    Analyze the object in the image
    """
    # inside roi, find objects
    img_copy, roi = create_roi(img)
    mask = pcv.roi.filter(mask=s_thresh, roi=roi, roi_type="partial")
    pcv.plot_image(mask)


def transform_file(args, file):
    """
    transform a single file
    """
    img, path, filename = pcv.readimage(filename=file)

    # Convert to HSV and use the saturation channel
    s = pcv.rgb2gray_hsv(rgb_img=img, channel="s")

    # Adjusted threshold
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=85, object_type="light")

    # Transform the image
    if args.gau:
        transformed_img = gaussian_blur(s_thresh)
    elif args.msk:
        transformed_img = mask(s_thresh, img)
    elif args.roi:
        transformed_img = create_roi(img)
    elif args.anz:
        transformed_img = analize_obj(s_thresh, img)
    elif args.psd:
        print("Generating Pseudolandmarks")
    pcv.plot_image(transformed_img)


def transform_dir_file(args):
    """
    transform pictures
    """
    print(args.src)
    if os.path.isdir(args.src):
        for file in os.listdir(args.src):
            file_path = os.path.join(args.src, file)
            if os.path.isfile(file_path):
                transform_file(args, file_path)
    else:
        transform_file(args, args.src)


def main():
    print(pcv.__version__)
    args = options()
    if args.file_path:
        print(f"Processing file at {args.file_path}")
        transform_file(args, args.file_path)
    else:
        if args.src and args.dst:
            if not any([args.gau, args.msk, args.roi, args.anz, args.psd]):
                print("No operation provided")
                exit(1)
            transform_dir_file(args)
        else:
            print("No file or directory provided")
            exit(1)


if __name__ == "__main__":
    main()
