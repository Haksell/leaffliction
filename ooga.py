import cv2
from plantcv import plantcv as pcv


def process_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Set up PlantCV
    pcv.params.debug = "plot"  # Set debug mode
    pcv.params.debug_outdir = "./"  # Directory for output images

    # Convert image to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert RGB image to LAB and extract the Blue-Yellow channel
    a = pcv.rgb2gray_lab(rgb_img, "l")

    # Threshold the a channel image (this is somewhat adaptive)
    a_thresh = pcv.threshold.otsu(a, "dark")

    # Apply median blur to clean up small noise
    a_blurred = cv2.medianBlur(a_thresh, 5)

    # Identify objects
    id_objects, obj_hierarchy = pcv.find_objects(rgb_img, a_blurred)

    # Define the region of interest (ROI)
    roi_contour, roi_hierarchy = pcv.roi.rectangle(
        img=rgb_img, x=0, y=0, h=rgb_img.shape[0], w=rgb_img.shape[1]
    )

    # Keep objects that overlap with the ROI
    roi_objects, roi_obj_hierarchy, kept_mask, obj_area = pcv.roi_objects(
        img=rgb_img,
        roi_contour=roi_contour,
        roi_hierarchy=roi_hierarchy,
        object_contour=id_objects,
        obj_hierarchy=obj_hierarchy,
        roi_type="partial",
    )

    # Display results
    pcv.plot_image(rgb_img)
    pcv.plot_image(kept_mask)

    # Save the mask image if needed
    cv2.imwrite("masked_image.jpg", kept_mask)


# Example usage:
process_image("garbage.jpg")
