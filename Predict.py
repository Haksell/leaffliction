import argparse
import cv2
import json
import numpy as np
from tensorflow import keras
from Train import IMAGE_SIZE
import Transformation  # noqa
from Transformation import TRANSFORMATION_CHOICES, TRANSFORMATION_IDENTITY


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filenames", nargs="+", type=str, help="Filenames of the image to predict"
    )
    parser.add_argument(
        "--plant",
        type=str,
        choices=["apple", "grape"],
        help="Type of image to predict",
        required=True,
    )
    parser.add_argument(
        "--transformation",
        type=str,
        choices=TRANSFORMATION_CHOICES,
        default=TRANSFORMATION_IDENTITY,
        help="Type of transformation to apply",
    )
    return parser.parse_args()


def plot(img, transformed, prediction):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].imshow(img)
    axes[0].set_title("original")
    axes[1].imshow(transformed)
    axes[1].set_title("transformed")
    fig.suptitle(f"Predicted: {prediction}", fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    transformation = eval(f"Transformation.transformation_{args.transformation}")
    classes = json.load(open(f"{args.plant}.classes"))
    model = keras.models.load_model(f"{args.plant}.keras")
    padding = max(map(len, args.filenames))
    for filename in args.filenames:
        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        transformed = transformation(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), None)
        predictions = model.predict(np.expand_dims(transformed, axis=0), verbose=0)
        prediction = classes[np.argmax(predictions)]
        print(f"{filename:<{padding}}: predicted {prediction}")
        if len(args.filenames) == 1:
            plot(img, transformed, prediction)


if __name__ == "__main__":
    main()
