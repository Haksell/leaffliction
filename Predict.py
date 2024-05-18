from tensorflow import keras
import numpy as np
from PIL import Image
import sys


def load_and_preprocess_image(image_path, image_size):
    image = Image.open(image_path)
    image = image.resize((image_size, image_size))
    image = np.array(image)
    image = image / 255.0
    return image


def main(image_path):
    model = keras.models.load_model("best_model.keras")  # MODEL_FILENAME
    processed_image = load_and_preprocess_image(image_path, 64)  # IMAGE_SIZE
    predictions = model.predict(np.expand_dims(processed_image, axis=0))
    prediction = np.argmax(predictions)
    print(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main(sys.argv[1])
