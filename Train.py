# [0.5941, 0.9445, 0.9619, 0.9747, 0.9563, 0.9758, 0.9747, 0.9748, 0.9764, 0.9747]

import pandas as pd
import statistics
from tensorflow import keras

NUM_CLASSES = 4  # TODO: automatically from image_dataset_from_directory
IMAGE_SIZE = 64
EPOCHS = 100
MEAN_EPOCHS = 10
assert EPOCHS % MEAN_EPOCHS == 0

ds_train, ds_rest = keras.preprocessing.image_dataset_from_directory(
    "../input/images/images/apple",
    labels="inferred",
    label_mode="categorical",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    interpolation="nearest",
    batch_size=64,
    shuffle=True,
    validation_split=0.3,
    subset="both",
    seed=42,
)
val_size = int(0.5 * len(ds_rest))
ds_valid = ds_rest.take(val_size)
ds_test = ds_rest.skip(val_size)

model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3]),
        # Data Augmentation
        keras.layers.RandomContrast(factor=0.1),
        keras.layers.RandomFlip(mode="horizontal"),
        keras.layers.RandomRotation(factor=0.1),
        # Block One
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
        keras.layers.MaxPool2D(),
        # Block Two
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
        keras.layers.MaxPool2D(),
        # Block Three
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(256, kernel_size=3, activation="relu", padding="same"),
        keras.layers.Conv2D(256, kernel_size=3, activation="relu", padding="same"),
        keras.layers.MaxPool2D(),
        # Head
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(epsilon=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

checkpoint = keras.callbacks.ModelCheckpoint(
    "best_model.keras", save_best_only=True, monitor="val_loss", mode="min"
)
history = model.fit(
    ds_train, validation_data=ds_valid, epochs=EPOCHS, callbacks=[checkpoint]
)
model.load_weights("best_model.keras")

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ["loss", "val_loss"]].plot()
history_frame.loc[:, ["accuracy", "val_accuracy"]].plot()

print(
    [
        round(statistics.mean(history.history["val_accuracy"][i : i + MEAN_EPOCHS]), 4)
        for i in range(0, EPOCHS, MEAN_EPOCHS)
    ]
)

test_loss, test_accuracy = model.evaluate(ds_test)
print(f"Test accuracy: {test_accuracy:.4f}")
