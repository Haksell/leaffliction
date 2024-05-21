import os
import pandas as pd
import statistics
from tensorflow import keras

IMAGE_SIZE = 64
CROP_SIZE = round(IMAGE_SIZE * 0.9)
EPOCHS = 100
MEAN_EPOCHS = 10
assert EPOCHS % MEAN_EPOCHS == 0
PLANT = "apple"
DIR = f"images/{PLANT}"
NUM_CLASSES = len(next(os.walk(DIR))[1])

ds_train, ds_valid = keras.preprocessing.image_dataset_from_directory(
    DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    interpolation="nearest",
    batch_size=64,
    shuffle=True,
    validation_split=0.2,
    subset="both",
    seed=42,
)

model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3]),
        # Data Augmentation
        keras.layers.RandomContrast(factor=0.1),
        keras.layers.RandomFlip(mode="horizontal_and_vertical"),
        keras.layers.RandomRotation(factor=0.1),
        keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        keras.layers.RandomBrightness(factor=0.1),
        keras.layers.RandomCrop(height=CROP_SIZE, width=CROP_SIZE),
        keras.layers.Rescaling(scale=1 / 127.5, offset=-1),
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
    f"{PLANT}.keras", save_best_only=True, monitor="val_loss", mode="min"
)
lr_scheduler = keras.callbacks.LearningRateScheduler(lambda _, lr: lr * 0.97)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=EPOCHS,
    callbacks=[checkpoint, lr_scheduler],
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ["loss", "val_loss"]].plot()
history_frame.loc[:, ["accuracy", "val_accuracy"]].plot()

print(
    [
        round(statistics.mean(history.history["val_accuracy"][i : i + MEAN_EPOCHS]), 4)
        for i in range(0, EPOCHS, MEAN_EPOCHS)
    ]
)
