import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "Vehicles"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

CLASSES = [
    "Cars",
    "Bikes",
    "Auto_Rickshaws",
    "Motorcycles"
]


# ============================================================
# READ DATASET
# ============================================================

images = []
labels = []

print("==========================================")
print("READING DATASET")
print("==========================================")


for class_index, class_name in enumerate(CLASSES):

    class_folder = os.path.join(
        DATA_PATH,
        class_name
    )

    print(f"\nReading folder: {class_folder}")

    if not os.path.exists(class_folder):
        print("Folder not found!")
        continue

    for file_name in os.listdir(class_folder):

        file_path = os.path.join(
            class_folder,
            file_name
        )

        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):

            try:

                image = tf.keras.utils.load_img(
                    file_path,
                    target_size=(
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    )
                )

                image_array = tf.keras.utils.img_to_array(
                    image
                )

                images.append(image_array)
                labels.append(class_index)

            except Exception as e:

                print(
                    f"Error reading: {file_path}"
                )

                print(e)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(images)
y = np.array(labels)


print("\n==========================================")
print("DATA INFORMATION")
print("==========================================")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Total images:", len(X))
print("Total labels:", len(y))


# ============================================================
# NORMALIZATION
# ============================================================

X = X / 255.0


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n==========================================")
print("TRAIN / TEST DATA")
print("==========================================")

print("Training images:", len(X_train))
print("Testing images:", len(X_test))


# ============================================================
# CREATE MODEL
# ============================================================

print("\n==========================================")
print("CREATING MODEL")
print("==========================================")

model = create_model()

model.summary()


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n==========================================")
print("START TRAINING")
print("==========================================")

history = model.fit(

    X_train,

    y_train,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_split=0.20
)


# ============================================================
# TEST MODEL
# ============================================================

print("\n==========================================")
print("TESTING MODEL")
print("==========================================")

test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test
)


print("Test Loss:", test_loss)

print(
    "Test Accuracy:",
    test_accuracy
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = "vehicle_classifier.keras"

model.save(MODEL_PATH)


print("\n==========================================")
print("MODEL SAVED")
print("==========================================")

print("Model:", MODEL_PATH)


# ============================================================
# TRAINING ACCURACY GRAPH
# ============================================================

plt.plot(
    history.history["accuracy"]
)

plt.plot(
    history.history["val_accuracy"]
)

plt.title(
    "Training Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend([
    "Training",
    "Validation"
])

plt.show()