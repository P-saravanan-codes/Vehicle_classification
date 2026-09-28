import os 
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models
from sklearn.model_selection import train_test_split

# Data path 
data_path = "Vehicles"

image_size = 128 
batch_size = 32
epoch =20

# class name

classes = ["Cars","Bikes","Auto_Rickshaws","Motorcycles"]

# Read images using for loop
images = []
labels = []

print("Reading images...")

for class_index, class_name in enumerate(classes):

    class_folder = os.path.join(data_path, class_name)

    print(f"\nReading folder: {class_folder}")

    for file_name in os.listdir(class_folder):

        file_path = os.path.join(class_folder, file_name)

        # Check whether it is an image
        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):

            try:

                # ------------------------------------------------
                # Read image
                # ------------------------------------------------
                image = tf.keras.utils.load_img(
                    file_path,
                    target_size=(image_size, image_size)
                )

                # ------------------------------------------------
                # Image -> NumPy array
                # ------------------------------------------------
                image_array = tf.keras.utils.img_to_array(image)

                # ------------------------------------------------
                # Store image
                # ------------------------------------------------
                images.append(image_array)

                # ------------------------------------------------
                # Store label
                # car = 0
                # bike = 1
                # ------------------------------------------------
                labels.append(class_index)

            except Exception as e:

                print(f"Error reading {file_path}")
                print(e)

X = np.array(images)
y = np.array(labels)

print("\n================================")
print("DATA INFORMATION")
print("================================")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Total images:", len(X))
print("Total labels:", len(y))

# ============================================================
# 4. NORMALIZE IMAGE
# ============================================================

# Pixel values:
#
# Before:
# 0 - 255
#
# After:
# 0 - 1
#
# Example:
# 255 -> 1.0
# 128 -> 0.501
# 0   -> 0.0

X = X / 255.0


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n================================")
print("TRAIN / TEST DATA")
print("================================")

print("Training images:", len(X_train))
print("Testing images:", len(X_test))


# ============================================================
# 6. CREATE CNN MODEL
# ============================================================

model = models.Sequential([

    # Input
    layers.Input(shape=(image_size, image_size, 3)),

    # --------------------------------------------------------
    # CNN Layer 1
    # --------------------------------------------------------
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),


    # --------------------------------------------------------
    # CNN Layer 2
    # --------------------------------------------------------
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),


    # --------------------------------------------------------
    # CNN Layer 3
    # --------------------------------------------------------
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),


    # --------------------------------------------------------
    # Convert feature maps -> vector
    # --------------------------------------------------------
    layers.Flatten(),


    # --------------------------------------------------------
    # Fully Connected Layer
    # --------------------------------------------------------
    layers.Dense(
        128,
        activation="relu"
    ),


    # --------------------------------------------------------
    # Output
    #
    # 0 -> car
    # 1 -> bike
    # --------------------------------------------------------
    layers.Dense(
        2,
        activation="softmax"
    )
])


# ============================================================
# 7. SHOW MODEL
# ============================================================

model.summary()


# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\n================================")
print("START TRAINING")
print("================================")

history = model.fit(
    X_train,
    y_train,
    epochs=epoch,
    batch_size=batch_size,
    validation_split=0.20
)


# ============================================================
# 10. TEST MODEL
# ============================================================

print("\n================================")
print("TESTING MODEL")
print("================================")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# 11. SAVE MODEL
# ============================================================

model.save("vehicle_classifier.keras")

print("\nModel saved as:")
print("vehicle_classifier.keras")


# ============================================================
# 12. PLOT TRAINING ACCURACY
# ============================================================

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Training",
    "Validation"
])

plt.show()


# ============================================================
# 13. PREDICT ONE TEST IMAGE
# ============================================================

image_path = os.path.join(
    data_path,
    "car",
    os.listdir(os.path.join(data_path, "car"))[0]
)

print("\n================================")
print("PREDICTION")
print("================================")

image = tf.keras.utils.load_img(
    image_path,
    target_size=(image_size, image_size)
)

image_array = tf.keras.utils.img_to_array(image)

# Image -> array
print("Original array shape:", image_array.shape)

# Normalize
image_array = image_array / 255.0

# Add batch dimension
image_array = np.expand_dims(
    image_array,
    axis=0
)

print("Input tensor shape:", image_array.shape)


# ============================================================
# 14. MODEL PREDICTION
# ============================================================

prediction = model.predict(image_array)

print("Raw model output:", prediction)

predicted_class_index = np.argmax(prediction)

predicted_class = classes[predicted_class_index]

confidence = prediction[0][predicted_class_index] * 100


print("\nImage:", image_path)
print("Prediction:", predicted_class)
print("Confidence:", confidence, "%")
