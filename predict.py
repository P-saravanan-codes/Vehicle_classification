import os
import sys
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 128

MODEL_PATH = "vehicle_classifier.keras"

CLASSES = [
    "Cars",
    "Bikes",
    "Auto_Rickshaws",
    "Motorcycles"
]


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("==========================================")
print("LOADING MODEL")
print("==========================================")

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model file not found!")

    print(
        "Expected:",
        MODEL_PATH
    )

    sys.exit(1)


model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    print("\n==========================================")
    print("IMAGE PREDICTION")
    print("==========================================")

    print("Image:", image_path)


    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        print(
            "ERROR: Image not found!"
        )

        return


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = tf.keras.utils.load_img(

        image_path,

        target_size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )


    # --------------------------------------------------------
    # Convert image to array
    # --------------------------------------------------------

    image_array = tf.keras.utils.img_to_array(
        image
    )


    print(
        "Image array shape:",
        image_array.shape
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image_array = image_array / 255.0


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(

        image_array,

        axis=0
    )


    print(
        "Model input shape:",
        image_array.shape
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(

        image_array,

        verbose=0
    )


    probabilities = prediction[0]


    print("\nRaw model output:")

    print(probabilities)


    # --------------------------------------------------------
    # Find highest probability
    # --------------------------------------------------------

    predicted_value = int(
        np.argmax(probabilities)
    )


    predicted_class = CLASSES[
        predicted_value
    ]


    confidence = (
        probabilities[predicted_value]
        * 100
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n==========================================")
    print("FINAL RESULT")
    print("==========================================")

    print(
        "Class Name :",
        predicted_class
    )

    print(
        "Class Value:",
        predicted_value
    )

    print(
        "Confidence :",
        f"{confidence:.2f}%"
    )


    # ========================================================
    # ALL PROBABILITIES
    # ========================================================

    print("\n==========================================")
    print("ALL CLASS PROBABILITIES")
    print("==========================================")

    for i, class_name in enumerate(CLASSES):

        percentage = (
            probabilities[i]
            * 100
        )

        print(
            f"{class_name:20}"
            f" Value={i}"
            f" Probability={percentage:.2f}%"
        )


# ============================================================
# GET IMAGE FROM COMMAND LINE
# ============================================================

if len(sys.argv) < 2:

    print("\nUsage:")

    print(
        "python3 predict.py <image_path>"
    )

    print("\nExample:")

    print(
        "python3 predict.py test_images/car.jpg"
    )

    sys.exit(1)


image_path = sys.argv[1]


# ============================================================
# RUN PREDICTION
# ============================================================

predict_image(image_path)