import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMAGE_SIZE = 128
NUM_CLASSES = 4


# ============================================================
# CREATE CNN MODEL
# ============================================================

def create_model():

    model = models.Sequential([

        # Input
        layers.Input(
            shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
        ),

        # CNN Layer 1
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # CNN Layer 2
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # CNN Layer 3
        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # Convert feature maps into one-dimensional vector
        layers.Flatten(),

        # Fully Connected Layer
        layers.Dense(
            128,
            activation="relu"
        ),

        # Output Layer
        # 4 classes:
        # 0 = Cars
        # 1 = Bikes
        # 2 = Auto_Rickshaws
        # 3 = Motorcycles
        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    return model