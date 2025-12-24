import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

data = np.load("data/dataset.npz")

X_train = data["X_train"]
X_val = data["X_val"]
yb_train = data["yb_train"]
yb_val = data["yb_val"]
yl_train = data["yl_train"]
yl_val = data["yl_val"]

base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)

bbox_output = Dense(4, activation="sigmoid", name="bbox")(x)
class_output = Dense(3, activation="softmax", name="class")(x)

model = Model(inputs=base.input, outputs=[bbox_output, class_output])

model.compile(
    optimizer="adam",
    loss={
        "bbox": "mse",
        "class": "categorical_crossentropy"
    },
    metrics={
        "class": "accuracy"
    }
)

model.fit(
    X_train,
    {"bbox": yb_train, "class": yl_train},
    validation_data=(X_val, {"bbox": yb_val, "class": yl_val}),
    epochs=10,
    batch_size=16
)

model.save("model/face_mask_detector.keras")
print("Model training complete")
