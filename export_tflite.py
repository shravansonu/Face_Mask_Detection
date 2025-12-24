import tensorflow as tf

model = tf.keras.models.load_model("model/face_mask_detector.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model/face_mask_detector.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved")
