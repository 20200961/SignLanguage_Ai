import tensorflow as tf

loaded_model = tf.keras.models.load_model("models/final.h5")


converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

with open("last.tflite", "wb") as f:
    f.write(tflite_model)

print("변환 완료")
