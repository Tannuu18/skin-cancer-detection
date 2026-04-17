from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
from PIL import Image
from flask_cors import CORS
from pathlib import Path
import traceback

app = Flask(__name__)
CORS(app)  

BASE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (128, 128)

# ---- Sakaguchi Loss ----
sakaguchi_matrix = np.array([
    [1.0, 0.2, 0.6, 0.3, 0.2, 0.1, 0.1],
    [0.2, 1.0, 0.3, 0.5, 0.6, 0.1, 0.1],
    [0.6, 0.3, 1.0, 0.4, 0.3, 0.1, 0.1],
    [0.3, 0.5, 0.4, 1.0, 0.6, 0.1, 0.1],
    [0.2, 0.6, 0.3, 0.6, 1.0, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.2],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 1.0]
])

sakaguchi_matrix_tf = tf.constant(sakaguchi_matrix, dtype=tf.float32)

def sakaguchi_loss(y_true, y_pred):
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    similarity = tf.matmul(y_true, sakaguchi_matrix_tf)
    penalty = tf.reduce_mean(tf.square(y_pred - similarity))
    return ce + 0.2 * penalty

# ---- Build Lesion Model ----
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


class DenseCompat(Dense):
    @classmethod
    def from_config(cls, config):
        # Keras 3 may fail on older H5 configs that include this key.
        config.pop("quantization_config", None)
        return super().from_config(config)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)

membership = Dense(7, activation='softmax', name='membership')(x)
non_membership = Dense(7, activation='sigmoid', name='non_membership')(x)

model = Model(inputs=base_model.input, outputs=[membership, non_membership])

# ---- Load Models Once at Startup ----
model.load_weights("model_weights.weights.h5")
melanoma_stage_model = tf.keras.models.load_model(
    "melanoma_stage_model.h5",
    compile=False,
    custom_objects={"Dense": DenseCompat}
)

labels = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
stage_labels = ["Low", "Intermediate", "High"]

def preprocess_image(file_stream):
    img = Image.open(file_stream).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def predict_lesion_and_stage(image_batch):
    membership_pred, _ = model(image_batch, training=False)
    membership_pred = membership_pred.numpy()[0]

    lesion_idx = int(np.argmax(membership_pred))
    lesion_type = labels[lesion_idx]
    lesion_confidence = float(np.max(membership_pred))

    if lesion_type != "mel":
        return {
            "type": lesion_type,
            "confidence": lesion_confidence
        }

    stage_pred = melanoma_stage_model(image_batch, training=False)
    if isinstance(stage_pred, (list, tuple)):
        stage_pred = stage_pred[0]
    stage_pred = stage_pred.numpy()[0]

    stage_idx = int(np.argmax(stage_pred))
    return {
        "type": "mel",
        "confidence": lesion_confidence,
        "stage": stage_labels[stage_idx],
        "stage_confidence": float(np.max(stage_pred))
    }

# ---- Routes ----
@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Empty file"}), 400

    try:
        image_batch = preprocess_image(file.stream)
        response = predict_lesion_and_stage(image_batch)
        return jsonify(response)

    except Exception as e:
        print("Prediction error:\n" + traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)