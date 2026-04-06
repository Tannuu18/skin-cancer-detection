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

# ---- Build Model ----
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)

membership = Dense(7, activation='softmax', name='membership')(x)
non_membership = Dense(7, activation='sigmoid', name='non_membership')(x)

model = Model(inputs=base_model.input, outputs=[membership, non_membership])

# ---- Load Weights ----
model.load_weights("model_weights.weights.h5")

labels = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

model.compile(
    optimizer='adam',
    loss={
        'membership': sakaguchi_loss,
        'non_membership': 'binary_crossentropy'
    }
)

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
        # Read image
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        membership_pred, _ = model.predict(img)

        pred_class = int(np.argmax(membership_pred))
        confidence = float(np.max(membership_pred))

        return jsonify({
            "prediction": labels[pred_class],
            "confidence": confidence
        })

    except Exception as e:
        print("Prediction error:\n" + traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)