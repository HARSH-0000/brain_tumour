from flask import Flask, jsonify, render_template, request
import os

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from pathlib import Path
import numpy as np

app = Flask(__name__)

HERE = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    HERE / "model.keras",
    HERE / "model.h5",
    HERE / "models" / "model.h5",
]

model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
if model_path is None:
    raise FileNotFoundError(
        "Could not find model.h5. Looked in: "
        + ", ".join(str(p) for p in MODEL_CANDIDATES)
    )

MODEL_LOAD_ERROR = None
model = None
try:
    model = load_model(str(model_path), compile=False)
except Exception as e:
    # Keep the server running so the UI can show a useful error message.
    MODEL_LOAD_ERROR = str(e)

# IMPORTANT: This order must match the model's training label order.
# The frontend expects these exact keys.
CLASS_LABELS = ["pituitary", "glioma", "meningioma", "notumor"]

LABEL_UI = {
    "notumor": {
        "label": "No Tumor Detected",
        "description": "No tumor patterns detected in the uploaded scan. If symptoms persist, consult a radiologist for confirmation.",
    },
    "glioma": {
        "label": "Glioma",
        "description": "Model indicates features consistent with glioma. Please consult a specialist; further imaging and clinical correlation are recommended.",
    },
    "meningioma": {
        "label": "Meningioma",
        "description": "Model indicates features consistent with meningioma. A radiologist's review is recommended for confirmation and treatment planning.",
    },
    "pituitary": {
        "label": "Pituitary Tumor",
        "description": "Model indicates features consistent with a pituitary tumor. Consider endocrinology and radiology evaluation for next steps.",
    },
}

upload_folder = str(HERE / "uploads")
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

def predict(image_path: str):
    if model is None:
        raise RuntimeError(
            "Model failed to load. "
            "If you trained with a different Keras/TensorFlow version, "
            "re-export the model in the current environment (preferred: .keras format). "
            f"Loader error: {MODEL_LOAD_ERROR}"
        )
    img_pil = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img_pil)
    img_normalized = img_array / 255.0
    img_for_prediction = np.expand_dims(img_normalized, axis=0)

    predictions = model.predict(img_for_prediction, verbose=0)
    probs = predictions[0].astype(float)

    if probs.shape[0] != len(CLASS_LABELS):
        raise ValueError(
            f"Model output has {probs.shape[0]} classes, expected {len(CLASS_LABELS)}."
        )

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_LABELS[predicted_idx]
    confidence_pct = float(probs[predicted_idx] * 100.0)

    all_scores = {cls: float(p * 100.0) for cls, p in zip(CLASS_LABELS, probs)}
    ui = LABEL_UI.get(predicted_class, {"label": predicted_class, "description": ""})

    return {
        "predicted_class": predicted_class,
        "confidence": confidence_pct,
        "label": ui["label"],
        "description": ui["description"],
        "all_scores": all_scores,
    }

#routes
@app.route("/",methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files.get("image") or request.files.get("file")
    if not file:
        return jsonify({"error": "Missing file. Upload with form field 'image' (or 'file')."}), 400

    filename = os.path.join(upload_folder, file.filename)
    file.save(filename)

    try:
        result = predict(filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(filename)
        except OSError:
            pass

if __name__=="__main__":
    app.run(debug=True)
