
import os
import uuid
import cv2
import numpy as np
import onnxruntime as ort

from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename

# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.onnx")

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

CLASS_NAMES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper"
]

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# --------------------------------------------------
# 2. LOAD ONNX MODEL
# --------------------------------------------------

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Circuit Detective AI loaded successfully!")


# --------------------------------------------------
# 3. IMAGE PREPROCESSING
# --------------------------------------------------

def letterbox(image):

    h, w = image.shape[:2]

    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    pad_w = INPUT_SIZE - new_w
    pad_h = INPUT_SIZE - new_h

    left = pad_w // 2
    top = pad_h // 2

    canvas = np.full(
        (INPUT_SIZE, INPUT_SIZE, 3),
        114,
        dtype=np.uint8
    )

    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas, scale, left, top


# --------------------------------------------------
# 4. CLASS-AWARE NON-MAXIMUM SUPPRESSION
# --------------------------------------------------

def class_aware_nms(boxes, scores, class_ids):

    selected_indices = []

    for class_id in range(len(CLASS_NAMES)):

        class_indices = [
            i for i in range(len(class_ids))
            if class_ids[i] == class_id
        ]

        if not class_indices:
            continue

        class_boxes = [boxes[i] for i in class_indices]
        class_scores = [scores[i] for i in class_indices]

        keep = cv2.dnn.NMSBoxes(
            class_boxes,
            class_scores,
            CONF_THRESHOLD,
            IOU_THRESHOLD
        )

        if len(keep) > 0:

            for index in np.array(keep).flatten():
                selected_indices.append(class_indices[index])

    return selected_indices


# --------------------------------------------------
# 5. PCB DEFECT DETECTION
# --------------------------------------------------

def inspect_image(image):

    original_h, original_w = image.shape[:2]

    padded, scale, left, top = letterbox(image)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    # Normalize pixel values
    tensor = rgb.astype(np.float32) / 255.0

    # Convert HWC to CHW
    tensor = np.transpose(tensor, (2, 0, 1))

    # Add batch dimension
    tensor = np.expand_dims(tensor, axis=0)

    # Run ONNX inference
    outputs = session.run(
        [output_name],
        {input_name: tensor}
    )

    predictions = np.squeeze(outputs[0])

    # Validate model output
    if predictions.ndim != 2:
        raise ValueError(
            f"Unexpected ONNX output shape: {predictions.shape}"
        )

    expected_values = 4 + len(CLASS_NAMES)

    if predictions.shape[0] == expected_values:
        predictions = predictions.T

    elif predictions.shape[1] != expected_values:
        raise ValueError(
            f"Unexpected model output shape: {predictions.shape}"
        )

    boxes = []
    scores = []
    class_ids = []

    # Process predictions
    for prediction in predictions:

        class_scores = prediction[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < CONF_THRESHOLD:
            continue

        cx, cy, width, height = prediction[:4]

        # Convert YOLO center coordinates to corner coordinates
        x1 = (cx - width / 2 - left) / scale
        y1 = (cy - height / 2 - top) / scale
        x2 = (cx + width / 2 - left) / scale
        y2 = (cy + height / 2 - top) / scale

        # Clip coordinates to original image dimensions
        x1 = max(0, min(original_w - 1, int(x1)))
        y1 = max(0, min(original_h - 1, int(y1)))
        x2 = max(0, min(original_w - 1, int(x2)))
        y2 = max(0, min(original_h - 1, int(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

        scores.append(confidence)
        class_ids.append(class_id)

    # Apply class-aware NMS
    selected_indices = class_aware_nms(
        boxes,
        scores,
        class_ids
    )

    annotated = image.copy()
    detections = []

    # Draw detection results
    for index in selected_indices:

        x, y, w, h = boxes[index]

        confidence = scores[index]
        class_id = class_ids[index]

        name = CLASS_NAMES[class_id]

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (34, 197, 94),
            2
        )

        label = f"{name}: {confidence:.1%}"

        cv2.putText(
            annotated,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (34, 197, 94),
            2
        )

        detections.append({
            "name": name,
            "confidence": confidence
        })

    return annotated, detections


# --------------------------------------------------
# 6. WEBSITE INTERFACE
# --------------------------------------------------

HTML = """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Circuit Detective | AI PCB Inspection</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 30px 15px;
    background: #0f172a;
    color: #f8fafc;
    font-family: Arial, sans-serif;
}

.container {
    max-width: 900px;
    margin: auto;
    background: #1e293b;
    padding: 35px;
    border-radius: 18px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.3);
}

h1 {
    color: #38bdf8;
    font-size: 34px;
    margin-bottom: 8px;
}

.subtitle {
    color: #cbd5e1;
    margin-bottom: 25px;
}

.upload-box {
    background: #334155;
    padding: 25px;
    border-radius: 12px;
    margin: 25px 0;
}

input[type="file"] {
    max-width: 100%;
    margin: 15px;
    color: white;
}

button {
    background: #2563eb;
    color: white;
    border: none;
    padding: 14px 28px;
    border-radius: 9px;
    cursor: pointer;
    font-size: 16px;
    font-weight: bold;
}

button:hover {
    background: #1d4ed8;
}

.result-section {
    margin-top: 35px;
}

.result-image {
    width: 100%;
    max-width: 750px;
    border-radius: 12px;
    margin-top: 15px;
    border: 2px solid #475569;
}

.report {
    background: #334155;
    padding: 22px;
    margin-top: 25px;
    border-radius: 12px;
    text-align: left;
}

.report h3 {
    color: #38bdf8;
    margin-top: 0;
}

.detection {
    padding: 12px;
    background: #1e293b;
    border-radius: 8px;
    margin: 10px 0;
}

.confidence {
    color: #4ade80;
    font-weight: bold;
}

.error {
    background: #7f1d1d;
    color: #fecaca;
    padding: 12px;
    border-radius: 8px;
    margin-top: 20px;
}

.footer {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 30px;
}

</style>

</head>

<body>

<div class="container">

    <h1>🔬 Circuit Detective</h1>

    <p class="subtitle">
        AI-Powered PCB Defect Inspection System
    </p>

    <p>
        Upload a PCB image and let our AI model
        identify potential manufacturing defects.
    </p>

    <div class="upload-box">

        <form method="POST" enctype="multipart/form-data">

            <input
                type="file"
                name="pcb_image"
                accept=".jpg,.jpeg,.png,.bmp"
                required
            >

            <br>

            <button type="submit">
                🔍 Inspect PCB
            </button>

        </form>

    </div>

    {% if result_image %}

    <div class="result-section">

        <h2>Inspection Results</h2>

        <img
            src="{{ result_image }}"
            class="result-image"
            alt="Annotated PCB inspection result"
        >

        <div class="report">

            <h3>📋 Defect Report</h3>

            {% if detections %}

                <p>
                    Total detections:
                    <strong>{{ detections|length }}</strong>
                </p>

                {% for item in detections %}

                <div class="detection">

                    <strong>{{ item.name.replace("_", " ")|title }}</strong>

                    <br>

                    <span class="confidence">
                        Confidence: {{ "%.1f"|format(item.confidence * 100) }}%
                    </span>

                </div>

                {% endfor %}

            {% else %}

                <p>
                    No defects detected at the current confidence threshold.
                </p>

                <p>
                    This does not guarantee that the PCB is defect-free.
                </p>

            {% endif %}

        </div>

    </div>

    {% endif %}

    {% if error %}

        <div class="error">
            {{ error }}
        </div>

    {% endif %}

    <div class="footer">
        Circuit Detective | AI-Based PCB Defect Detection
    </div>

</div>

</body>
</html>
"""


# --------------------------------------------------
# 7. MAIN WEBSITE ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result_image = None
    detections = []
    error = ""

    if request.method == "POST":

        file = request.files.get("pcb_image")

        if not file or not file.filename:

            error = "Please select a PCB image."

        else:

            filename = secure_filename(file.filename)

            extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

            if extension not in ALLOWED_EXTENSIONS:

                error = "Please upload a JPG, JPEG, PNG, or BMP image."

            else:

                try:

                    image_bytes = file.read(MAX_IMAGE_SIZE + 1)

                    if len(image_bytes) > MAX_IMAGE_SIZE:
                        raise ValueError("Image must be smaller than 10 MB.")

                    if not image_bytes:
                        raise ValueError("The uploaded file is empty.")

                    image_data = np.frombuffer(
                        image_bytes,
                        dtype=np.uint8
                    )

                    image = cv2.imdecode(
                        image_data,
                        cv2.IMREAD_COLOR
                    )

                    if image is None:
                        raise ValueError("Invalid image file.")

                    annotated, detections = inspect_image(image)

                    result_name = str(uuid.uuid4()) + ".jpg"

                    output_path = os.path.join(
                        "static",
                        "results",
                        result_name
                    )

                    success = cv2.imwrite(
                        output_path,
                        annotated
                    )

                    if not success:
                        raise ValueError("Could not save inspection result.")

                    result_image = "/" + output_path

                except Exception as e:

                    error = f"Inspection failed: {str(e)}"

    return render_template_string(
        HTML,
        result_image=result_image,
        detections=detections,
        error=error
    )


# --------------------------------------------------
# 8. HEALTH CHECK ROUTE
# --------------------------------------------------

@app.route("/health")
def health():

    return {
        "status": "healthy",
        "application": "Circuit Detective"
    }


# --------------------------------------------------
# 9. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )

