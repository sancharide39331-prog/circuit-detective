
import os
import uuid
import cv2
import numpy as np
import onnxruntime as ort

from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = "best.onnx"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

CLASS_NAMES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper"
]

# Load ONNX model
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)


def letterbox(image):
    h, w = image.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h))

    pad_w = INPUT_SIZE - new_w
    pad_h = INPUT_SIZE - new_h

    left = pad_w // 2
    top = pad_h // 2

    canvas = np.full(
        (INPUT_SIZE, INPUT_SIZE, 3),
        114,
        dtype=np.uint8
    )

    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas, scale, left, top


def inspect_image(image):

    original_h, original_w = image.shape[:2]

    padded, scale, left, top = letterbox(image)

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    outputs = session.run(
        [output_name],
        {input_name: tensor}
    )

    predictions = outputs[0]

    predictions = np.squeeze(predictions)

    # Convert output to [number of detections, 10]
    if predictions.ndim != 2:
        raise ValueError("Unexpected ONNX output shape")

    if predictions.shape[0] == 4 + len(CLASS_NAMES):
        predictions = predictions.T
    elif predictions.shape[1] != 4 + len(CLASS_NAMES):
        raise ValueError(
            f"Unexpected model output shape: {predictions.shape}"
        )

    boxes = []
    scores = []
    class_ids = []

    for prediction in predictions:

        class_scores = prediction[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < CONF_THRESHOLD:
            continue

        cx, cy, width, height = prediction[:4]

        x1 = (cx - width / 2 - left) / scale
        y1 = (cy - height / 2 - top) / scale
        x2 = (cx + width / 2 - left) / scale
        y2 = (cy + height / 2 - top) / scale

        x1 = max(0, min(original_w - 1, int(x1)))
        y1 = max(0, min(original_h - 1, int(y1)))
        x2 = max(0, min(original_w - 1, int(x2)))
        y2 = max(0, min(original_h - 1, int(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        IOU_THRESHOLD
    )

    annotated = image.copy()
    detections = []

    if len(indices) > 0:

        for index in np.array(indices).flatten():

            x, y, w, h = boxes[index]
            confidence = scores[index]
            class_id = class_ids[index]

            name = CLASS_NAMES[class_id]

            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            label = f"{name}: {confidence:.0%}"

            cv2.putText(
                annotated,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            detections.append({
                "name": name,
                "confidence": confidence
            })

    return annotated, detections


HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Circuit Detective</title>
<style>
body {
    background: #101827;
    color: white;
    font-family: Arial;
    text-align: center;
    padding: 35px;
}
.container {
    max-width: 850px;
    margin: auto;
    background: #1e293b;
    padding: 30px;
    border-radius: 16px;
}
h1 { color: #38bdf8; }
button {
    background: #2563eb;
    color: white;
    border: none;
    padding: 13px 25px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
}
img {
    max-width: 100%;
    margin-top: 20px;
    border-radius: 10px;
}
.report {
    background: #334155;
    padding: 20px;
    margin-top: 20px;
    border-radius: 10px;
    text-align: left;
}
input { margin: 15px; }
</style>
</head>
<body>
<div class="container">
<h1>🔬 Circuit Detective</h1>
<h3>AI-Powered PCB Defect Inspection</h3>
<p>Upload a PCB image and inspect potential defects.</p>

<form method="POST" enctype="multipart/form-data">
<input type="file" name="pcb_image"
accept="image/*" required>
<br>
<button type="submit">🔍 Inspect PCB</button>
</form>

{% if result_image %}
<h2>Inspection Result</h2>
<img src="{{ result_image }}">
<div class="report">
<h3>Defect Report</h3>
<pre>{{ report }}</pre>
</div>
{% endif %}

{% if error %}
<p style="color:#f87171">{{ error }}</p>
{% endif %}

</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result_image = None
    report = ""
    error = ""

    if request.method == "POST":

        file = request.files.get("pcb_image")

        if not file or not file.filename:
            error = "Please select a PCB image."

        else:
            filename = secure_filename(file.filename)

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):
                error = "Please upload a JPG or PNG image."

            else:
                try:
                    image_bytes = np.frombuffer(
                        file.read(),
                        dtype=np.uint8
                    )

                    image = cv2.imdecode(
                        image_bytes,
                        cv2.IMREAD_COLOR
                    )

                    if image is None:
                        raise ValueError("Invalid image file.")

                    annotated, detections = inspect_image(image)

                    result_name = str(uuid.uuid4()) + ".jpg"
                    output_path = os.path.join(
                        "static/results",
                        result_name
                    )

                    cv2.imwrite(output_path, annotated)

                    result_image = "/" + output_path

                    if detections:
                        report = (
                            f"Total defects detected: "
                            f"{len(detections)}\\n\\n"
                        )

                        for item in detections:
                            report += (
                                f"{item['name']} - "
                                f"{item['confidence']:.1%} confidence\\n"
                            )
                    else:
                        report = (
                            "No defects detected at the current "
                            "confidence threshold.\\n"
                            "This does not guarantee the PCB is defect-free."
                        )

                except Exception as e:
                    error = f"Inspection failed: {str(e)}"

    return render_template_string(
        HTML,
        result_image=result_image,
        report=report,
        error=error
    )


@app.route("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
