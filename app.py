
import os
import cv2
from flask import Flask, request, render_template_string
from ultralytics import YOLO
from collections import Counter
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load trained AI model
model = YOLO("best.onnx", task="detect")

# Create folders
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Circuit Detective</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #101827;
            color: white;
            text-align: center;
            padding: 40px;
        }
        .container {
            max-width: 800px;
            margin: auto;
            background: #1e293b;
            padding: 30px;
            border-radius: 15px;
        }
        h1 { color: #38bdf8; }
        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 14px 30px;
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
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 Circuit Detective</h1>
    <h3>AI-Powered PCB Defect Inspection</h3>

    <p>Upload a PCB image to inspect potential defects.</p>

    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="pcb_image"
               accept="image/*" required>
        <br><br>
        <button type="submit">Inspect PCB</button>
    </form>

    {% if result_image %}
        <h2>Inspection Result</h2>
        <img src="{{ result_image }}">

        <div class="report">
            <h3>Defect Inspection Report</h3>
            <pre>{{ report }}</pre>
        </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    result_image = None
    report = ""

    if request.method == "POST":

        file = request.files.get("pcb_image")

        if not file or not file.filename:
            return "Please upload a PCB image.", 400

        filename = secure_filename(file.filename)

        if not filename:
            return "Invalid filename.", 400

        image_path = os.path.join(
            "static/uploads", filename
        )

        file.save(image_path)

        # Run AI inspection
        results = model.predict(
            source=image_path,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        # Save highlighted image
        output_path = os.path.join(
            "static/results", "result.jpg"
        )

        cv2.imwrite(output_path, result.plot())

        # Generate defect report
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = model.names[class_id]

            detections.append((name, confidence))

        if detections:
            counts = Counter(
                name for name, confidence in detections
            )

            report = f"Total detections: {len(detections)}\\n\\n"

            for name, count in counts.items():
                report += f"{name}: {count}\\n"

            report += "\\nDetection details:\\n"

            for name, confidence in detections:
                report += (
                    f"{name}: {confidence:.1%} confidence\\n"
                )
        else:
            report = (
                "No defects detected at the current "
                "confidence threshold.\\n"
                "This does not guarantee the PCB is defect-free."
            )

        result_image = "/static/results/result.jpg"

    return render_template_string(
        HTML,
        result_image=result_image,
        report=report
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
