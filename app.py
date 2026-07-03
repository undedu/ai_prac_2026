from pathlib import Path

import torch
from flask import Flask, render_template, request, send_file

from config import DEVICE
from training.model_factory import create_model
from training.dataset_loader import get_transforms, load_class_names
from utils.report import export_to_excel
from utils.predictor import predict
from utils.database import (
    init_db,
    save_history,
    load_history
)

# ------------------------------------
# Flask
# ------------------------------------

app = Flask(__name__)

UPLOAD_FOLDER = Path("static/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ------------------------------------
# БД
# ------------------------------------

init_db()

# ------------------------------------
# Загружаем модель
# ------------------------------------

MODEL_NAME = "resnet50"

model = create_model(
    MODEL_NAME,
    101
)

model.load_state_dict(
    torch.load(
        "models/resnet50_best.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

transform = get_transforms()

class_names = load_class_names()

# ------------------------------------
# Routes
# ------------------------------------


@app.route("/", methods=["GET", "POST"])
def index():

    image_name = None
    predictions = None

    if request.method == "POST":

        file = request.files.get("image")

        if file:

            image_name = file.filename

            save_path = UPLOAD_FOLDER / image_name

            file.save(save_path)

            predictions = predict(
                image_path=save_path,
                model=model,
                transform=transform,
                class_names=class_names
            )

            save_history(
                filename=image_name,
                predictions=predictions
            )

    return render_template(
        "index.html",
        image=image_name,
        predictions=predictions,
        model_name="ResNet50"
    )


@app.route("/history")
def history():

    history_data = load_history()

    return render_template(
        "history.html",
        history=history_data
    )

@app.route("/download_report")
def download_report():

    file_path = export_to_excel()

    return send_file(
        file_path,
        as_attachment=True,
        download_name="classification_report.xlsx"
    )

# ------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )