from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import os
import uuid
import cv2
import imageio

app = FastAPI()

model = YOLO("best.pt")

UPLOAD_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def process_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    best_conf: dict[str, float] = {}

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        result = results[0]
        plotted = result.plot()
        writer.append_data(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            if label not in best_conf or conf > best_conf[label]:
                best_conf[label] = conf

    cap.release()
    writer.close()

    return [
        {"label": label, "confidence": round(conf * 100, 2)}
        for label, conf in best_conf.items()
    ]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"result_image": None, "result_video": None, "detections": [], "file_type": None}
    )


@app.post("/detect", response_class=HTMLResponse)
async def detect(request: Request, file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{file.filename}")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    if ext in VIDEO_EXTENSIONS:
        output_path = os.path.join(UPLOAD_FOLDER, f"result_{file_id}.mp4")
        detections = process_video(input_path, output_path)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "result_image": None,
                "result_video": "/" + output_path.replace("\\", "/"),
                "detections": detections,
                "file_type": "video"
            }
        )

    results = model(input_path)
    result = results[0]

    plotted = result.plot()
    output_path = os.path.join(UPLOAD_FOLDER, f"result_{file_id}.jpg")
    cv2.imwrite(output_path, plotted)

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        detections.append({"label": label, "confidence": round(conf * 100, 2)})

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "result_image": "/" + output_path.replace("\\", "/"),
            "result_video": None,
            "detections": detections,
            "file_type": "image"
        }
    )
