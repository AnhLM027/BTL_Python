from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, uuid, shutil

from backend.inference_service import run_inference_path

app = FastAPI(title="GenConViT Deepfake Detector", root_path="/deepfake")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        filename = f"{uuid.uuid4()}_{file.filename}"
        save_path = os.path.join(UPLOAD_DIR, filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"✅ Uploaded: {save_path}")
        return {"filename": filename}
    except Exception as e:
        print("❌ Upload error:", e)
        return JSONResponse({"error": str(e)})

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    filename = data.get("filename")
    if not filename:
        return {"error": "No filename provided"}

    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": f"File not found: {filename}"}

    result = await run_inference_path(file_path)
    return result

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
