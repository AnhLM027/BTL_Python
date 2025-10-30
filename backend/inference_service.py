import os, torch
from GenConViT.model.config import load_config
from GenConViT.model.pred_func import set_result, load_genconvit, is_video, is_video_folder, df_face, df_face_from_folder, pred_vid, store_result, real_or_fake

CONFIG = load_config()
MODEL = None

def load_model_once():
    global MODEL
    if MODEL is None:
        print("🔹 Loading GenConViT model...")
        ed_weight = "genconvit_ed_inference"
        vae_weight = "genconvit_vae_inference"
        MODEL = load_genconvit(CONFIG, "genconvit", ed_weight, vae_weight, fp16=False)
        print("✅ Model loaded successfully.")
    return MODEL

async def run_inference_path(video_path: str, num_frames: int = 15):
    try:
        model = load_model_once()
        result = set_result()

        if not os.path.exists(video_path):
            return {"error": f"File not found: {video_path}"}

        print(f"\nRunning inference on: {video_path}")
        is_vid_folder = is_video_folder(video_path)
        if not (is_video(video_path) or is_vid_folder):
            return {"error": f"Invalid video format: {video_path}"}

        df = df_face(video_path, num_frames)

        y, y_val = pred_vid(df, model)
        label = real_or_fake(y)
        conf = float(y_val)

        result = store_result(result, os.path.basename(video_path), y, y_val, "uncategorized", "unknown", None)
        
        print(f"Prediction done: {label} ({conf:.4f})")
        return {
            "filename": os.path.basename(video_path),
            "label": label,
            "confidence": conf,
        }

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return {"error": str(e)}
