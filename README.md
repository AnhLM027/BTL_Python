# 🧠 GenConViT - Deepfake Video Detection System

GenConViT (**Generative Convolutional Vision Transformer**) là một mô hình học sâu được thiết kế cho bài toán **phát hiện video Deepfake**, kết hợp giữa mạng sinh (VAE/Autoencoder) và Vision Transformer để khai thác đồng thời **đặc trưng không gian** và **biểu diễn tiềm ẩn thời gian** trong video.

Hệ thống bao gồm pipeline xử lý video, trích xuất đặc trưng, phân loại và giao diện web demo được xây dựng bằng **FastAPI + HTML/JS**.

---

## 📚 Mục lục

* [1️⃣ Quy trình phát hiện Deepfake](#1️⃣-quy-trình-phát-hiện-deepfake)
* [2️⃣ Thiết kế & triển khai mô hình GenConViT](#2️⃣-thiết-kế--triển-khai-mô-hình-genconvit)
* [3️⃣ Kiến trúc hệ thống](#3️⃣-kiến-trúc-hệ-thống)
* [4️⃣ Cấu trúc thư mục](#4️⃣-cấu-trúc-thư-mục)
* [5️⃣ Hướng dẫn chạy demo web](#5️⃣-hướng-dẫn-chạy-demo-web)
* [6️⃣ Đánh giá mô hình](#6️⃣-đánh-giá-mô-hình)
* [7️⃣ Ghi chú kỹ thuật](#7️⃣-ghi-chú-kỹ-thuật)

---

## 1️⃣ Quy trình phát hiện Deepfake

Hệ thống phát hiện video Deepfake điển hình, dựa trên **GenConViT**, gồm 4 khối chức năng chính:

### 🧬 1. Tiền xử lý video

* Trích xuất khung hình từ video (.mp4)
* Phát hiện & căn chỉnh khuôn mặt (RetinaFace/MTCNN)
* Chuẩn hóa kích thước, cân bằng ánh sáng, khử nhiễu

### ⚙️ 2. Trích xuất đặc trưng

* Sử dụng **GenConViT** để mã hóa đặc trưng **không gian–thời gian**
* Kết hợp **VAE/Autoencoder + Transformer Encoder** để học biểu diễn tiềm ẩn (latent embedding)

### 🧠 3. Phân loại Deepfake

* Transformer head hoặc Linear classifier phân biệt video thật/giả
* Loss kết hợp: Classification loss + Reconstruction loss → cải thiện khả năng nhận biết sai lệch sinh học

### 📊 4. Đánh giá & hiển thị

* Chỉ số: **Accuracy, Precision, Recall, F1-score, AUC**
* Kết quả cuối: Xác suất video là Deepfake

---

## 2️⃣ Thiết kế & triển khai mô hình GenConViT

### 🎯 Mục tiêu

* Xây dựng pipeline chuẩn cho bài toán **phân loại video Deepfake**
* Tối ưu đặc trưng không gian–thời gian với **CNN + Transformer**

### 🔧 Các bước triển khai

1. Trích khung hình, phát hiện khuôn mặt, resize 224x224 px
2. Thiết kế mô hình:

   * Encoder CNN (ResNet/ConvNeXt)
   * Generative branch (VAE)
   * Transformer encoder cho quan hệ thời gian
3. Huấn luyện trên dataset: **DFDC, FaceForensics++, Celeb-DF v2**
4. Đánh giá bằng Accuracy, Precision, Recall, F1, AUC

### 🖥️ Yêu cầu hệ thống

**Chức năng:**

* Upload video .mp4
* Tách khung hình, phát hiện khuôn mặt, resize
* Huấn luyện & infer với GenConViT
* Hiển thị xác suất Deepfake

**Phi chức năng:**

* GPU (CUDA)
* Mở rộng cho nhiều dataset
* Giao diện web (FastAPI)
* Cấu trúc module dễ bảo trì

---

## 3️⃣ Kiến trúc hệ thống

```
Input Video (.mp4)
   ↓
Frame Extraction → Face Detection (RetinaFace)
   ↓
GenConViT Model → Feature Extraction
   ↓
Classification Head → Prob(Deepfake)
   ↓
FastAPI Web UI → Kết quả hiển thị
```

---

## 4️⃣ Cấu trúc thư mục

```
Video_Deepfake_Detection/
├── backend
│   ├── inference_service.py
│   ├── __init__.py
│   ├── main.py
│   └── uploads
├── frontend
│   ├── favicon.svg
│   ├── index.html
│   ├── script.js
│   └── style.css
├── GenConViT
│   ├── dataset
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── __init__.py
│   ├── model
│   │   ├── config.py
│   │   ├── config.yaml
│   │   ├── genconvit_ed.py
│   │   ├── genconvit.py
│   │   ├── genconvit_vae.py
│   │   ├── __init__.py
│   │   ├── model_embedder.py
│   │   └── pred_func.py
│   ├── prediction.py
│   ├── requirements.txt
│   ├── result
│   ├── train
│   │   ├── __init__.py
│   │   ├── train_ed.py
│   │   └── train_vae.py
│   └── train.py
└── scripts.sh
```

---

## 5️⃣ Hướng dẫn chạy demo web

### Cài đặt môi trường

```bash
git clone https://github.com/AnhLM027/BTL_Python.git Video_Deepfake_Detection
cd Video_Deepfake_Detection
pip install -r requirements.txt
```

### Chạy backend (FastAPI)

```bash
# Tạo terminal ngay tại thư mục Video_Deepfake_Detection
uvicorn backend.main:app --host 0.0.0.0 --port 9000
```

### Truy cập giao diện web

* Local: [http://127.0.0.1:9000/deepfake](http://127.0.0.1:9000/deepfake)
* Web PTIT: [https://aispeech.ptit.edu.vn/deepfake/](https://aispeech.ptit.edu.vn/deepfake/)

---

## 6️⃣ Đánh giá mô hình

| Chỉ số    | Ý nghĩa                              |
| --------- | ------------------------------------ |
| Accuracy  | Độ chính xác toàn bộ video           |
| Precision | Tỷ lệ phát hiện đúng trong nhóm FAKE |
| Recall    | Tỷ lệ phát hiện được FAKE            |
| F1-score  | Trung bình Precision & Recall        |
| AUC       | Diện tích dưới đường cong ROC        |

> Mô hình GenConViT đạt Accuracy trung bình ~90–92% trên DFDC public.

---

## 7️⃣ Ghi chú kỹ thuật

* Framework: **PyTorch 2.5+**
* Backend: **FastAPI**
* Frontend: **HTML / JS / CSS**
* Face Detector: **RetinaFace**
* GPU: **CUDA 12+**

---

## 👥 Tác giả

**Lê Minh Anh** – PTIT

> Dự án: *Hệ thống phát hiện video Deepfake dựa trên mô hình GenConViT (Generative Convolutional Vision Transformer)*
MIT License © 2025 AnhLM027
