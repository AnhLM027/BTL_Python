let API_URL = '';
if (window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost') {
  API_URL = 'http://127.0.0.1:9000/deepfake';
} else {
  API_URL = 'https://aispeech.ptit.edu.vn/deepfake';
}

const form = document.getElementById('uploadForm');
const loading = document.getElementById('loading');
const preview = document.getElementById('preview');
const videoElement = document.getElementById('uploadedVideo');
const detectBtn = document.getElementById('detectBtn');
const detecting = document.getElementById('detecting');
const resultDiv = document.getElementById('result');
let uploadedFilename = null;

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = document.getElementById('videoInput').files[0];
  if (!file) return alert("Vui lòng chọn video!");

  const formData = new FormData();
  formData.append("file", file);

  loading.style.display = "block";
  preview.style.display = "none";
  resultDiv.innerHTML = "";

  const res = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
  const data = await res.json();
  loading.style.display = "none";

  if (data.error) {
    resultDiv.innerHTML = `<p class="error">Lỗi: ${data.error}</p>`;
    return;
  }

  uploadedFilename = data.filename;
  const uploadedURL = `${API_URL}/uploads/${encodeURIComponent(uploadedFilename)}`;
  videoElement.src = uploadedURL;
  preview.style.display = "block";
});

detectBtn.addEventListener('click', async () => {
  if (!uploadedFilename) return alert("Chưa có video nào được tải lên!");

  detecting.style.display = "block";
  resultDiv.innerHTML = "";
  
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: uploadedFilename })
  });

  const data = await res.json();
  detecting.style.display = "none";

  if (data.error) {
    resultDiv.innerHTML = `<p class="error">Lỗi: ${data.error}</p>`;
    return;
  }

  const labelClass = data.label === "FAKE" ? "fake" : "real";
  const labelText = data.label === "FAKE" ? "FAKE" : "REAL";
  const prob = data.confidence ? (parseFloat(data.confidence) * 100).toFixed(2) : "—";

  resultDiv.innerHTML = `
    <h3>Kết quả phân tích</h3>
    <p><b>Xác suất Deepfake:</b> ${prob}%</p>
    <p><b>Kết luận:</b> <span class="${labelClass}">${labelText}</span></p>
  `;
});
