import dlib, cv2, os
from tqdm import tqdm

detector = dlib.get_frontal_face_detector()
input_dir = "Images"
output_dir = "Faces"
os.makedirs(output_dir, exist_ok=True)

for f in tqdm(os.listdir(input_dir)):
    path = os.path.join(input_dir, f)
    img = cv2.imread(path)
    if img is None: continue
    faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i, face in enumerate(faces):
        x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_crop = img[y:y2, x:x2]
        cv2.imwrite(f"{output_dir}/{os.path.splitext(f)[0]}_{i}.jpg", face_crop)
