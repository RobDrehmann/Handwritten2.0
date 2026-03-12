import base64
import io

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps

print("starting main.py")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image: str


class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward_with_activations(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        z1 = self.fc1(x)
        a1 = self.relu(z1)

        z2 = self.fc2(a1)
        a2 = self.relu(z2)

        z3 = self.fc3(a2)

        return z3, a1, a2

    def forward(self, x):
        z3, _, _ = self.forward_with_activations(x)
        return z3


model = DigitNet()

try:
    model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device("cpu")))
    model.eval()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False


def preprocess_image(base64_image: str):
    image_data = base64_image.split(",")[1]
    decoded = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(decoded)).convert("L")

    image = ImageOps.autocontrast(image)
    img = np.array(image).astype(np.uint8)

    if img.mean() > 127:
        img = 255 - img

    img[img < 20] = 0

    coords = np.argwhere(img > 0)

    if coords.size == 0:
        img_resized = np.zeros((28, 28), dtype=np.float32)
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = img[y0:y1, x0:x1]

        h, w = cropped.shape
        scale = 20.0 / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        pil_cropped = Image.fromarray(cropped)
        pil_resized = pil_cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        new_img = Image.new("L", (28, 28), 0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        new_img.paste(pil_resized, (paste_x, paste_y))

        arr = np.array(new_img).astype(np.float32)
        ys, xs = np.nonzero(arr > 0)
        if len(xs) > 0 and len(ys) > 0:
            cx = xs.mean()
            cy = ys.mean()
            shift_x = int(round(13.5 - cx))
            shift_y = int(round(13.5 - cy))

            shifted = Image.new("L", (28, 28), 0)
            shifted.paste(new_img, (shift_x, shift_y))
            new_img = shifted

        new_img.save("debug_input.png")
        img_resized = np.array(new_img).astype(np.float32)

    input_pixels = img_resized.copy() / 255.0
    normalized = (input_pixels - 0.1307) / 0.3081

    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, input_pixels.flatten().tolist()


def normalize_activations(values):
    arr = np.array(values, dtype=np.float32)
    max_val = float(arr.max()) if arr.size > 0 else 0.0
    if max_val <= 1e-8:
        return arr.tolist()
    return (arr / max_val).tolist()


@app.get("/")
def root():
    return {"message": "Digit recognizer API running"}


@app.post("/predict")
def predict(request: PredictRequest):
    if not MODEL_LOADED:
        return {
            "error": "Model file not found. Train the model first.",
            "prediction": None,
            "probabilities": [0.0] * 10,
            "hidden1": [0.0] * 128,
            "hidden2": [0.0] * 64,
            "input_pixels": [0.0] * 784,
        }

    tensor, input_pixels = preprocess_image(request.image)

    with torch.no_grad():
        logits, hidden1, hidden2 = model.forward_with_activations(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))

        probs_percent = (probs * 100).tolist()
        hidden1_vals = normalize_activations(hidden1[0].cpu().numpy())
        hidden2_vals = normalize_activations(hidden2[0].cpu().numpy())

    return {
        "prediction": pred,
        "probabilities": probs_percent,
        "hidden1": hidden1_vals,
        "hidden2": hidden2_vals,
        "input_pixels": input_pixels,
    }