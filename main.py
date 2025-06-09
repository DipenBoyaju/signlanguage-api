from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import hashlib
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://prateekinnovations.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Input model
class ImageData(BaseModel):
    image: str  # base64 string

# Sign-label mappings
class_map = [
    {"label": "Dhanyabaad", "audio": "Dhanyabaad"},
    {"label": "Ghar", "audio": "Ghar"},
    {"label": "Ma", "audio": "Ma"},
    {"label": "Namaskaar", "audio": "Namaskaar"}
]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Global state
model = None
last_loaded_time = None

# Load model lazily
def get_model():
    global model, last_loaded_time
    current_time = time.time()

    if model is None or last_loaded_time is None or (current_time - last_loaded_time > 1800):
        device = torch.device('cpu')
        model_instance = EfficientNet.from_name('efficientnet-b0')
        model_instance._fc = torch.nn.Linear(model_instance._fc.in_features, 4)
        model_instance.load_state_dict(torch.load('model/sign_language_model.pth', map_location=device))
        model_instance.eval()
        model = model_instance
        last_loaded_time = current_time
        logger.info("Model loaded/reloaded at %s", time.strftime('%Y-%m-%d %H:%M:%S'))

    return model

# Hashing function for cache
def get_image_hash(image_str: str):
    return hashlib.sha256(image_str.encode()).hexdigest()

# Decode base64 to image
def decode_base64_image(image_base64: str) -> Image.Image:
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    image_data = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_data)).convert('RGB')

@app.post("/predict")
async def predict_sign(data: ImageData):
    try:
        image = decode_base64_image(data.image)
        input_tensor = transform(image).unsqueeze(0)

        model_instance = get_model()
        with torch.no_grad():
            outputs = model_instance(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
            confidence_score = confidence.item() * 100

            if confidence_score < 80:
                result = {
                    "success": False,
                    "sign": "",
                    "message": "Please try again. Sign not recognized clearly.",
                    "confidence": confidence_score
                }
            else:
                predicted_class = class_map[predicted.item()]
                result = {
                    "success": True,
                    "sign": predicted_class["label"],
                    "message": "Prediction successful!",
                    "audio": f"{predicted_class['audio']}.mp3",
                    "confidence": confidence_score
                }

            logger.info(f"Prediction made: {result}")
            return result

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail="Failed to process image")

@app.get("/")
async def root():
    return {"message": "Sign Language Recognition Backend"}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": time.time()
    }