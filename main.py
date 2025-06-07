from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import io
import base64
from PIL import Image
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["https://e2fc0427.sitepreview.org"], 
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Input model
class ImageData(BaseModel):
    image: str  # base64 string

# Global state
model = None
prediction_cache = {}
last_loaded_time = None

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

# Load model lazily
def get_model():
    global model, last_loaded_time
    current_time = time.time()

    if model is None or last_loaded_time is None or (current_time - last_loaded_time > 1800):
        device = torch.device('cpu')  # Use 'cuda' if using GPU
        model_instance = EfficientNet.from_name('efficientnet-b0')
        model_instance._fc = torch.nn.Linear(model_instance._fc.in_features, 4)
        model_instance.load_state_dict(torch.load('model/sign_language_model.pth', map_location=device))
        model_instance.eval()
        model = model_instance
        last_loaded_time = current_time
        logger.info("Model loaded or reloaded at %s", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)))

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
        image_hash = get_image_hash(data.image)
        if image_hash in prediction_cache:
            logger.info("Cache hit for image")
            return prediction_cache[image_hash]

        image = decode_base64_image(data.image)
        input_tensor = transform(image).unsqueeze(0)

        model_instance = get_model()
        with torch.no_grad():
            outputs = model_instance(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
            confidence_score = confidence.item() * 100

            # Include confidence score in the response
            if confidence_score < 80:
                result = {
                    "sign": "",
                    "message": "Sign not recognized—please try one of the supported signs!",
                    "confidence": confidence_score
                }
            else:
                predicted_class = class_map[predicted.item()]
                result = {
                    "sign": predicted_class["label"],
                    "message": "Prediction successful!",
                    "audio": f"/Audio/{predicted_class['audio']}.mp3",
                    "confidence": confidence_score
                }

            prediction_cache[image_hash] = result
            logger.info("Prediction made with %.2f%% confidence for class %s", confidence_score, predicted_class["label"] if confidence_score >= 80 else "Unknown")
            return result

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail="Failed to process image")

@app.get("/")
async def root():
    return {"message": "Sign Language Recognition Backend"}

@app.head("/")
async def head_root():
    return Response(status_code=200)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": time.time()
    }
