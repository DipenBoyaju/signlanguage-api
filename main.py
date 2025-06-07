from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import base64
import torch
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import torch.nn.functional as F
from fastapi import Response

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:5173"],
    # allow_origins=["https://prateek-1.vercel.app"],
    allow_origins=["https://e2fc0427.sitepreview.org"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ImageData(BaseModel):
    image: str  # base64 string

# Load model
device = torch.device('cpu')
model = EfficientNet.from_name('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 4)
model.load_state_dict(torch.load('model/sign_language_model.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['Dhanyabaad', 'Ghar', 'Ma', 'Namaskaar']
audio_filenames = ['Dhanyabaad', 'Ghar', 'Ma', 'Namaskaar']

@app.post("/predict")
async def predict_sign(data: ImageData):
    try:
        image_data = base64.b64decode(data.image.split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
            confidence_score = confidence.item() * 100

            if confidence_score < 80:
                return {"sign": "", "message": "Sign not recognized—please try one of the supported signs!"}

            pred_class = class_names[predicted.item()]
            audio_filename = audio_filenames[predicted.item()]
            audio_path = f"../Audio/{audio_filename}.mp3"

            # with open(audio_path, "rb") as audio_file:
            #     audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

            # return {
            #     "sign": pred_class,
            #     "message": "Prediction successful!",
            #     "audio": f"data:audio/mp3;base64,{audio_data}"
            # }
            return {
                "sign": pred_class,
                "message": "Prediction successful!",
                "audio": f"/Audio/{audio_filename}.mp3"
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.get("/")

async def root():
    return {"message": "Sign Language Recognition Backend"}

@app.head("/")
async def head_root():
    return Response(status_code=200)