import torch
import torchvision.transforms as T
from torchvision.models import resnet50
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(weights="IMAGENET1K_V1")
model.fc = torch.nn.Identity()
model.eval().to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256,192)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def extract_appearance_feature(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img)
    feat = feat.cpu().numpy().flatten()
    return feat / np.linalg.norm(feat)