from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch
from torchvision import transforms

# Load FaceNet model (pretrained on VGGFace2)
def setup_facenet():
    model = InceptionResnetV1(pretrained='vggface2').eval()  # Use the VGGFace2 pretrained weights
    return model


# Function to generate embedding using FaceNet
def get_facenet_embedding(model, image_path):
    image = Image.open(image_path).convert('RGB')
    
    # Preprocessing (Resize, crop and normalization)
    preprocess = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.squeeze().numpy()
