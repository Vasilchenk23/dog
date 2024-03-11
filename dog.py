import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

owner_image = Image.open('owner.jpg')
owner_image_tensor = transform(owner_image).unsqueeze(0)

video_dog_happy = cv2.VideoCapture('dog_happy.mp4')
video_dog_barking = cv2.VideoCapture('dog_barking.mp4')

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return len(faces) > 0


def compare_faces(frame, owner_image_tensor):
    with torch.no_grad():
        frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = transform(frame_image).unsqueeze(0)

        owner_features = model(owner_image_tensor)
        frame_features = model(frame_tensor)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cos(frame_features, owner_features)

        threshold = 0.7
        if similarity.item() > threshold:
            return "owner"
        else:
            return "other"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if detect_face(frame):
        face_status = compare_faces(frame, owner_image_tensor)

        if face_status == "owner":
            while True:
                ret, dog_happy_frame = video_dog_happy.read()
                if not ret:
                    video_dog_happy.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break
                cv2.imshow('Dog Happy', dog_happy_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
        else:
            while True:
                ret, dog_barking_frame = video_dog_barking.read()
                if not ret:
                    video_dog_barking.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break
                cv2.imshow('Dog Barking', dog_barking_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
    else:
        cv2.putText(frame, 'Dog is resting', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Dog Resting', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_dog_happy.release()
video_dog_barking.release()
cv2.destroyAllWindows()