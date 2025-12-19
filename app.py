import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
from gtts import gTTS
import io

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('hotdog_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(img):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return 'Hot Dog' if predicted.item() == 0 else 'Not Hot Dog'

def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("speech.mp3")
        st.audio("speech.mp3")
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")

# UI
st.title('🌭 Hot Dog Classifier')
st.write("Upload an image or take a photo to check if it's a hot dog!")

uploaded_file = st.file_uploader("📤 Pick a picture", type=["jpg", "png", "jpeg"])
camera_input = st.camera_input("📷 Or take a picture")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    prediction = predict_image(img)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption='Uploaded Image', use_column_width=True)
    with col2:
        st.markdown(f"### Prediction: **{prediction}**")
        if prediction == 'Hot Dog':
            st.success("✅ It's a hot dog!")
        else:
            st.error("❌ Not a hot dog!")
    
    speak_text(f"The image is a {prediction}")

elif camera_input:
    img = Image.open(io.BytesIO(camera_input.getvalue())).convert('RGB')
    prediction = predict_image(img)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption='Captured Image', use_column_width=True)
    with col2:
        st.markdown(f"### Prediction: **{prediction}**")
        if prediction == 'Hot Dog':
            st.success("✅ It's a hot dog!")
        else:
            st.error("❌ Not a hot dog!")
    
    speak_text(f"The image is a {prediction}")