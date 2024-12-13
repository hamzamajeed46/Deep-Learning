import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn


# Define the same model architecture used during training
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the trained model
model = DigitClassifier()
model.load_state_dict(torch.load("digit_classifier.pth", map_location=torch.device('cpu')))
model.eval()


# Prediction function
def predict_digit(image):
    # Preprocess image to 28x28 grayscale
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    image = (image - 0.5) / 0.5  # Scale image values to [-1, 1]
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, 28, 28)

    # Perform prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)  # Get predicted class
    return predicted.item()


# Streamlit Interface
st.title("Handwritten Digit Recognition")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        predicted_digit = predict_digit(image)
        st.write(f"Predicted Digit: {predicted_digit}")
