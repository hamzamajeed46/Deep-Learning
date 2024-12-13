### Handwritten Digit Recognition with Streamlit
This repository contains a complete end-to-end implementation for recognizing handwritten digits using a neural network, trained on the MNIST dataset. The project includes a Jupyter notebook for data loading and model training, a saved model for inference, and a Streamlit-based interactive web application for real-time digit recognition.

**Overview**
This project allows a user to:

Train and save a neural network model using MNIST dataset.
Use a Streamlit application to upload an image and recognize the handwritten digit from it.
The project uses:

PyTorch for model creation and training.
Streamlit for a simple user interface to interact with the trained model.

**Requirements**
Include these dependencies in a requirements.txt file:
torch
torchvision
streamlit
numpy
Pillow
matplotlib

Install the dependencies:
pip install -r requirements.txt

**How to Train the Model**
Navigate to the Jupyter Notebook:
Open notebook/train_model.ipynb.
Run the notebook cells sequentially to:
Load the MNIST dataset.
Train the model.
Save the trained model (digit_classifier.pth) in the model/ directory.

**How to Run the Streamlit App**
Ensure the trained model is saved: The model should be in the model/ directory. The file is named digit_classifier.pth.

Start the Streamlit server using this command:

streamlit run app/streamlit_app.py
Open the Streamlit application in the browser. Follow the provided instructions to upload an image, and the model will predict the handwritten digit.

**Model Input Details**
The model expects:

Grayscale images of size 28x28.
Normalized pixel values between [-1, 1].
The app processes uploaded images accordingly to match these requirements before prediction.
