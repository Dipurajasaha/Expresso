# Expresso: Real-Time Facial Emotion Recognition 📸

Expresso is a web-based application that detects and classifies human emotions from a live webcam feed in real-time.  
It leverages a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras and serves the video stream using a Flask backend.

---

## ✨ Features
- **Real-Time Detection**: Analyzes video frames directly from the user's webcam.  
- **Seven Emotion Classes**: Classifies faces into one of seven emotions: *angry, disgust, fear, happy, neutral, sad, surprise*.  
- **Face Detection**: Uses OpenCV's Haar Cascade classifier to accurately locate faces in the video stream before emotion classification.  
- **Web Interface**: A clean and simple front-end built with Flask, HTML, and CSS to display the camera feed and predictions.  

---

## 🛠️ Tech Stack & Libraries
- **Backend**: Python, Flask  
- **Deep Learning**: TensorFlow, Keras  
- **Computer Vision**: OpenCV, NumPy  
- **Data Science & Plotting**: Scikit-learn, Matplotlib, Seaborn  

---

## 📂 Project Structure
```bash
Expresso/
├── 📂 data/                     # Contains the image dataset
├── 📂 notebooks/                # Jupyter notebook for EDA and model prototyping
├── 📂 saved_models/             # Stores the trained .h5 model file
├── 📂 src/                      # Source code modules
│   ├── data_preprocessing.py    # Data loading and augmentation
│   ├── model.py                 # CNN architecture definition
│   ├── train.py                 # Script to train the model
│   └── predict.py               # Functions for making predictions
├── 📜 app.py                    # Main Flask application file
├── 📜 requirements.txt          # Project dependencies
├── 📜 haarcascade_frontalface_default.xml # OpenCV face detector
└── 📜 README.md                 # You are here!
