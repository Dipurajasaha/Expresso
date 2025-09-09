# Expresso: Real-Time Facial Emotion Recognition 📸

Expresso is a web-based application that detects and classifies human emotions from a live webcam feed in real time.  
It leverages a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras and serves the video stream using a Flask backend.

---

## 🌟 Live Demo
Below is a demonstration of the application identifying emotions from a live camera feed.  
*(Drop in a GIF or screenshot if you have one—makes it pop!)*

---

## ✨ Features
- **Real-Time Detection**: Processes webcam frames on-the-fly.  
- **Seven Emotion Classes**: Recognizes *angry, disgust, fear, happy, neutral, sad,* and *surprise*.  
- **Face Detection**: Uses OpenCV's Haar Cascade to locate faces before classifying emotions.  
- **Web Interface**: Clean Flask + HTML/CSS front-end displaying live feed with emotion labels.

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
├── Dataset/                  # Raw image dataset (place dataset here)
├── notebooks/                # Jupyter notebooks for EDA and prototyping
│   └── 01_cnn_fer_prototyping.ipynb
├── saved_models/             # Trained model files
│   └── best_model.h5
├── src/                      # Core Python modules
│   ├── data_preprocessing.py  
│   ├── model.py              
│   ├── train.py              
│   └── predict.py            
├── app.py                    # Flask application entry point
├── requirements.txt          # Project dependencies
├── haarcascade_frontalface_default.xml  # OpenCV face detector
└── README.md                 # You're looking at it!
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- Webcam/Camera access
- Git (optional, for cloning)

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/Dipurajasaha/Expresso.git
   cd Expresso
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to `http://localhost:5000`

---

## 📊 Model Details
- **Architecture**: Convolutional Neural Network (CNN)
- **Dataset**: FER-2013 (Facial Expression Recognition)
- **Training**: Custom training pipeline in `src/train.py`
- **Accuracy**: [Add your model's accuracy here]
- **Input Size**: 48x48 grayscale images

---

## 🔧 Usage
1. Launch the Flask application using `python app.py`
2. Allow camera permissions when prompted by your browser
3. Position your face in front of the camera
4. Watch as the model predicts your emotion in real-time!

---

## 📈 Model Training
To retrain the model with your own dataset:

1. Place your dataset in the `Dataset/` folder
2. Run the preprocessing script:
   ```bash
   python src/data_preprocessing.py
   ```
3. Train the model:
   ```bash
   python src/train.py
   ```

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments
- FER-2013 dataset creators
- OpenCV community for face detection algorithms
- TensorFlow/Keras teams for the deep learning framework

---

## 📧 Contact
**Dipuraj Asaha** - [Your Email] - [Your LinkedIn]

Project Link: [https://github.com/Dipurajasaha/Expresso](https://github.com/Dipurajasaha/Expresso)