# DeepFake_Audio_Detection
Deepfake Voice Detection with LSTM Detect AI-generated voices using signal processing (Librosa Libarary) + LSTM deep learning. Achieves 92% accuracy. Includes:  🎚️ Feature extraction notebook  🧠 LSTM model training notebook  🤖 Pretrained model (my_model.h5)  🌐 Streamlit demo app
Deepfake Voice Detection System
Accurately identify synthetic voices using deep learning (92.09% accuracy)

This repository contains a production-ready pipeline for detecting AI-generated deepfake voices by combining signal processing (MFCCs, chroma features) with deep learning (LSTM model). Designed to combat voice fraud and misinformation.

## 📁 Repository Contents
File/Folder	Description
EDA Images/	Visualizations of audio features and dataset analysis
Evaluation/	Performance metrics (confusion matrices, ROC curves)
streamit_app/	Interactive web demo for real-time detection
testing_audio/	Sample recordings for validation
Combined_extracted_features_2.csv	Preprocessed feature dataset
feature_extraction.ipynb	Jupyter notebook for audio feature extraction (MFCCs, spectrograms)
DT-Detection_2_Improved.ipynb	Main detection model training & evaluation notebook
d1_v.csv	Supplementary dataset/validation logs
my_model.h5	Pretrained Keras model (ready for inference)
Project_Report.docx	Complete technical documentation
## ✨ Key Features
Hybrid Detection: Combines traditional audio features + deep learning spectrogram analysis

Production-Ready: Includes trained model (my_model.h5) and Streamlit demo

Comprehensive: From EDA to deployment-ready code

State-of-the-Art: 92.09% accuracy on diverse datasets (FoR, VoxCeleb1, DEEP-VOICE)


## 💡 Use Cases
Banking/fraud prevention (vishing detection)

Media authenticity verification

Secure voice authentication systems

Contributions welcome! Help us stay ahead of evolving deepfake tech.
