import streamlit as st
from PIL import Image

def show():
    st.title("Audio Guard - Model Design")

    st.markdown("## Project Overview")
    st.markdown("""
    Our semester project focused on detecting **audio deepfakes** using machine learning.  
    We used labeled **real** and **fake** voice recordings from Kaggle and other public sources, 
    and built a classifier using a deep **LSTM neural network**.
    """)

    st.markdown("---")
    st.markdown("## Datasets Used")
    st.markdown("""
    We combined multiple datasets to ensure diversity in accents, tone, and types of speech:
    
    - **DEEP-VOICE: DeepFake Voice Recognition**
    - **VoxCeleb1** (Indian celebrity audio WAV files)
    - **Fake-or-Real (FoR) Dataset** – Designed for detecting synthetic speech

    All audio files were taken from these datastes and placed in two folder named **real** and **fake**.
    """)

    st.markdown("---")
    st.markdown("## Feature Extraction with Librosa")
    st.markdown("""
    We extracted the following **18 features** from each audio sample using the **Librosa** library:
    
    ```python
    [
        'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7',
        'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13',
        'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
        'zero_crossing_rate', 'rms'
    ]
    ```
    
    These features help capture:
    - **Pitch and tone** (MFCC)
    - **Energy** (RMS)
    - **Timbre characteristics** (centroid, bandwidth, rolloff)
    - **Voicing behavior** (zero crossing rate)
                
    **while feature extraction, we also removed silent segments to focus on active speech. this was done by**
    **thrusholding the RMS energy of each segment. If the RMS energy is below a certain threshold, we consider**
    **it silent and skip that segment.** 
    """)

    st.markdown("---")
    st.markdown("## Data Preparation")

    st.markdown("""
    - We **balanced** the dataset with an equal number of real and fake samples.
    - Then, we **reshaped** the tabular feature data into 2D temporal windows for LSTM input using this function:
    """)

    st.code("""
def prepare_data(X, window_size=10):
    data = []
    for i in range(len(X)):
        row = X.iloc[i].values
        row_data = []
        for j in range(len(row) - window_size):
            window = row[j:j + window_size]
            row_data.append(window)
        data.append(row_data)
    return np.array(data)

new_X = prepare_data(X, window_size=5)
    """, language="python")

    st.markdown("""
    This reshaping creates small time-based chunks of feature sequences.  
    It's crucial for **LSTM** to capture temporal patterns in the audio signal.
    """)

    st.markdown("---")
    st.markdown("## LSTM Model Architecture")

    st.code("""
model = Sequential()

input_shape = (X_train.shape[1], X_train.shape[2])
model.add(Input(shape=input_shape))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])
    """, language="python")

    st.markdown("""
    - **2 LSTM layers** capture time-based dependencies
    - **Dropout layers** reduce overfitting
    - **Dense layers** build up to final decision
    - **Sigmoid output** predicts real (1) or fake (0)
    """)

    st.markdown("---")
    st.markdown("## Why LSTM?")
    st.markdown("""
    We used LSTM over CNNs, ANNs, or ensemble models for the following reasons:

    - Audio is **sequential** — LSTMs handle temporal relationships better
    - LSTMs **retain memory** over time, useful for catching subtle manipulations in voice
    - CNNs and ANNs treat features **independently**, missing contextual flow
    - Ensemble techniques like **boosting/bagging** work better with tabular data, not sequences
    """)

    st.markdown("---")
    st.markdown("## Accuracy Metrics")

    st.markdown("""
    Our model was evaluated on a **balanced test set of 17,615 samples**, half real and half fake.  
    Here are the detailed classification metrics:
    """)

    st.markdown("""
    | Metric       | Real (0) | Fake (1) | Average |
    |--------------|----------|----------|---------|
    | Precision    | 0.91     | 0.93     | 0.92    |
    | Recall       | 0.93     | 0.91     | 0.92    |
    | F1-Score     | 0.92     | 0.92     | 0.92    |
    | Support      | 8808     | 8807     | 17615   |
    """)

    st.markdown("""
    - **Accuracy:** 92% overall
    - The model performs equally well on both real and fake samples
    - Slight variation in precision and recall is acceptable due to natural variation in audio complexity
    """)
    
    st.markdown("To visualize accuracy during training:")
    try:
        img = Image.open(r"Evaluation\Training_vs_validation_plot.png")
        st.image(img, caption='Model Training History: Accuracy and Loss')
    except FileNotFoundError:
        st.warning("Plot image not found. Using placeholder")
        # Create a placeholder plot if real image is missing
        placeholder = Image.new('RGB', (600, 400), color='gray')
        st.image(placeholder, caption='Training Plot (Placeholder)')
    
    st.markdown("Confusion Matrix on test Data:")
    try:
        img = Image.open(r"Evaluation\output.png")
        st.image(img, caption='Confusion Matrix')
    except FileNotFoundError:
        st.warning("Plot image not found. Using placeholder")
        # Create a placeholder plot if real image is missing
        placeholder = Image.new('RGB', (600, 400), color='gray')
        st.image(placeholder, caption='Training Plot (Placeholder)')

    st.markdown("---")
    st.markdown("## Final Thoughts")

    st.markdown("""
    This project taught us about:
    - Audio signal processing
    - Practical application of LSTM for real-world AI problems
    - How deepfake audio differs from natural speech patterns

    The system is a **proof-of-concept** and opens up further possibilities in speech security and AI ethics.
    """)

if __name__ == "__main__":
    show()