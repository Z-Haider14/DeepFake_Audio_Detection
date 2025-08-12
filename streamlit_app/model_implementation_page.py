import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import librosa
import tempfile
import plotly.express as px  # Added for custom bar chart

def show():
    # ======= GLOBAL CONFIGURATION =======
    MODEL_PATH = r'my_model.h5'
    SEGMENT_INTERVAL = 0.25  # Now defined globally
    WINDOW_SIZE = 5
    SILENCE_THRESHOLD = 0.2  # 2% of max RMS
    # ====================================

    # Feature Extraction Functions with silence removal
    def extract_features(file_path, sr=16000, silence_threshold=SILENCE_THRESHOLD):
        """
        Extract audio features excluding silent segments
        """
        y, _ = librosa.load(file_path, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        
        features = []
        num_segments = int(duration / SEGMENT_INTERVAL)
        
        # Calculate RMS for entire audio to determine silence threshold
        frame_length = int(SEGMENT_INTERVAL * sr)
        hop_length = frame_length
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        max_rms = np.max(rms)
        silence_threshold = max_rms * silence_threshold
        
        for i in range(num_segments):
            start = int(i * SEGMENT_INTERVAL * sr)
            end = int((i + 1) * SEGMENT_INTERVAL * sr)
            
            if end > len(y):
                break
                
            y_segment = y[start:end]
            
            # Calculate RMS for current segment
            segment_rms = librosa.feature.rms(y=y_segment)[0][0]
            
            # Skip silent segments
            if segment_rms < silence_threshold:
                continue
                
            segment_features = []

            # MFCC (13 coefficients)
            mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=13)
            segment_features.extend(np.mean(mfcc, axis=1))

            # Spectral features
            centroid = librosa.feature.spectral_centroid(y=y_segment, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr)
            
            segment_features.append(np.mean(centroid))
            segment_features.append(np.mean(bandwidth))
            segment_features.append(np.mean(rolloff))

            # Temporal features
            zcr = librosa.feature.zero_crossing_rate(y_segment)
            
            segment_features.append(np.mean(zcr))
            segment_features.append(segment_rms)  # Use actual RMS we calculated

            features.append(segment_features)
            
        return features

    def build_feature_names():
        """Generate consistent feature names"""
        return [f'mfcc_{i+1}' for i in range(13)] + [
            'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
            'zero_crossing_rate', 'rms'
        ]

    def process_audio_file(file_path):
        """Process audio file and return features DataFrame with silence removal"""
        filename = os.path.splitext(os.path.basename(file_path))[0]
        features = extract_features(file_path)
        
        if not features:
            return None
        
        df = pd.DataFrame(features, columns=build_feature_names())
        df['filename'] = filename  # Store filename
        return df

    # Model Prediction Functions
    def prepare_data(X):
        """Prepare data for model prediction with sliding window"""
        data = []
        for i in range(len(X)):
            row = X.iloc[i].values
            row_data = [row[j:j + WINDOW_SIZE] for j in range(len(row) - WINDOW_SIZE)]
            data.append(row_data)
        return np.array(data)

    @st.cache_resource
    def load_model():
        """Load and cache the Keras model"""
        return keras.models.load_model(MODEL_PATH)

    def predict_audio(file_path):
        """Complete prediction pipeline with silence removal"""
        with st.spinner('Extracting audio features (excluding silence)...'):
            df = process_audio_file(file_path)
        
        # Check if any segments were extracted
        if df is None or len(df) == 0:
            return "INSUFFICIENT SPEECH", 0.0, None, 0
        
        with st.spinner('Loading detection model...'):
            model = load_model()
        
        with st.spinner('Preparing data for prediction...'):
            X_to_predict = df.drop(columns=['filename'])
            X_processed = prepare_data(X_to_predict)
        
        with st.spinner('Analyzing audio segments...'):
            predictions = model.predict(X_processed).flatten()
            
            # Corrected confidence calculation
            df['prediction'] = ["REAL" if pred >= 0.5 else "FAKE" for pred in predictions]
            df['confidence'] = [pred if pred >= 0.5 else 1-pred for pred in predictions]
        
        # Get final prediction and confidence
        final_pred = df['prediction'].mode()[0] if len(df) > 0 else "UNDETERMINED"
        confidence_avg = df['confidence'].mean() if len(df) > 0 else 0.0
        
        return final_pred, confidence_avg, df, len(df)

    # Streamlit UI
    st.title("Audio Guard")
    st.markdown("#### 🔊Deepfake Audio Detection:")
    st.markdown("""
    Upload an audio file (WAV format) to detect if it's real or AI-generated.
    Silent segments are automatically excluded from analysis.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

    if uploaded_file is not None:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Make prediction
        final_pred, confidence_avg, result_df, segments_analyzed = predict_audio(file_path)
        
        # Display results
        st.subheader("Detection Results")
        
        if final_pred == "REAL":
            st.success(f"✅ Prediction: {final_pred}")
        elif final_pred == "FAKE":
            st.error(f"❌ Prediction: {final_pred}")
        elif final_pred == "INSUFFICIENT SPEECH":
            st.warning("⚠️ Could not analyze: Audio contains insufficient speech segments")
        else:
            st.warning("⚠️ Analysis inconclusive")
        
        # Show statistics
        if result_df is not None and len(result_df) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Speech Segments Analyzed", segments_analyzed)
            with col2:
                # Now SEGMENT_INTERVAL is defined globally
                st.metric("Segment Duration", f"{SEGMENT_INTERVAL} seconds")
            
            # Show prediction distribution
            pred_counts = result_df['prediction'].value_counts()
            if 'REAL' in pred_counts:
                st.metric("REAL Segments", f"{pred_counts['REAL']} ({pred_counts['REAL']/len(result_df):.1%})")
            if 'FAKE' in pred_counts:
                st.metric("FAKE Segments", f"{pred_counts['FAKE']} ({pred_counts['FAKE']/len(result_df):.1%})")
            
            # MODIFIED: Show average confidence by prediction type
            st.subheader("Average Confidence by Prediction Type")
            
            # Calculate average confidence for each prediction type
            avg_confidence = result_df.groupby('prediction')['confidence'].mean().reset_index()

            # Create a custom bar chart with specified colors
            fig = px.bar(
                avg_confidence,
                x='prediction',
                y='confidence',
                color='prediction',
                color_discrete_map={'REAL': 'green', 'FAKE': 'red'},
                labels={'confidence': 'Average Confidence', 'prediction': 'Prediction'},
                title=''
            )

            # Customize layout to make it smaller and centered
            fig.update_layout(
                xaxis_title='Prediction Type',
                yaxis_title='Average Confidence',
                yaxis_tickformat='.0%',
                showlegend=False,
                # Set smaller dimensions
                width=500,   # Reduced width
                height=400,  # Reduced height
                # Center the plot
                margin=dict(l=50, r=50, b=50, t=30, pad=10),  # Balanced margins
                autosize=False
            )

            # Create centered columns for layout
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Display the plot in the center column
                st.plotly_chart(fig, use_container_width=True)

            # Clean up temporary file
            os.unlink(file_path)
if __name__ == "__main__":
    show()