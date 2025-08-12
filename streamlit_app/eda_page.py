import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def show():
    # Set page config
    st.set_page_config(
        page_title="Data Visualization",
        page_icon="🎵",
        layout="wide"
    )

    # Feature descriptions
    feature_descriptions = {
        "mfcc": "MFCC (Mel Frequency Cepstral Coefficients): Represents the short-term power spectrum of sound. This plot shows the distribution of 13 MFCCs used for characterizing the timbre and tonal aspects of audio.",
        "spectral_centroid": "Spectral Centroid: Indicates where the 'center of mass' of the spectrum is located. Higher values imply brighter sounds.",
        "spectral_bandwidth": "Spectral Bandwidth: Measures the spread of the spectrum around the centroid. Helps capture how wide the frequency content is.",
        "spectral_rolloff": "Spectral Rolloff: The frequency below which 85% of the spectral energy is contained. Often used to distinguish voiced/unvoiced or tonal/noisy sounds.",
        "zero_crossing_rate": "Zero Crossing Rate: Measures how frequently the signal changes sign. Higher values typically indicate noisy or high-frequency content.",
        "rms": "RMS Energy: Root Mean Square energy represents the loudness of the signal in each frame. Useful for detecting silence and active speech segments.",
    }

    # Load data
    @st.cache_data
    def load_data():
        return pd.read_csv(r"df_v.csv")

    df_v = load_data()

    # Define columns
    mfcc_cols = [f'mfcc_{i}' for i in range(1, 14)]
    other_cols = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 
                'zero_crossing_rate', 'rms']

    # Title
    st.title("Data Visualization")
    st.markdown("#### Visualization of Extracted Audio Features from Voice Recordings" \
    " with Fake and Real Labels")

    # === 1. MFCCs in Subplots ===
    st.header("MFCC Features Distribution")
    st.markdown(f"**Description:** {feature_descriptions['mfcc']}")

    def plot_mfcc_subplots():
        n_cols = 3
        n_rows = (len(mfcc_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols, 
            subplot_titles=[f'Distribution of {col}' for col in mfcc_cols]
        )
        
        for idx, col in enumerate(mfcc_cols, 1):
            row = (idx - 1) // n_cols + 1
            col_num = (idx - 1) % n_cols + 1
            
            # Add fake data trace
            fig.add_trace(
                go.Histogram(
                    x=df_v[df_v['LABEL'] == 'fake'][col],
                    name='Fake',
                    marker_color='red',
                    opacity=0.75,
                    nbinsx=50,
                    showlegend=True if idx == 1 else False
                ),
                row=row, col=col_num
            )
            
            # Add real data trace
            fig.add_trace(
                go.Histogram(
                    x=df_v[df_v['LABEL'] == 'real'][col],
                    name='Real',
                    marker_color='green',
                    opacity=0.75,
                    nbinsx=50,
                    showlegend=True if idx == 1 else False
                ),
                row=row, col=col_num
            )
        
        fig.update_layout(
            height=n_rows * 300,
            width=1200,
            title_text="MFCC Features Distribution",
            title_x=0.5,
            legend_title_text='Label',
            bargap=0.1,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    plot_mfcc_subplots()

    # === 2. Other Audio Features ===
    st.header("Other Audio Features")

    def plot_other_features():
        for col in other_cols:
            st.subheader(f"Feature: {col}")
            
            # Display feature description right below the subheader
            if col in feature_descriptions:
                st.markdown(feature_descriptions[col])
            elif col.lower().replace('spectral_', '') in feature_descriptions:
                # Handle variations in naming
                key = col.lower().replace('spectral_', '')
                st.markdown(feature_descriptions[key])
            
            # Create and display the plot
            fig = px.histogram(
                df_v,
                x=col,
                color='LABEL',
                color_discrete_map={'fake': 'red', 'real': 'green'},
                opacity=0.7,
                nbins=50,
                barmode='overlay'
            )
            
            fig.update_layout(
                width=1000,
                height=500,
                legend_title_text='Label',
                legend=dict(
                    title_font_size=12,
                    font_size=10
                ),
                xaxis_title=col,
                yaxis_title='Count',
                template='plotly_white'
            )
            
            # Update legend labels
            fig.for_each_trace(lambda t: t.update(name='Fake' if t.name == 'fake' else 'Real'))
            
            st.plotly_chart(fig, use_container_width=True)

    plot_other_features()

    # Update sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This dashboard visualizes audio features from voice recordings:
        - **MFCCs**: Mel-frequency cepstral coefficients (1-13)
        - **Other features**: Spectral and temporal characteristics
        
        Colors indicate:
        - 🟢 Green: Real voices
        - 🔴 Red: Fake/synthetic voices
        """
    )

if __name__ == "__main__":
    show()