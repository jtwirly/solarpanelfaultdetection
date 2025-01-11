import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
from datetime import datetime
import tensorflow as tf
import pandas as pd

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None
    
def load_model_once():
    """Load the model once and store in session state."""
    if st.session_state.model is None:
        try:
            # Custom error handling for model loading
            model_path = "keras_model.h5"
            if not os.path.exists(model_path):
                st.error(f"Model file not found at {model_path}. Please ensure the model file is in the correct location.")
                return None
                
            st.session_state.model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    return st.session_state.model

def classify_solar_panel(img):
    """Classify a single solar panel image."""
    try:
        # Load model
        model = load_model_once()
        if model is None:
            return None, None
            
        # Load labels
        labels_path = "labels.txt"
        if not os.path.exists(labels_path):
            st.error(f"Labels file not found at {labels_path}")
            return None, None
            
        class_names = open(labels_path, "r").readlines()
        
        # Prepare image
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Predict
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        return class_name, confidence_score
        
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        return None, None

def get_condition_color(condition):
    """Return appropriate color for each condition."""
    if condition == "Clean":
        return "green"
    elif condition == "Physical Damage":
        return "orange"
    elif condition == "Electrical Damage":
        return "red"
    return "gray"

def main():
    st.set_page_config(layout='wide', page_title="Solar Panel Fault Detection")
    
    st.title("Solar Panel Fault Detection System")
    st.write("Upload solar panel images for fault detection")
    
    # Analysis mode selection
    analysis_mode = st.radio("Choose Analysis Mode:", ["Single Image", "Multiple Images"])

    if analysis_mode == "Single Image":
        input_img = st.file_uploader("Upload a solar panel image", type=['jpg', 'png', 'jpeg'], key="single")
        
        if input_img is not None:
            try:
                image = Image.open(input_img).convert("RGB")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("Analyze Panel"):
                    with col2:
                        with st.spinner("Analyzing..."):
                            label, confidence = classify_solar_panel(image)
                            
                            if label is not None:
                                condition = label.split(' ')[1].strip()
                                color = get_condition_color(condition)
                                
                                st.markdown(f"""
                                <div style='padding: 20px; border-radius: 10px; background-color: {color}; color: white;'>
                                    <h3>Analysis Results</h3>
                                    <p>Condition: {condition}</p>
                                    <p>Confidence: {confidence:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    else:  # Multiple Images
        uploaded_files = st.file_uploader("Upload multiple solar panel images", 
                                        type=['jpg', 'png', 'jpeg'], 
                                        accept_multiple_files=True,
                                        key="multiple")
        
        if uploaded_files:
            if st.button("Analyze All Panels"):
                results = []
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        image = Image.open(uploaded_file).convert("RGB")
                        label, confidence = classify_solar_panel(image)
                        
                        if label is not None:
                            condition = label.split(' ')[1].strip()
                            results.append({
                                'Image': uploaded_file.name,
                                'Condition': condition,
                                'Confidence': f"{confidence:.2%}"
                            })
                    except Exception as e:
                        results.append({
                            'Image': uploaded_file.name,
                            'Condition': 'Error',
                            'Confidence': str(e)
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                if results:
                    df = pd.DataFrame(results)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    total_panels = len(df)
                    problem_panels = len(df[df['Condition'].isin(['Physical Damage', 'Electrical Damage'])])
                    
                    col1.metric("Total Panels", total_panels)
                    col2.metric("Panels Needing Attention", problem_panels)
                    col3.metric("Clean Panels", total_panels - problem_panels)
                    
                    # Results table
                    st.dataframe(df)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "solar_panel_analysis.csv",
                        "text/csv"
                    )

if __name__ == "__main__":
    main()