import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
from datetime import datetime
import tensorflow as tf

def load_and_prep_image(img):
    """Prepare image for the model"""
    # Convert to RGB if not already
    img = img.convert("RGB")
    
    # Resize and preprocess
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    
    # Normalize the image
    normalized_img = (img_array.astype(np.float32) / 127.5) - 1
    
    # Add batch dimension
    return np.expand_dims(normalized_img, axis=0)

def get_condition_message(condition):
    """Get message and color based on condition"""
    if condition == "Clean":
        return "✅ Panel is clean and functioning normally", "success"
    elif condition == "Physical Damage":
        return "⚠️ Physical damage detected - Maintenance required", "warning"
    else:  # Electrical Damage
        return "🚨 Electrical damage detected - Immediate attention needed", "error"

def main():
    st.set_page_config(layout='wide', page_title="Solar Panel Fault Detection")
    
    st.title("Solar Panel Fault Detection System")
    
    # Analysis mode selection
    analysis_mode = st.radio("Choose Analysis Mode:", ["Single Image", "Multiple Images"])

    try:
        interpreter = tf.lite.Interpreter(model_path="keras_model.tflite")
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Load labels
        with open("labels.txt", "r") as f:
            labels = f.read().splitlines()
        
        if analysis_mode == "Single Image":
            uploaded_file = st.file_uploader("Upload a solar panel image", type=['jpg', 'png', 'jpeg'])
            
            if uploaded_file is not None:
                # Display image and analyze
                col1, col2 = st.columns(2)
                
                with col1:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("Analyze Panel"):
                    with col2:
                        with st.spinner("Analyzing panel condition..."):
                            # Prepare image
                            processed_img = load_and_prep_image(image)
                            
                            # Set tensor and run inference
                            interpreter.set_tensor(input_details[0]['index'], processed_img)
                            interpreter.invoke()
                            
                            # Get results
                            output_data = interpreter.get_tensor(output_details[0]['index'])
                            prediction_idx = np.argmax(output_data)
                            confidence = output_data[0][prediction_idx]
                            
                            # Get predicted condition
                            condition = labels[prediction_idx].split(' ')[1].strip()
                            message, status_type = get_condition_message(condition)
                            
                            # Display results
                            getattr(st, status_type)(message)
                            st.metric("Confidence Score", f"{confidence * 100:.1f}%")
                            
                            # Additional details
                            st.subheader("Analysis Details")
                            st.write(f"Detected Condition: {condition}")
                            st.progress(float(confidence))
        
        else:  # Multiple Images
            uploaded_files = st.file_uploader("Upload multiple solar panel images", 
                                            type=['jpg', 'png', 'jpeg'], 
                                            accept_multiple_files=True)
            
            if uploaded_files:
                if st.button("Analyze All Panels"):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, file in enumerate(uploaded_files):
                        status_text.text(f"Processing image {idx + 1} of {len(uploaded_files)}")
                        
                        try:
                            # Process image
                            image = Image.open(file)
                            processed_img = load_and_prep_image(image)
                            
                            # Run inference
                            interpreter.set_tensor(input_details[0]['index'], processed_img)
                            interpreter.invoke()
                            
                            # Get results
                            output_data = interpreter.get_tensor(output_details[0]['index'])
                            prediction_idx = np.argmax(output_data)
                            confidence = output_data[0][prediction_idx]
                            condition = labels[prediction_idx].split(' ')[1].strip()
                            
                            results.append({
                                'Image': file.name,
                                'Condition': condition,
                                'Confidence': f"{confidence:.2%}"
                            })
                            
                        except Exception as e:
                            results.append({
                                'Image': file.name,
                                'Condition': 'Error',
                                'Confidence': str(e)
                            })
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    status_text.text("Analysis Complete!")
                    
                    # Display results
                    import pandas as pd
                    df = pd.DataFrame(results)
                    
                    col1, col2, col3 = st.columns(3)
                    total_panels = len(df)
                    problem_panels = len(df[df['Condition'].isin(['Physical Damage', 'Electrical Damage'])])
                    
                    col1.metric("Total Panels", total_panels)
                    col2.metric("Panels Needing Attention", problem_panels)
                    col3.metric("Clean Panels", total_panels - problem_panels)
                    
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "solar_panel_analysis.csv",
                        "text/csv"
                    )
                    
    except Exception as e:
        st.error(f"Error loading model or labels: {str(e)}")
        st.info("Please ensure 'keras_model.tflite' and 'labels.txt' are present in the app directory.")

if __name__ == "__main__":
    main()