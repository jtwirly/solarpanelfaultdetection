# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pandas as pd

# Set page config first
st.set_page_config(layout='wide', page_title="Solar Panel Fault Detection")

def load_model():
    """Load the TFLite model"""
    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

def process_image(image):
    """Process image to match Teachable Machine's requirements"""
    # Resize the image to match what Teachable Machine expects
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert image to numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Create the array of the right shape
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    return data

def predict(interpreter, image_data):
    """Run prediction on processed image data"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])[0]

def format_condition(condition):
    """Format the condition text for display"""
    if condition == "Physical":
        return "Physical Damage"
    elif condition == "Electrical":
        return "Electrical Damage"
    return condition

def main():
    st.title("Solar Panel Fault Detection")
    
    try:
        # Load model and labels
        interpreter = load_model()
        with open("labels.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        
        # Create tabs for different modes
        tab1, tab2 = st.tabs(["Single Image Analysis", "Batch Analysis"])
        
        with tab1:
            uploaded_file = st.file_uploader("Upload a solar panel image", type=['jpg', 'png', 'jpeg'])
            
            if uploaded_file:
                # Display image and analysis side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    if st.button("Analyze Image"):
                        with st.spinner("Analyzing..."):
                            # Process and predict
                            processed_image = process_image(image)
                            predictions = predict(interpreter, processed_image)
                            
                            # Get the results
                            class_index = np.argmax(predictions)
                            confidence = predictions[class_index]
                            condition = labels[class_index].split(' ')[1]  # Get condition after the number
                            condition_display = format_condition(condition)
                            
                            # Display results with appropriate styling
                            if condition == "Clean":
                                st.success(f"✅ Panel Status: {condition_display}")
                            elif condition == "Physical":
                                st.warning(f"⚠️ Panel Status: {condition_display}")
                            else:
                                st.error(f"🚨 Panel Status: {condition_display}")
                            
                            # Show confidence score
                            st.metric("Confidence", f"{confidence * 100:.1f}%")
                            
                            # Show confidence bars for all classes
                            st.subheader("Detailed Analysis")
                            for idx, score in enumerate(predictions):
                                label = labels[idx].split(' ')[1]  # Get condition after the number
                                label_display = format_condition(label)
                                st.write(f"{label_display}: {score * 100:.1f}%")
                                st.progress(float(score))
        
        with tab2:
            uploaded_files = st.file_uploader("Upload multiple images", 
                                            type=['jpg', 'png', 'jpeg'],
                                            accept_multiple_files=True)
            
            if uploaded_files:
                if st.button("Analyze All Images"):
                    results = []
                    progress = st.progress(0)
                    status = st.empty()
                    
                    for idx, file in enumerate(uploaded_files):
                        status.text(f"Processing image {idx + 1} of {len(uploaded_files)}")
                        
                        try:
                            # Process image
                            image = Image.open(file).convert('RGB')
                            processed_image = process_image(image)
                            predictions = predict(interpreter, processed_image)
                            
                            # Get prediction
                            class_index = np.argmax(predictions)
                            confidence = predictions[class_index]
                            condition = labels[class_index].split(' ')[1]
                            condition_display = format_condition(condition)
                            
                            results.append({
                                'Image': file.name,
                                'Condition': condition_display,
                                'Confidence': f"{confidence:.1%}"
                            })
                            
                        except Exception as e:
                            results.append({
                                'Image': file.name,
                                'Condition': 'Error',
                                'Confidence': str(e)
                            })
                        
                        progress.progress((idx + 1) / len(uploaded_files))
                    
                    status.text("Analysis Complete!")
                    
                    # Show summary
                    df = pd.DataFrame(results)
                    
                    col1, col2, col3 = st.columns(3)
                    total = len(df)
                    problems = len(df[df['Condition'].isin(['Physical Damage', 'Electrical Damage'])])
                    
                    col1.metric("Total Panels", total)
                    col2.metric("Issues Found", problems)
                    col3.metric("Clean Panels", total - problems)
                    
                    # Show detailed results
                    st.subheader("Detailed Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results (CSV)",
                        csv,
                        "solar_panel_analysis.csv",
                        "text/csv"
                    )
    
    except FileNotFoundError:
        st.error("Model or labels file not found!")
        st.info("Please ensure 'converted_model.tflite' and 'labels.txt' are in the app directory")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()