from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st 
from dotenv import load_dotenv 
import os
import openai
from datetime import datetime
import pandas as pd

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def classify_solar_panel(img):
    """Classify a single solar panel image using the Teachable Machine model."""
    np.set_printoptions(suppress=True)
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def generate_analysis(label):
    """Generate maintenance recommendations based on the condition."""
    condition = label.split(' ')[1].strip()
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Provide maintenance recommendations for a solar panel showing {condition} condition. Include potential causes and suggested actions.\n",
        temperature=0.7,
        max_tokens=200
    )
    return response['choices'][0]['text']

st.set_page_config(layout='wide')
st.title("Solar Panel Fault Detection System")

# Add option to choose between single and batch upload
analysis_mode = st.radio("Choose Analysis Mode:", ["Single Image", "Multiple Images"])

if analysis_mode == "Single Image":
    # Original single image analysis
    input_img = st.file_uploader("Upload a solar panel image", type=['jpg', 'png', 'jpeg'], key="single")
    
    if input_img:
        if st.button("Analyze Panel"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(input_img, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                with st.spinner("Analyzing panel condition..."):
                    image_file = Image.open(input_img)
                    label, confidence_score = classify_solar_panel(image_file)
                    condition = label.split(' ')[1].strip()
                    
                    if condition == "Clean":
                        st.success(f"Panel Status: CLEAN (Confidence: {confidence_score:.2%})")
                    elif condition == "Physical Damage":
                        st.warning(f"Panel Status: PHYSICAL DAMAGE (Confidence: {confidence_score:.2%})")
                    else:  # Electrical Damage
                        st.error(f"Panel Status: ELECTRICAL DAMAGE (Confidence: {confidence_score:.2%})")
                    
                    analysis = generate_analysis(label)
                    st.subheader("Recommendations:")
                    st.write(analysis)

else:
    # Batch image analysis
    uploaded_files = st.file_uploader("Upload multiple solar panel images", 
                                    type=['jpg', 'png', 'jpeg'], 
                                    accept_multiple_files=True,
                                    key="multiple")
    
    if uploaded_files:
        if st.button("Analyze All Panels"):
            results = []
            
            # Progress bar for batch processing
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing image {idx + 1} of {len(uploaded_files)}")
                
                try:
                    image = Image.open(uploaded_file)
                    label, confidence = classify_solar_panel(image)
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
            
            status_text.text("Analysis Complete!")
            
            # Display results
            df = pd.DataFrame(results)
            
            # Summary statistics
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            total_panels = len(df)
            problem_panels = len(df[df['Condition'].isin(['Physical Damage', 'Electrical Damage'])])
            
            col1.metric("Total Panels", total_panels)
            col2.metric("Panels Needing Attention", problem_panels)
            col3.metric("Clean Panels", total_panels - problem_panels)
            
            # Detailed results
            st.subheader("Detailed Results")
            st.dataframe(df)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Results CSV",
                csv,
                "solar_panel_analysis.csv",
                "text/csv",
                key='download-csv'
            )