# Solar Panel Fault Detection with Machine Learning (Image Classification)

## Overview
This project is an example from the **AI for Energy Solutions** course at MIT's Independent Activities Period (IAP) that I am teaching. It showcases how **machine learning** and **image classification** can be applied to energy infrastructure monitoring. 

The system is a web application that detects and classifies faults in solar panels, capable of analyzing:
- **Clean panels**
- **Physical damage**
- **Electrical damage**

The project is designed to be accessible, beginning with a **no-code approach** using Google Teachable Machine, and optionally extending to **coding and deployment** for students looking to delve deeper into development.

<img width="1426" alt="Screen Shot 2025-01-10 at 9 48 39 PM" src="https://github.com/user-attachments/assets/9b0a3885-ad92-4a50-b66a-51cba7113530" />
<img width="1421" alt="Screen Shot 2025-01-10 at 10 11 50 PM" src="https://github.com/user-attachments/assets/40a7cadd-eed3-4bd2-b5ba-be299b2cac79" />
<img width="1434" alt="Screen Shot 2025-01-10 at 9 49 29 PM" src="https://github.com/user-attachments/assets/bd7e334c-c1f6-4ae2-88d1-2bd918689df5" />

---

## Project Highlights

### Beginner-Friendly: No-Code Model Training
- **Google Teachable Machine** makes training an image classification model simple and intuitive. 
- No prior coding knowledge is required.
- Learn to create a powerful AI model with just a few clicks.

### Optional Advanced: Coding & Deployment
For students interested in exploring coding and deployment:
- Use **Python** to process data and deploy a real-world application.
- Build a web app with **Streamlit** to make the model user-friendly.
- Enhance functionality with batch processing, CSV downloads, and more.

---

## What You'll Learn
This project teaches students to:
1. **No-Code Training**:
   - Train an image classifier using Google Teachable Machine.
2. **Optional Coding Section**:
   - Deploy the trained model in a Python-based web app.
   - Use Streamlit for a professional, user-friendly interface.
3. **Practical Applications**:
   - Analyze single images or batches for energy monitoring tasks.
   - Expand capabilities to other types of energy infrastructure.

---

## Features
- **No-Code**: Train a model without programming.
- **Single & Batch Processing**: Analyze one or many images.
- **Real-Time Results**: Confidence scores for multiple classifications.
- **Web Interface**: Interactive and accessible, powered by Streamlit.
- **Deployment Ready**: Preconfigured for easy deployment.

---

## Future Directions
- Integrating drone imagery for automated inspections.
- Expanding to other renewable energy systems (e.g., wind turbines).
- Mobile app development for field use.

--- 

## Getting Started

1. **Prepare Your Dataset**
   - To follow along with this tutorial, you can download images from this Kaggle solar panel dataset: https://www.kaggle.com/datasets/pythonafroz/solar-panel-images. Unzip the files.
OR
   - Collect ~50+ images per category. You could find a dataset through a website like Kaggle or create your own dataset. 
   - Ensure diverse conditions (lighting, angles, distances)
   - Include both positive and negative examples
   - Label your images consistently

3. **Train Using Google Teachable Machine**
   - Visit [Teachable Machine](https://teachablemachine.withgoogle.com/)
   - Create a new Image Project
   - Upload your images for each class and name each class

<img width="1418" alt="Screen Shot 2025-01-10 at 8 16 43 PM" src="https://github.com/user-attachments/assets/8f467d39-5507-4923-b2c7-75a74ee81a52" />

   - Set training parameters:
     ```
     Epochs: 50 (adjust based on performance)
     Batch Size: 16
     Learning Rate: 0.001
     ```
<img width="1420" alt="Screen Shot 2025-01-10 at 8 31 25 PM" src="https://github.com/user-attachments/assets/5318919e-b5ad-454c-9732-bbdaabd654ab" />

   - Train the model
   - Test it with other images
   - Export it to the web by clicking Export, then Upload. Feel free to test the web link and share it with your friends. 
   - Congratulations, you have created an image classification machine learning model project! 
<img width="879" alt="Screen Shot 2025-01-15 at 9 42 50 AM" src="https://github.com/user-attachments/assets/c66fa4ac-675c-4528-8ad1-7a9ebfa5d39a" />


### 2. (Optional) Code & Deploy
Export the model as TensorFlow Lite.

<img width="1404" alt="Screen Shot 2025-01-10 at 9 36 39 PM" src="https://github.com/user-attachments/assets/e67545cf-8365-4897-b183-81f5eb34e70b" />

Follow the instructions in the repository below to:
- Build a Python-based app using Streamlit.
- Customize features for advanced use cases.

---

## Overview of Project - with Code
This project was created as an example project for the AI for Energy Solutions course that I am teaching during MIT's Independent Activities Period (IAP). It demonstrates how machine learning and image classification can be applied to real-world energy infrastructure monitoring.

The system is a machine learning-powered web application that detects and classifies faults in solar panels using transfer learning and image classification. It can analyze both single images and batch process multiple images to identify three conditions:
- Clean panels
- Physical damage
- Electrical damage

[Live Demo](https://solarpanelfaultdetection.streamlit.app)

This project was partly inspired by [AIAnytime's Waste Classifier project](https://www.youtube.com/watch?v=s3e2JJxvwPM) and adapted its initial codebase from their [waste classification system](https://github.com/AIAnytime/Waste-Classifier-Sustainability-App).

This implementation shows students how to:
- Train a custom image classification model using Google Teachable Machine
- Deploy a machine learning model in a user-friendly web interface
- Process both single and batch images for real-world applications
- Handle multiple classification categories with confidence scoring

## Features
- Single image analysis with detailed confidence scores
- Batch processing capability for multiple panels
- Real-time analysis with progress tracking
- Downloadable results in CSV format
- User-friendly interface built with Streamlit
- High accuracy classification using TensorFlow Lite

## Tech Stack
- Python
- TensorFlow Lite
- Streamlit
- Google Teachable Machine
- PIL (Python Imaging Library)
- Pandas
- NumPy

## Model Training
The model was trained using Google Teachable Machine with:
- 50+ images for each condition
- Three classification categories
- Image augmentation for better generalization
- Transfer learning using MobileNet architecture

## Setup and Installation

### Local Development
1. Clone the repository:
```bash
git clone https://github.com/jtwirly/solarpanelfaultdetection/
cd solarpanelfaultdetection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application locally:
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Fork the repository to your GitHub account

2. Visit [Streamlit Share](https://share.streamlit.io/) and sign in with your GitHub account

3. Click "New app" and select:
   - Repository: solarpanelfaultdetection
   - Branch: main
   - Main file path: app.py

4. Click "Deploy"

Your app will be available at: https://[your-username]-solarpanelfaultdetection-streamlit-app.streamlit.app or whatever you choose to set the URL as.

Note: The repository already includes:
- `converted_model.tflite` (TensorFlow Lite model) - or use your own (exported from Teachable Machine)
- `labels.txt` (Class labels)
- `requirements.txt` (Dependencies)

No additional configuration is needed for deployment.

## Usage

### Single Image Analysis
1. Select the "Single Image Analysis" tab
2. Upload a solar panel image
3. Click "Analyze Image"
4. View the results showing:
   - Panel condition
   - Confidence score
   - Detailed probability breakdown

### Batch Analysis
1. Select the "Batch Analysis" tab
2. Upload multiple solar panel images
3. Click "Analyze All Images"
4. Monitor the progress bar
5. View summary statistics and detailed results
6. Download results as CSV if needed

## Model Performance
The current model achieves:
- High confidence scores for clean panels (typically >90%)
- Accurate classification of physical damage patterns
- Reliable identification of electrical faults
- Fast inference time suitable for real-time analysis

## Future Enhancements
- Integration with drone imagery systems
- Time series analysis for degradation tracking
- Automated report generation
- Geographic mapping of panel conditions
- Mobile app development

## Customizing the Project
This project can be adapted for various energy infrastructure monitoring applications. Check `otherideas.txt` for a comprehensive list of potential adaptations.
You can also modify the Streamlit code to suit your needs, using Claude or otherwise to generate the code or writing it yourself. 

3. **Modify the Code**
   - Update `labels.txt` with your categories:
     ```
     0 YourClass1
     1 YourClass2
     2 YourClass3
     ```
   - Rename your model to `converted_model.tflite`
   - Adjust the display messages in `app.py`:
     ```python
     def format_condition(condition):
         """Format condition text for display"""
         if condition == "YourClass1":
             return "Your Display Text 1"
         elif condition == "YourClass2":
             return "Your Display Text 2"
         return condition
     ```

4. **Customize the Interface**
   - Modify the code in app.py (you can use Claude or ChatGPT to help if desired)
   - Modify the title and descriptions
   - Adjust confidence thresholds if needed
   - Add custom metrics relevant to your use case
   - Update the results display format

### Best Practices for Adaptation
1. **Data Collection**
   - Use real-world examples
   - Include various environmental conditions
   - Consider seasonal variations
   - Document image sources and conditions

2. **Model Training**
   - Start with default parameters
   - Experiment with learning rates
   - Monitor for overfitting
   - Validate with test images

3. **Testing & Validation**
   - Use a separate test dataset
   - Verify performance across all categories
   - Test edge cases
   - Document accuracy metrics

4. **Deployment Considerations**
   - Update requirements.txt if needed
   - Test thoroughly before deployment
   - Consider scalability
   - Document any special requirements

### Example Adaptation (Wind Turbine)
```python
# Example labels.txt for wind turbine inspection
0 Normal
1 BladeDamage
2 IceAccumulation
3 Corrosion

# Example condition formatting
def format_condition(condition):
    conditions = {
        "Normal": "✅ Normal Operation",
        "BladeDamage": "🚨 Blade Damage Detected",
        "IceAccumulation": "❄️ Ice Accumulation",
        "Corrosion": "⚠️ Corrosion Detected"
    }
    return conditions.get(condition, condition)
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT

## Acknowledgments
- Original inspiration: [AIAnytime's Waste Classifier](https://github.com/AIAnytime/Waste-Classifier-Sustainability-App)
- Thanks to Google Teachable Machine for making ML model training accessible
