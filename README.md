# Solar Panel Fault Detection System

## Overview
This project is a machine learning-powered web application that detects and classifies faults in solar panels using computer vision. The system can analyze both single images and batch process multiple images to identify three conditions:
- Clean panels
- Physical damage
- Electrical damage

This project was partly inspired by [AIAnytime's Waste Classifier project](https://www.youtube.com/watch?v=s3e2JJxvwPM) and adapted its initial codebase from their [waste classification system](https://github.com/AIAnytime/Waste-Classifier-Sustainability-App).

## Features
- Single image analysis with detailed confidence scores
- Batch processing capability for multiple panels
- Real-time analysis with progress tracking
- Downloadable results in CSV format
- User-friendly interface built with Streamlit
- High accuracy fault detection using TensorFlow Lite

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
- Transfer learning on a MobileNet architecture

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

Your app will be available at: https://[your-username]-solarpanelfaultdetection-streamlit-app.streamlit.app

Note: The repository already includes:
- `converted_model.tflite` (TensorFlow Lite model)
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
- Accurate detection of physical damage patterns
- Reliable identification of electrical faults
- Fast inference time suitable for real-time analysis

## Future Enhancements
- Integration with drone imagery systems
- Time series analysis for degradation tracking
- Automated report generation
- Geographic mapping of panel conditions
- Mobile app development

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT

## Acknowledgments
- Original inspiration: [AIAnytime's Waste Classifier](https://github.com/AIAnytime/Waste-Classifier-Sustainability-App)
- Thanks to Google Teachable Machine for making ML model training accessible
