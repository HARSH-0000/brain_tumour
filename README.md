# brain_tumour
# MRI Brain Tumor Detection System

A web-based application that utilizes deep learning (TensorFlow/Keras) to classify brain MRI scans. It is capable of detecting and classifying tumors into three distinct categories, or detecting the absence of a tumor.

## Overview

This project consists of a Flask backend serving an HTML interface. Users can upload an MRI scan image, which is processed by a pre-trained Convolutional Neural Network (CNN) model. The system provides a prediction along with a confidence score.

### Classes Detected:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

## Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** TensorFlow, Keras, NumPy
- **Frontend:** HTML, Bootstrap CSS

## Local Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repository-url>
   cd brain_tumour
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To run the Flask application locally:

```bash
python main.py
```

The application will start, typically on `http://127.0.0.1:5000`. Keep in mind that depending on your specific script, it may serve on a different port. Visit this address in your web browser to access the upload form.

