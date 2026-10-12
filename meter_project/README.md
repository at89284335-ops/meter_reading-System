# Meter Reading Extraction System

## Overview
A Python and Django-based system to extract numerical readings from meter images using image preprocessing and OCR.

## Features
- Meter image preprocessing
- Image cropping and quality enhancement
- OCR-based meter reading extraction
- Reading validation
- REST API for a single image
- Batch processing of multiple images

## Tech Stack
- Python
- Django REST Framework
- OpenCV
- Tesseract OCR
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Open the project folder:
   ```bash
   cd meter_project
   ```

3. Activate your virtual environment on Windows:
   ```cmd
   .venv\Scripts\activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the Django server:
   ```bash
   python manage.py runserver
   ```

## API Endpoints

### Single Meter Image
- **Method:** POST
- **Endpoint:** `/api/v1/meter/read`
- **Request:** `multipart/form-data`
- **Field:** `image` (file)

### Batch Meter Images
- **Method:** POST
- **Endpoint:** `/api/v1/meter/read/batch`
- **Request:** `multipart/form-data`
- **Field:** `images` (multiple files)

## API Documentation
If Swagger is configured, open:
`http://127.0.0.1:8000/api/docs/`

## Notes
- Tesseract OCR must be installed and configured on the system.
- OCR results may vary depending on image quality and meter type.
- Check the API response to determine whether a reading was successfully extracted and validated.