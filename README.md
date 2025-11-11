# Image Processing API

A FastAPI-based service for computer vision tasks including object detection and OCR processing.

## Features

- Object detection using YOLOv8
- Text extraction using Tesseract OCR
- Image processing pipeline
- RESTful API endpoints

## Requirements

- Python 3.9+
- Docker & Docker Compose (optional)

## Run The Project

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run
python3 main.py
```
## Install system dependencies
You need to install system dependencies for Tesseract OCR and tessdata, such as `libtesseract-dev`, `libleptonica-dev`, and `pkg-config`.

```bash
sudo apt-get install tesseract-ocr libtesseract-dev libleptonica-dev pkg-config libgl1-mesa-glx libglib2.0-0
```

### Using Docker (Optional)

```bash
# Build and run
docker-compose up --build
```

# Access API Docs
```bash
http://localhost:8000/docs/
```