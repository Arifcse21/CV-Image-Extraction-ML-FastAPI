from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import io
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Processing API",
    description="API for object detection (YOLO) and text extraction (OCR)",
    version="1.0.0"
)

try:
    model = YOLO('yolov8n.pt')
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    model = None

@app.get("/")
async def root():
    return {
        "message": "Image Processing API",
        "endpoints": {
            "/detect": "POST - Object detection using YOLO",
            "/ocr": "POST - Text extraction using OCR",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "yolo_loaded": model is not None,
        "ocr_available": True
    }

def validate_image(file: UploadFile) -> Image.Image:
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = file.file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
    
    try:
        image = Image.open(io.BytesIO(contents))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="YOLO model not available")
        
        image = validate_image(file)
        img_array = np.array(image)
        results = model(img_array, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bounding_box": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })
        
        logger.info(f"Detected {len(detections)} objects in image")
        
        return {
            "success": True,
            "image_size": {
                "width": image.width,
                "height": image.height
            },
            "detections_count": len(detections),
            "detections": detections
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/ocr")
async def extract_text(file: UploadFile = File(...), lang: str = "eng"):
    try:
        image = validate_image(file)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        ocr_data = pytesseract.image_to_data(
            thresh, 
            lang=lang, 
            output_type=pytesseract.Output.DICT
        )
        
        text_blocks = []
        full_text = []
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            
            if text:
                confidence = float(ocr_data['conf'][i])
                
                if confidence > 0:
                    text_blocks.append({
                        "text": text,
                        "confidence": round(confidence, 2),
                        "bounding_box": {
                            "x": ocr_data['left'][i],
                            "y": ocr_data['top'][i],
                            "width": ocr_data['width'][i],
                            "height": ocr_data['height'][i]
                        }
                    })
                    full_text.append(text)
        
        complete_text = pytesseract.image_to_string(thresh, lang=lang).strip()
        
        logger.info(f"Extracted {len(text_blocks)} text blocks from image")
        
        return {
            "success": True,
            "image_size": {
                "width": image.width,
                "height": image.height
            },
            "full_text": complete_text,
            "word_count": len(text_blocks),
            "text_blocks": text_blocks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR extraction failed: {str(e)}")

@app.post("/process")
async def process_image(file: UploadFile = File(...), mode: str = "both"):
    try:
        if mode not in ['detect', 'ocr', 'both']:
            raise HTTPException(status_code=400, detail="Mode must be 'detect', 'ocr', or 'both'")
        
        results = {}
        
        if mode in ['detect', 'both']:
            file.file.seek(0)
            detection_result = await detect_objects(file)
            results['detection'] = detection_result
        
        if mode in ['ocr', 'both']:
            file.file.seek(0)
            ocr_result = await extract_text(file)
            results['ocr'] = ocr_result
        
        return {
            "success": True,
            "mode": mode,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)