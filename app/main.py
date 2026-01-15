from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import uvicorn
from datetime import datetime
import os
import speech_recognition as sr
import tempfile
import io
import logging

# Import predictor
from model.predictor import SpamPredictor

# Import phishing detector
try:
    from detectors.phishing_detector import PhishingDetector
    HAS_PHISHING_DETECTOR = True
except ImportError:
    HAS_PHISHING_DETECTOR = False
    logging.warning("PhishingDetector not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Anti-Spam Shield API",
    description="Spam detection service using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
try:
    predictor = SpamPredictor(model_dir='model')
    print("Spam predictor initialized successfully!")
except Exception as e:
    print(f"Warning: Could not initialize predictor: {e}")
    print("Please train the model first using: python model/train.py")
    predictor = None

# Initialize phishing detector
phishing_detector = None
if HAS_PHISHING_DETECTOR:
    try:
        phishing_detector = PhishingDetector(model_dir='models')
        print("Phishing detector initialized successfully!")
    except Exception as e:
        print(f"Warning: Could not initialize phishing detector: {e}")
        phishing_detector = None

# Request/Response Models


class PredictionRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000,
                         description="Text message to analyze")


class BatchPredictionRequest(BaseModel):
    messages: List[str] = Field(..., min_items=1, max_items=100,
                                description="List of messages to analyze")


class PredictionResponse(BaseModel):
    is_spam: bool
    prediction: str
    confidence: float
    probability: float
    probabilities: dict
    details: Optional[dict] = None
    timestamp: str


class VoicePredictionResponse(BaseModel):
    transcribed_text: str
    is_spam: bool
    prediction: str
    confidence: float
    probability: float
    probabilities: dict
    details: Optional[dict] = None
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    version: str


# Phishing Detection Models
class PhishingRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000,
                      description="Text message or URL to analyze for phishing")
    scan_type: Literal['email', 'sms', 'url', 'auto'] = Field(
        default='auto',
        description="Type of scan: email, sms, url, or auto-detect"
    )


class URLScanRequest(BaseModel):
    url: str = Field(..., min_length=5, max_length=2000,
                     description="URL to analyze for phishing")


class BatchPhishingRequest(BaseModel):
    items: List[str] = Field(..., min_items=1, max_items=100,
                             description="List of texts/URLs to analyze")
    scan_type: Literal['email', 'sms', 'url', 'auto'] = Field(
        default='auto',
        description="Type of scan for all items"
    )


class URLAnalysisResponse(BaseModel):
    url: str
    is_suspicious: bool
    score: float
    reasons: List[str]


class BrandImpersonationResponse(BaseModel):
    detected: bool
    brand: Optional[str]
    similarity_score: float


class PhishingResponse(BaseModel):
    is_phishing: bool
    confidence: float
    phishing_type: str
    threat_level: str
    indicators: List[str]
    urls_analyzed: List[dict]
    brand_impersonation: Optional[dict]
    recommendation: str
    details: dict
    timestamp: str


# Routes


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AI Anti-Spam Shield API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_voice": "/predict-voice",
            "batch_predict": "/batch-predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if predictor is not None else "model_not_loaded",
        "model_loaded": predictor is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict if a message is spam or not

    - **message**: The text message to analyze

    Returns spam prediction with confidence score
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )

    try:
        result = predictor.predict(request.message)
        result['timestamp'] = datetime.utcnow().isoformat()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/batch-predict", tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict multiple messages at once

    - **messages**: List of text messages to analyze

    Returns list of predictions
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )

    try:
        results = predictor.batch_predict(request.messages)
        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.post("/predict-voice", response_model=VoicePredictionResponse, tags=["Prediction"])
async def predict_voice(audio: UploadFile = File(...)):
    """
    Predict if a voice message is spam or not

    - **audio**: Audio file (WAV, MP3, OGG, FLAC)

    Transcribes audio to text, then analyzes for spam
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )

    try:
        # Read audio file
        audio_data = await audio.read()

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1]) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        try:
            # Initialize speech recognizer
            recognizer = sr.Recognizer()

            # Load audio file
            with sr.AudioFile(temp_audio_path) as source:
                audio_content = recognizer.record(source)

            # Transcribe audio to text using Google Speech Recognition
            try:
                transcribed_text = recognizer.recognize_google(audio_content)
            except sr.UnknownValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not understand audio. Please speak clearly."
                )
            except sr.RequestError as e:
                # Fallback to basic transcription
                transcribed_text = "Audio transcription service unavailable"

            # Analyze transcribed text for spam
            result = predictor.predict(transcribed_text)
            result['transcribed_text'] = transcribed_text
            result['timestamp'] = datetime.utcnow().isoformat()

            return result

        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice prediction error: {str(e)}"
        )


@app.get("/stats", tags=["Statistics"])
async def get_stats():
    """Get model statistics"""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return {
        "model_type": predictor.model.__class__.__name__,
        "features_count": predictor.vectorizer.get_feature_names_out().shape[0] if hasattr(predictor.vectorizer, 'get_feature_names_out') else "N/A",
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================
# PHISHING DETECTION ENDPOINTS
# ============================================

@app.post("/predict-phishing", response_model=PhishingResponse, tags=["Phishing"])
async def predict_phishing(request: PhishingRequest):
    """
    Detect phishing in text messages or URLs

    - **text**: The text message or URL to analyze
    - **scan_type**: Type of content (email, sms, url, or auto-detect)

    Returns comprehensive phishing analysis with:
    - Phishing classification and confidence
    - Threat level (CRITICAL, HIGH, MEDIUM, LOW, NONE)
    - Detected indicators
    - URL analysis results
    - Brand impersonation detection
    - Actionable recommendations
    """
    if phishing_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phishing detector not available"
        )

    try:
        result = phishing_detector.detect(request.text, request.scan_type)
        response = result.to_dict()
        response['timestamp'] = datetime.utcnow().isoformat()
        return response
    except Exception as e:
        logger.error(f"Phishing detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Phishing detection error: {str(e)}"
        )


@app.post("/scan-url", tags=["Phishing"])
async def scan_url(request: URLScanRequest):
    """
    Scan a specific URL for phishing indicators

    - **url**: The URL to analyze

    Returns URL-specific analysis including:
    - Suspicious indicators
    - TLD analysis
    - Brand impersonation check
    - Obfuscation detection
    """
    if phishing_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phishing detector not available"
        )

    try:
        result = phishing_detector.detect(request.url, scan_type='url')
        response = result.to_dict()
        response['timestamp'] = datetime.utcnow().isoformat()
        return response
    except Exception as e:
        logger.error(f"URL scan error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URL scan error: {str(e)}"
        )


@app.post("/batch-phishing", tags=["Phishing"])
async def batch_phishing_scan(request: BatchPhishingRequest):
    """
    Scan multiple texts/URLs for phishing at once

    - **items**: List of texts or URLs to analyze (max 100)
    - **scan_type**: Type of content for all items

    Returns list of phishing analysis results
    """
    if phishing_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phishing detector not available"
        )

    try:
        results = []
        for item in request.items:
            result = phishing_detector.detect(item, request.scan_type)
            result_dict = result.to_dict()
            result_dict['input'] = item[:100] + '...' if len(item) > 100 else item
            results.append(result_dict)

        # Summary statistics
        phishing_count = sum(1 for r in results if r['is_phishing'])
        threat_levels = {}
        for r in results:
            level = r['threat_level']
            threat_levels[level] = threat_levels.get(level, 0) + 1

        return {
            "results": results,
            "summary": {
                "total": len(results),
                "phishing_detected": phishing_count,
                "safe": len(results) - phishing_count,
                "threat_levels": threat_levels
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Batch phishing scan error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch phishing scan error: {str(e)}"
        )


@app.get("/phishing-health", tags=["Phishing"])
async def phishing_health():
    """Check phishing detector health status"""
    return {
        "status": "healthy" if phishing_detector is not None else "unavailable",
        "detector_loaded": phishing_detector is not None,
        "ml_enabled": phishing_detector.ml_model is not None if phishing_detector else False,
        "transformer_enabled": phishing_detector.transformer_model is not None if phishing_detector else False,
        "timestamp": datetime.utcnow().isoformat()
    }


# Run server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
