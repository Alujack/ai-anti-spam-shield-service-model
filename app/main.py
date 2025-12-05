from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from datetime import datetime
import os
import speech_recognition as sr
import tempfile
import io

# Import predictor
from model.predictor import SpamPredictor

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
    print("✅ Spam predictor initialized successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize predictor: {e}")
    print("Please train the model first using: python model/train.py")
    predictor = None

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
