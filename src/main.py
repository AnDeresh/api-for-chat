from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import json
from typing import Dict, Any
import uvicorn

from parser import DialogParser
from train import ModelTrainer

# Initialize FastAPI app
app = FastAPI(
    title="LLM Reincarnation Service",
    description="Microservice for fine-tuning LLM model on personal messages",
    version="1.0.0"
)

# Initialize components
dialog_parser = DialogParser()
model_trainer = ModelTrainer()

# Pydantic models for request/response
class UploadTextRequest(BaseModel):
    speaker: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "speaker": "User1"
            }
        }

class TrainRequest(BaseModel):
    speaker: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "speaker": "User1"
            }
        }

class UploadTextResponse(BaseModel):
    message: str
    speaker: str
    total_dialog_messages: int
    filtered_messages_count: int
    saved_to: str

class TrainResponse(BaseModel):
    message: str
    speaker: str
    model_saved_to: str
    training_messages_count: int


@app.post("/upload-text", response_model=UploadTextResponse)
async def upload_text(
    file: UploadFile = File(..., description="JSON file with dialog"),
    speaker: str = "User1"
):
    """
    Upload dialog JSON file and filter messages by specified speaker.
    
    Args:
        file: JSON file with dialog structure
        speaker: Speaker to filter messages for ("User1" or "User2")
        
    Returns:
        Information about processed dialog
    """
    try:
        # Validate speaker parameter
        if speaker not in ["User1", "User2"]:
            raise HTTPException(
                status_code=400, 
                detail="Parameter 'speaker' must be 'User1' or 'User2'"
            )
        
        # Check file type
        if not file.filename.endswith('.json'):
            raise HTTPException(
                status_code=400,
                detail="File must be a JSON file"
            )
        
        # Read and parse file content
        try:
            content = await file.read()
            dialog_data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON format: {str(e)}"
            )
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File encoding error. Please use UTF-8 encoded JSON file"
            )
        
        # Process dialog
        try:
            result = dialog_parser.process_dialog_file(dialog_data, speaker)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        return UploadTextResponse(
            message=f"Successfully processed dialog and filtered messages for {speaker}",
            speaker=result["speaker"],
            total_dialog_messages=result["total_dialog_messages"],
            filtered_messages_count=result["filtered_messages_count"],
            saved_to=result["saved_to"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/train", response_model=TrainResponse)
async def train_model(speaker: str = "User1"):
    """
    Train LLM model on filtered messages for specified speaker.
    
    Args:
        speaker: Speaker to train model for ("User1" or "User2")
        
    Returns:
        Information about training result
    """
    try:
        # Validate speaker parameter
        if speaker not in ["User1", "User2"]:
            raise HTTPException(
                status_code=400,
                detail="Parameter 'speaker' must be 'User1' or 'User2'"
            )
        
        # Check if filtered messages exist
        try:
            messages = dialog_parser.load_filtered_messages(speaker)
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail=f"No messages found for {speaker}. Please upload dialog first via /upload-text"
            )
        
        # Validate minimum messages count
        if len(messages) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 2 messages for training, but found {len(messages)} for {speaker}"
            )
        
        # Start training
        try:
            result = model_trainer.train_model(messages, speaker)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Training failed: {str(e)}"
            )
        
        return TrainResponse(
            message=f"Successfully trained model for {speaker}",
            speaker=speaker,
            model_saved_to=result["model_path"],
            training_messages_count=result["training_messages"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )