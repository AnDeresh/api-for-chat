from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import json
import uvicorn

from parser import DialogParser
from train import ModelTrainer

app = FastAPI(
    title="LLM Reincarnation Service",
    description="Microservice for fine-tuning LLM model on personal messages",
    version="1.0.0"
)

dialog_parser = DialogParser()
model_trainer = ModelTrainer()

# Response models
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


def validate_speaker(speaker: str):
    """Validate speaker parameter."""
    if speaker not in ["User1", "User2"]:
        raise HTTPException(400, "Parameter 'speaker' must be 'User1' or 'User2'")


@app.post("/upload-text", response_model=UploadTextResponse)
async def upload_text(file: UploadFile = File(...), speaker: str = "User1"):
    """Upload dialog JSON and filter messages by speaker."""
    validate_speaker(speaker)
    
    if not file.filename.endswith('.json'):
        raise HTTPException(400, "File must be a JSON file")
    
    try:
        content = await file.read()
        dialog_data = json.loads(content.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON format: {str(e)}")
    except UnicodeDecodeError:
        raise HTTPException(400, "File encoding error. Use UTF-8 encoded JSON")
    
    try:
        result = dialog_parser.process_dialog_file(dialog_data, speaker)
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    return UploadTextResponse(
        message=f"Successfully processed dialog for {speaker}",
        speaker=result["speaker"],
        total_dialog_messages=result["total_dialog_messages"],
        filtered_messages_count=result["filtered_messages_count"],
        saved_to=result["saved_to"]
    )


@app.post("/train", response_model=TrainResponse)
async def train_model(speaker: str = "User1"):
    """Train LLM model on filtered messages."""
    validate_speaker(speaker)
    
    try:
        messages = dialog_parser.load_filtered_messages(speaker)
    except FileNotFoundError:
        raise HTTPException(404, f"No messages found for {speaker}. Upload dialog first")
    
    if len(messages) < 2:
        raise HTTPException(400, f"Need at least 2 messages, found {len(messages)}")
    
    try:
        result = model_trainer.train_model(messages, speaker)
    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")
    
    return TrainResponse(
        message=f"Successfully trained model for {speaker}",
        speaker=speaker,
        model_saved_to=result["model_path"],
        training_messages_count=result["training_messages"]
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)