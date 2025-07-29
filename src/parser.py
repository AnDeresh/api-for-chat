import json
from typing import List, Dict, Any
from pathlib import Path


class DialogParser:
    """Process dialogs and filter messages by user."""
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir).resolve()  # Resolve absolute path
        self.data_dir.mkdir(exist_ok=True, parents=True)  # Create parent dirs too
    
    def parse_dialog(self, dialog_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Parse and validate dialog structure."""
        if "dialog" not in dialog_data:
            raise ValueError("Missing 'dialog' key in JSON file")
        
        dialog = dialog_data["dialog"]
        if not isinstance(dialog, list):
            raise ValueError("Field 'dialog' must be a list")
        
        # Validate messages
        for i, msg in enumerate(dialog):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            if not all(k in msg for k in ["speaker", "content"]):
                raise ValueError(f"Message {i} must contain 'speaker' and 'content'")
            if not all(isinstance(msg[k], str) for k in ["speaker", "content"]):
                raise ValueError(f"Message {i} 'speaker' and 'content' must be strings")
        
        return dialog
    
    def filter_messages_by_speaker(self, dialog: List[Dict[str, str]], speaker: str) -> List[str]:
        """Filter messages by speaker."""
        if speaker not in ["User1", "User2"]:
            raise ValueError("Parameter 'speaker' must be 'User1' or 'User2'")
        
        return [msg["content"] for msg in dialog if msg["speaker"] == speaker]
    
    def save_filtered_messages(self, messages: List[str], speaker: str) -> str:
        """Save filtered messages to JSON."""
        filepath = self.data_dir / f"{speaker}_messages.json"
        
        data = {
            "speaker": speaker,
            "messages": messages,
            "total_messages": len(messages)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return str(filepath)
    
    def load_filtered_messages(self, speaker: str) -> List[str]:
        """Load saved messages."""
        filepath = self.data_dir / f"{speaker}_messages.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found. Upload dialog via /upload-text first")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)["messages"]
    
    def process_dialog_file(self, dialog_data: Dict[str, Any], speaker: str) -> Dict[str, Any]:
        """Complete processing: parse, filter, save."""
        dialog = self.parse_dialog(dialog_data)
        filtered_messages = self.filter_messages_by_speaker(dialog, speaker)
        
        if not filtered_messages:
            raise ValueError(f"No messages found from user {speaker}")
        
        filepath = self.save_filtered_messages(filtered_messages, speaker)
        
        return {
            "speaker": speaker,
            "total_dialog_messages": len(dialog),
            "filtered_messages_count": len(filtered_messages),
            "saved_to": filepath
        }