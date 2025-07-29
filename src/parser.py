import json
import os
from typing import List, Dict, Any
from pathlib import Path


class DialogParser:
    """Class for processing dialogs and filtering messages by user."""
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize parser.
        
        Args:
            data_dir: Directory for saving filtered data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)  # Auto-create directory
    
    def parse_dialog(self, dialog_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Parse dialog from JSON structure.
        
        Args:
            dialog_data: Dictionary with 'dialog' key and list of messages
            
        Returns:
            List of messages
            
        Raises:
            ValueError: If data structure is incorrect
        """
        if "dialog" not in dialog_data:
            raise ValueError("Missing 'dialog' key in JSON file")
        
        dialog = dialog_data["dialog"]
        
        if not isinstance(dialog, list):
            raise ValueError("Field 'dialog' must be a list")
        
        # Validate message structure
        for i, message in enumerate(dialog):
            if not isinstance(message, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            
            if "speaker" not in message or "content" not in message:
                raise ValueError(f"Message {i} must contain 'speaker' and 'content'")
            
            if not isinstance(message["speaker"], str) or not isinstance(message["content"], str):
                raise ValueError(f"In message {i} 'speaker' and 'content' must be strings")
        
        return dialog
    
    def filter_messages_by_speaker(self, dialog: List[Dict[str, str]], speaker: str) -> List[str]:
        """
        Filter messages by specified user.
        
        Args:
            dialog: List of all messages
            speaker: User name for filtering
            
        Returns:
            List of messages from specified user
        """
        if speaker not in ["User1", "User2"]:
            raise ValueError("Parameter 'speaker' must be 'User1' or 'User2'")
        
        filtered_messages = []
        
        for message in dialog:
            if message["speaker"] == speaker:
                filtered_messages.append(message["content"])
        
        return filtered_messages
    
    def save_filtered_messages(self, messages: List[str], speaker: str) -> str:
        """
        Save filtered messages to JSON file.
        
        Args:
            messages: List of messages to save
            speaker: User name
            
        Returns:
            Path to saved file
        """
        filename = f"{speaker}_messages.json"
        filepath = self.data_dir / filename
        
        data_to_save = {
            "speaker": speaker,
            "messages": messages,
            "total_messages": len(messages)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        return str(filepath)
    
    def load_filtered_messages(self, speaker: str) -> List[str]:
        """
        Load saved user messages.
        
        Args:
            speaker: User name
            
        Returns:
            List of user messages
            
        Raises:
            FileNotFoundError: If file not found
        """
        filename = f"{speaker}_messages.json"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File {filename} not found. First upload dialog via /upload-text")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data["messages"]
    
    def process_dialog_file(self, dialog_data: Dict[str, Any], speaker: str) -> Dict[str, Any]:
        """
        Full dialog processing cycle: parse, filter, save.
        
        Args:
            dialog_data: Dialog data from JSON
            speaker: User to filter
            
        Returns:
            Information about processing result
        """
        # Parse dialog
        dialog = self.parse_dialog(dialog_data)
        
        # Filter messages
        filtered_messages = self.filter_messages_by_speaker(dialog, speaker)
        
        if not filtered_messages:
            raise ValueError(f"No messages found from user {speaker}")
        
        # Save filtered messages
        filepath = self.save_filtered_messages(filtered_messages, speaker)
        
        return {
            "speaker": speaker,
            "total_dialog_messages": len(dialog),
            "filtered_messages_count": len(filtered_messages),
            "saved_to": filepath
        }