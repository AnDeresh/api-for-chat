import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from pathlib import Path
from typing import List, Dict, Any
import os


class ModelTrainer:
    """Class for fine-tuning LLM model with LoRA."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        """
        Initialize trainer.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # LoRA configuration from requirements
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with different configurations based on available hardware
        if torch.cuda.is_available():
            print(f"Loading model on GPU: {torch.cuda.get_device_name()}")
            try:
                # Use float32 for stability (slower but more stable)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # Use float32 for stability
                    trust_remote_code=True
                )
                # Move to GPU manually
                self.model = self.model.to("cuda")
                print("Model loaded on GPU (float32)")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("GPU out of memory, falling back to CPU...")
                    # Clear GPU cache and load on CPU
                    torch.cuda.empty_cache()
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                    self.model = self.model.to("cpu")
                    print("Model loaded on CPU")
                else:
                    raise e
        else:
            print("CUDA not available, loading model on CPU...")
            # CPU loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                trust_remote_code=True
            )
            # Move to CPU explicitly
            self.model = self.model.to("cpu")
            print("Model loaded on CPU")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        print(f"Model loaded on device: {next(self.model.parameters()).device}")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
    
    def prepare_dataset(self, messages: List[str], speaker: str) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            messages: List of messages from speaker
            speaker: Speaker name
            
        Returns:
            Dataset for training
        """
        # Create training examples in chat format
        training_texts = []
        
        for message in messages:
            # Format as instruction-following conversation
            formatted_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are {speaker}, respond in their style.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRespond as {speaker}:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{message}<|eot_id|>"
            training_texts.append(formatted_text)
        
        # Tokenize texts
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors=None
            )
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Create dataset
        dataset = Dataset.from_dict({"text": training_texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        print(f"Prepared dataset with {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def setup_lora_model(self):
        """Setup LoRA model for fine-tuning."""
        print("Setting up LoRA configuration...")
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        return self.model
    
    def train_model(self, messages: List[str], speaker: str) -> Dict[str, Any]:
        """
        Full training pipeline.
        
        Args:
            messages: List of messages for training
            speaker: Speaker name
            
        Returns:
            Training result information
        """
        try:
            # Create output directory (auto-create if doesn't exist)
            output_dir = Path(f"../output_lora/{speaker}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Starting training for {speaker}...")
            print(f"Training messages count: {len(messages)}")
            
            # Load model and tokenizer
            if self.model is None or self.tokenizer is None:
                self.load_model_and_tokenizer()
            
            # Setup LoRA
            self.setup_lora_model()
            
            # Prepare dataset
            train_dataset = self.prepare_dataset(messages, speaker)
            
            # Determine if we're using GPU or CPU
            use_gpu = torch.cuda.is_available() and next(self.model.parameters()).is_cuda
            print(f"Training will use: {'GPU' if use_gpu else 'CPU'}")
            
            # Training arguments (minimal as specified)
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=1,  # 1 epoch as specified
                per_device_train_batch_size=1,  # batch size = 1 as specified
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                warmup_steps=10,
                logging_steps=1,
                save_steps=100,
                save_total_limit=1,
                prediction_loss_only=True,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                fp16=False,  # Disable FP16 for stability
                no_cuda=not use_gpu,  # Disable CUDA if not available
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # We're doing causal LM, not masked LM
                pad_to_multiple_of=8
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Start training
            print("Starting training...")
            trainer.train()
            
            # Save the fine-tuned model
            trainer.save_model()
            self.tokenizer.save_pretrained(str(output_dir))
            
            print(f"Training completed! Model saved to: {output_dir}")
            
            return {
                "model_path": str(output_dir),
                "training_messages": len(messages),
                "epochs": 1,
                "speaker": speaker
            }
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise e
    
    def cleanup(self):
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Memory cleaned up")