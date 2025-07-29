import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from pathlib import Path
from typing import List, Dict, Any


class ModelTrainer:
    """Fine-tune LLM model with LoRA."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM
        )
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with GPU/CPU fallback."""
        print(f"Loading {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Try GPU first, fallback to CPU
        if torch.cuda.is_available():
            try:
                print(f"Loading on GPU: {torch.cuda.get_device_name()}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype=torch.float32, trust_remote_code=True
                ).to("cuda")
                print("Model loaded on GPU")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("GPU OOM, loading on CPU...")
                    torch.cuda.empty_cache()
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, torch_dtype=torch.float32, trust_remote_code=True
                    ).to("cpu")
                else:
                    raise e
        else:
            print("Loading on CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float32, trust_remote_code=True
            ).to("cpu")
        
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def prepare_dataset(self, messages: List[str], speaker: str) -> Dataset:
        """Prepare training dataset."""
        training_texts = [
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are {speaker}, respond in their style.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\nRespond as {speaker}:<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg}<|eot_id|>"
            for msg in messages
        ]
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"], truncation=True, padding=False, 
                max_length=512, return_tensors=None
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        dataset = Dataset.from_dict({"text": training_texts})
        return dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    def train_model(self, messages: List[str], speaker: str) -> Dict[str, Any]:
        """Complete training pipeline."""
        output_dir = Path(f"../output_lora/{speaker}").resolve()  # Resolve absolute path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Training {speaker} on {len(messages)} messages...")
        
        if self.model is None:
            self.load_model_and_tokenizer()
        
        # Setup LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
        # Prepare data
        train_dataset = self.prepare_dataset(messages, speaker)
        use_gpu = torch.cuda.is_available() and next(self.model.parameters()).is_cuda
        
        # Training setup
        training_args = TrainingArguments(
            output_dir=str(output_dir), num_train_epochs=1, per_device_train_batch_size=1,
            learning_rate=2e-4, warmup_steps=10, logging_steps=1, save_steps=100,
            save_total_limit=1, prediction_loss_only=True, remove_unused_columns=False,
            dataloader_pin_memory=False, fp16=False, no_cuda=not use_gpu
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8
        )
        
        trainer = Trainer(
            model=self.model, args=training_args, train_dataset=train_dataset,
            data_collator=data_collator, tokenizer=self.tokenizer
        )
        
        # Train and save
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(str(output_dir))
        
        print(f"Training completed! Saved to: {output_dir}")
        return {
            "model_path": str(output_dir),
            "training_messages": len(messages),
            "epochs": 1,
            "speaker": speaker
        }
    
    def cleanup(self):
        """Clean up memory."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()