import sys
import site
from pathlib import Path
import logging
import os
import sys
from math import ceil
from typing import Optional, Tuple
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import (
    DataCollatorForSeq2Seq,
    LlamaTokenizer,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM, 
    AutoTokenizer
)

import config

import wandb
os.environ["WANDB_NOTEBOOK_NAME"] = 'finetuning'
os.environ["WANDB_PROJECT"] = f"text-to-sql-finetune-model-name_{config.BASE_MODEL.replace('/', '_')}"
wandb.login()
ENABLE_WANDB = True


from tokenizer import Tokenizer


transformers.logging.set_verbosity_error()

device = config.device
local_model_id = config.BASE_MODEL.replace("/", "--")
local_model_path = os.path.join(config.MODEL_CACHE_PATH, local_model_id)


## Model 
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            quantization_config=quantization_config)

model = prepare_model_for_kbit_training(model)

LORA_CONFIG = LoraConfig(
    r=16,  # rank
    lora_alpha=32,  # scaling factor
    target_modules=["q_proj", "k_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, LORA_CONFIG).to(device)

### tokenizer and data prep
tokenizer_llama = LlamaTokenizer.from_pretrained(local_model_path)
tokenizer_llama.pad_token_id = 0
tokenizer_llama.padding_side = "left"


data = load_dataset(config.DATA_PATH)
val_set_size = config.val_set_size
train_val_split = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )

tokenizer = Tokenizer(tokenizer_llama)
train_data = train_val_split["train"].shuffle().map(tokenizer.tokenize_data)
val_data = train_val_split["test"].shuffle().map(tokenizer.tokenize_data)


training_args = TrainingArguments(
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accum_steps,
            warmup_steps=config.warmup_steps,
            save_steps=config.save_steps,
            save_strategy="steps",
            eval_steps=config.eval_steps,
            evaluation_strategy="steps",
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            #max_grad_norm=max_grad_norm,
            bf16=True,
            #lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            save_total_limit=config.save_total_limit,
            logging_steps=config.logging_steps,
            optim="adamw_hf",
            output_dir="./lora_adapters",
            logging_dir="./logs",
            report_to="wandb" if ENABLE_WANDB else [],
        )

trainer = Trainer(
                model=model,
                train_dataset=train_data,
                eval_dataset=val_data,
                args=training_args,
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer_llama,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                    padding=True,
                ),
            )
model.config.use_cache = False

trainer.train()
model.save_pretrained(config.MODEL_PATH)


