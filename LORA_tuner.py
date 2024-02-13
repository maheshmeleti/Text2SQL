import sys
import site
from pathlib import Path
import logging
import os
import sys
from math import ceil
from typing import Optional, Tuple
import torch
# import intel_extension_for_pytorch as ipex
from datasets import load_dataset
from datasets import Dataset
# from bigdl.llm.transformers import AutoModelForCausalLM
# from bigdl.llm.transformers.qlora import (
#     get_peft_model,
#     prepare_model_for_kbit_training as prepare_model,
# )
from peft import LoraConfig
# from bigdl.llm.transformers.qlora import PeftModel
import transformers
from transformers import (
    DataCollatorForSeq2Seq,
    LlamaTokenizer,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import config

from transformers import AutoModelForCausalLM, AutoTokenizer

transformers.logging.set_verbosity_error()
# import wandb
# wandb.login()
# ENABLE_WANDB = True

def get_python_version():
    return "python" + ".".join(map(str, sys.version_info[:2]))

def set_local_bin_path():
    local_bin = str(Path.home() / ".local" / "bin") 
    local_site_packages = str(
        Path.home() / ".local" / "lib" / get_python_version() / "site-packages"
    )
    sys.path.append(local_bin)
    sys.path.insert(0, site.getusersitepackages())
    sys.path.insert(0, sys.path.pop(sys.path.index(local_site_packages)))

set_local_bin_path()

local_model_id = config.BASE_MODEL.replace("/", "--")
local_model_path = os.path.join(config.MODEL_CACHE_PATH, local_model_id)
print(f"local model path is: {local_model_path}")


model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            load_in_low_bit="nf4",
            optimize_model=False,
            torch_dtype=torch.float16,
            modules_to_not_convert=["lm_head"],
        )

# training_args = TrainingArguments(
#             per_device_train_batch_size=config.per_device_batch_size,
#             gradient_accumulation_steps=config.gradient_accum_steps,
#             warmup_steps=config.warmup_steps,
#             save_steps=config.save_steps,
#             save_strategy="steps",
#             eval_steps=config.eval_steps,
#             evaluation_strategy="steps",
#             max_steps=config.max_steps,
#             learning_rate=config.learning_rate,
#             #max_grad_norm=max_grad_norm,
#             bf16=True,
#             #lr_scheduler_type="cosine",
#             load_best_model_at_end=True,
#             ddp_find_unused_parameters=False,
#             group_by_length=True,
#             save_total_limit=config.save_total_limit,
#             logging_steps=config.logging_steps,
#             optim=config.optim,
#             output_dir=config.output_dir,
#             logging_dir=config.logging_dir,
#             report_to=config.report_to
#         )



# data = load_dataset(config.data_path)

# train_val_split = data["train"].train_test_split(
#                 test_size=config.val_set_size, shuffle=True, seed=42
#             )

# train_data = train_val_split["train"].shuffle().map(self.tokenize_data)
# val_data = train_val_split["test"].shuffle().map(self.tokenize_data)

# train_data, val_data = self.prepare_data(data)
# self.train_model(train_data, val_data, training_args)
     
