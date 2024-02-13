
#model cache path
MODEL_CACHE_PATH = '/scratch/umeleti/code/LLM/Text2SQL/MODEL_CACHE/'

#### paths
data_path = ''
val_set_size = 100

### models
BASE_MODELS = {
    "0": "NousResearch/Nous-Hermes-Llama-2-7b",  # https://huggingface.co/NousResearch/Nous-Hermes-llama-2-7b
    "1": "NousResearch/Llama-2-7b-chat-hf",  # https://huggingface.co/NousResearch/Llama-2-7b-chat-hf
    "2": "NousResearch/Llama-2-13b-hf",  # https://huggingface.co/NousResearch/Llama-2-13b-hf
    "3": "NousResearch/CodeLlama-7b-hf",  # https://huggingface.co/NousResearch/CodeLlama-7b-hf
    "4": "Phind/Phind-CodeLlama-34B-v2",  # https://huggingface.co/Phind/Phind-CodeLlama-34B-v2
    "5": "openlm-research/open_llama_3b_v2",  # https://huggingface.co/openlm-research/open_llama_3b_v2
    "6": "openlm-research/open_llama_13b",  # https://huggingface.co/openlm-research/open_llama_13b
    "7": "HuggingFaceH4/zephyr-7b-beta", # https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
}
BASE_MODEL = BASE_MODELS["3"]


#### training arguments
# per_device_train_batch_size=per_device_batch_size,
# gradient_accumulation_steps=gradient_accum_steps,
# warmup_steps=warmup_steps,
# save_steps=save_steps,
# save_strategy="steps",
# eval_steps=eval_steps,
# evaluation_strategy="steps",
# max_steps=max_steps,
# learning_rate=learning_rate,
# #max_grad_norm=max_grad_norm,
# bf16=True,
# #lr_scheduler_type="cosine",
# load_best_model_at_end=True,
# ddp_find_unused_parameters=False,
# group_by_length=True,
# save_total_limit=save_total_limit,
# logging_steps=logging_steps,
# optim="adamw_hf",
# output_dir="./lora_adapters",
# logging_dir="./logs",
# report_to="wandb" if ENABLE_WANDB else []