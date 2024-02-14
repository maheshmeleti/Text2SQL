#model cache path
MODEL_CACHE_PATH = '/scratch/umeleti/code/LLM/Text2SQL/MODEL_CACHE/'

#Device
device = 'cuda'


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

#### paths
BASE_MODEL = BASE_MODELS["3"]
DATA_PATH = "b-mc2/sql-create-context"
MODEL_PATH = "./final_model"
ADAPTER_PATH = "./lora_adapters"

## training args
per_device_batch_size=4
warmup_steps=20
learning_rate=2e-5
max_steps=200
gradient_accum_steps=4
save_steps = 20
eval_steps = 20
max_grad_norm = 0.3
save_total_limit = 3
logging_steps = 20