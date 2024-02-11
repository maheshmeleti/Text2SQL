import os
import shutil
import logging
import os
import sys
from math import ceil
from typing import Optional, Tuple
import warnings
import pandas as pd
import random

warnings.filterwarnings(
    "ignore", category=UserWarning, module="intel_extension_for_pytorch"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.io.image", lineno=13
)
warnings.filterwarnings("ignore", message="You are using the default legacy behaviour")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Parameter.*")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="This implementation of AdamW is deprecated",
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NUMEXPR_MAX_THREADS"] = "28"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bigdl").setLevel(logging.ERROR)


import torch
import intel_extension_for_pytorch as ipex
from datasets import load_dataset
from datasets import Dataset
from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.transformers.qlora import (
    get_peft_model,
    prepare_model_for_kbit_training as prepare_model,
)
from peft import LoraConfig
from bigdl.llm.transformers.qlora import PeftModel
import transformers
from transformers import (
    DataCollatorForSeq2Seq,
    LlamaTokenizer,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

transformers.logging.set_verbosity_error()


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

MODEL_CACHE_PATH = "/home/common/data/Big_Data/GenAI/llm_models"

INFERENCE_DEVICE = torch.device("cpu")  
INFERENCE_DEVICE = torch.device("xpu")  

def setup_model_and_tokenizer(base_model_id: str):
    """Downloads / Load the pre-trained model in 4bit and tokenizer based on the given base model ID for inference."""
    local_model_id = base_model_id.replace("/", "--")
    local_model_path = os.path.join(MODEL_CACHE_PATH, local_model_id)
    print(f"local model path is: {local_model_path}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            load_in_4bit=True,
            optimize_model=True,
            use_cache=True,
            torch_dtype=torch.float16,
            modules_to_not_convert=["lm_head"],
        )
    except OSError:
        logging.info(
            f"Model not found locally. Downloading {base_model_id} to cache..."
        )
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            load_in_4bit=True,
            optimize_model=True,
            use_cache=True,
            torch_dtype=torch.float16,
            modules_to_not_convert=["lm_head"],
        )

    try:
        if "llama" in base_model_id.lower():
            tokenizer = LlamaTokenizer.from_pretrained(local_model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    except OSError:
        logging.info(
            f"Tokenizer not found locally. Downloading tokenizer for {base_model_id} to cache..."
        )
        if "llama" in base_model_id.lower():
            tokenizer = LlamaTokenizer.from_pretrained(base_model_id)
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return model, tokenizer

class TextToSQLGenerator:
    """Handles SQL query generation for a given text prompt."""

    def __init__(
        self, base_model_id=BASE_MODEL, use_adapter=False, lora_checkpoint=None, loaded_base_model=None
    ):
        """
        Initialize the InferenceModel class.
        Parameters:
            use_adapter (bool, optional): Whether to use LoRA model. Defaults to False.
        """
        try:
            if loaded_base_model:
                self.model = loaded_base_model.model
                self.tokenizer = loaded_base_model.tokenizer
            else:
                self.model, self.tokenizer = setup_model_and_tokenizer(base_model_id)
            if use_adapter:
                self.model = PeftModel.from_pretrained(self.model, lora_checkpoint)
        except Exception as e:
            logging.error(f"Exception occurred during model initialization: {e}")
            raise

        self.model.to(INFERENCE_DEVICE)
        self.max_length = 512


    def generate(self, prompt, **kwargs):
        """Generates an SQL query based on the given prompt.
        Parameters:
            prompt (str): The SQL prompt.
        Returns:
            str: The generated SQL query.
        """
        try:
            encoded_prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
            ).input_ids.to(INFERENCE_DEVICE)
            with torch.no_grad():
                with torch.xpu.amp.autocast():
                    outputs = self.model.generate(
                        input_ids=encoded_prompt,
                        do_sample=True,
                        max_length=self.max_length,
                        temperature=0.3,
                        repetition_penalty=1.2,
                    )
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated
        except Exception as e:
            logging.error(f"Exception occurred during query generation: {e}")
            raise
def generate_prompt_sql(input_question, context, output=""):
    """
    Generates a prompt for fine-tuning the LLM model for text-to-SQL tasks.

    Parameters:
        input_question (str): The input text or question to be converted to SQL.
        context (str): The schema or context in which the SQL query operates.
        output (str, optional): The expected SQL query as the output.

    Returns:
        str: A formatted string serving as the prompt for the fine-tuning task.
    """
    return f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. 

You must output the SQL query that answers the question.

### Input:
{input_question}

### Context:
{context}

### Response:
{output}"""

USING_CHECKPOINT=480
LORA_CHECKPOINT = f"./lora_adapters/checkpoint-{USING_CHECKPOINT}/"
base_model = TextToSQLGenerator(
    use_adapter=False,
    lora_checkpoint="",
)

finetuned_model = TextToSQLGenerator(
            use_adapter=True,
            lora_checkpoint=LORA_CHECKPOINT,
            loaded_base_model=base_model
        )
# run_inference(sample_data, model=finetuned_model, finetuned=True)

def _extract_sections(output):
    input_section = output.split("### Input:")[1].split("### Context:")[0]
    context_section = output.split("### Context:")[1].split("### Response:")[0]
    response_section = output.split("### Response:")[1]
    return input_section, context_section, response_section


mimicsql_data = 'data_base/mimicsql_data'
mimicsql_natural = os.path.join(mimicsql_data, 'mimicsql_natural')
DATA_PATH = os.path.join(mimicsql_natural, 'test.csv')

test_data_path = os.path.join(DATA_PATH)

test_data = pd.read_csv(test_data_path)

q = []
a = []
res = []
for i in range(100):
    ix = random.randint(0, len(test_data))
    row = test_data.loc[ix]
    question = row['question']
    answer = row['answer']

    prompt = generate_prompt_sql(question, context='')
    output = finetuned_model.generate(prompt)            
    input_section, context_section, response_section = _extract_sections(output)

    q.append(question)
    a.append(answer)
    res.append(response_section)


final_csv = {'question':q, 'answer':a, 'prediction':res}
final_csv = pd.DataFrame(final_csv)

final_csv.to_csv('data_base/lama7b.csv')