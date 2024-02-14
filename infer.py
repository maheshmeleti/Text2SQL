import os
import torch

from utils import _extract_sections, generate_prompt_sql    
import config
from transformers import (BitsAndBytesConfig, 
                          AutoModelForCausalLM, 
                          LlamaTokenizer)
from peft import PeftModel

class TextToSQLGenerator:
    """Handles SQL query generation for a given text prompt."""

    def __init__(
        self, base_model_id = config.BASE_MODEL, 
        use_adapter=False, lora_checkpoint=None, loaded_base_model=None):


        local_model_id = config.BASE_MODEL.replace("/", "--")
        local_model_path = os.path.join(config.MODEL_CACHE_PATH, local_model_id)

        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.float16,
                                                 bnb_4bit_quant_type="nf4")
        
        base_model = AutoModelForCausalLM.from_pretrained(local_model_path,
                                                               quantization_config=quantization_config)
        
        self.model = PeftModel.from_pretrained(base_model, lora_checkpoint)

        self.model.to(config.device)
        self.max_length = 512

        tokenizer_llama = LlamaTokenizer.from_pretrained(local_model_path)
        tokenizer_llama.pad_token_id = 0
        tokenizer_llama.padding_side = "left"
        self.tokenizer = tokenizer_llama

    def generate(self, prompt):

        encoded_prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
            ).input_ids.to(config.device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.model.generate(
                    input_ids=encoded_prompt,
                    do_sample=True,
                    max_length=self.max_length,
                    temperature=0.3,
                    repetition_penalty=1.2,)
                
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated
    

if __name__ == '__main__':

    USING_CHECKPOINT=180
    LORA_CHECKPOINT = f"./lora_adapters/checkpoint-{USING_CHECKPOINT}/"

    # base_model = TextToSQLGenerator(
    #     use_adapter=False,
    #     lora_checkpoint="",
    # )

    finetuned_model = TextToSQLGenerator(
                use_adapter=True,
                lora_checkpoint=LORA_CHECKPOINT
            )
    
    question = "How many heads of the departments are older than 56 ?"
    context =  "CREATE TABLE stadium_info (team_name VARCHAR, stadium_name VARCHAR, capacity INT)"
    prompt = generate_prompt_sql(question, context=context)

    output = finetuned_model.generate(prompt)    
    input_section, context_section, response_section = _extract_sections(output)

    print(input_section)
    print(context_section)
    print(response_section)