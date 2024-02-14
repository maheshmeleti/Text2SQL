from utils import generate_prompt_sql


class Tokenizer():
    def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    def tokenize_data(self, data_points, add_eos_token=True, train_on_inputs=False, cutoff_len=512):
            question = data_points["question"]
            context = data_points["context"]
            answer = data_points["answer"]
            combined_text = generate_prompt_sql(question, context, answer)
            tokenized = self.tokenizer(
                    combined_text,
                    truncation=True,
                    max_length=cutoff_len,
                    padding=False,
                    return_tensors=None)

            if (tokenized["input_ids"][-1] != self.tokenizer.eos_token_id and add_eos_token
                    and len(tokenized["input_ids"]) < cutoff_len):
                    tokenized["input_ids"].append(self.tokenizer.eos_token_id)
                    tokenized["attention_mask"].append(1)
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized