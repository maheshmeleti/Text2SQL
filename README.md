# Text2SQL_RAG

QA Sql to text conversion

Contains training and inference scrips for llm based text to sql conversion.

usage:

Train the model on hugging face sql-create-context [b-mc2](https://huggingface.co/datasets/b-mc2/sql-create-context) using Llama-7b [NousResearch](https://huggingface.co/NousResearch/CodeLlama-7b-hf)

Train:
```
python train.py
```

Inference: (change the contents in main)
'''
python infer.py
'''

