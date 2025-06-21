# from transformers import T5Tokenizer, T5ForConditionalGeneration

# def summarize_text(text):
#     tokenizer = T5Tokenizer.from_pretrained("t5-base")
#     model = T5ForConditionalGeneration.from_pretrained("t5-base")

#     preprocess_text = text.strip().replace("\n", " ")
#     t5_input_text = "summarize: " + preprocess_text

#     tokenized_text = tokenizer.encode(t5_input_text, return_tensors="pt", max_length=512, truncation=True)

#     summary_ids = model.generate(tokenized_text, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=200, early_stopping=True)

#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     return summary
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model and tokenizer once
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text, max_len=150):
    input_text = "summarize: " + text.strip().replace("\n", " ")
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    summary_ids = model.generate(
        inputs,
        max_length=max_len,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
