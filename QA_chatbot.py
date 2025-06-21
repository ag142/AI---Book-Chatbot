# from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# # Initialize the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
# model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

# # Initialize the pipeline for question answering
# qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# def ask_question(question, context):
#     result = qa_pipeline(question=question, context=context)
#     return result['answer']
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model and tokenizer once
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def ask_question(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    output = model.generate(
        inputs,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer
