from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto", 
    torch_dtype=torch.float16
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_flashcards(context_chunks, max_tokens=800, temperature=0.5, num_cards=3):
    context = "\n\n".join(context_chunks)
    prompt = f"""[INST] You are a helpful AI study assistant. Generate exactly {num_cards} flashcards based on the following context to help the student revise.
Each flashcard should be in the format:
Q: <question>
A: <answer>
Separate each flashcard with "---"
Output exactly {num_cards} flashcards. No explanations. No extra text.
Context: {context} [/INST]"""
    
    response = generator(prompt, max_new_tokens=max_tokens, temperature=temperature)
    return response[0]['generated_text']

def ask_model(question, context_chunks, max_tokens=500, temperature=0.5):
    context = "\n\n".join(context_chunks)
    prompt = f"""[INST] You are a helpful AI study assistant. Use the following context to answer the user's question clearly.
Context: {context}
Question: {question} [/INST]"""

    response = generator(prompt, max_new_tokens=max_tokens, temperature=temperature)
    return response[0]['generated_text']
