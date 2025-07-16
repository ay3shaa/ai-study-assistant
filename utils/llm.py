from llama_cpp import Llama

# Load Llama model
llm = Llama(model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=2048 , n_threads = 10, verbose=False)

# Generate flashcards based on context
def generate_flashcards(context_chunks,max_tokens,temperature,num_cards=3):
    context = "\n\n".join(context_chunks)
    prompt = f"""[INST] You are a helpful AI study assistant. Generate  exactly {num_cards} flashcards based on the following context to help the student revise.
    Each flashcard should be in the format:
    Q: <question>
    A: <answer>
    separate each flashcard with "---"
    output exactly {num_cards} flashcards. No explanations. No extra text.
    Context: {context}
    [/INST]"""

    response = llm(prompt, max_tokens=800, temperature=0.5)
    return response['choices'][0]['text'].strip()

# Ask the model with context
def ask_model(question,context,max_tokens,temperature):
    context = "\n\n".join(context)
    prompt = f"""[INST] You are a helpful AI study assistant. Use the following context to answer the user's question clearly.
    Context: {context}
    Question: {question}
    [/INST]"""

    response = llm(prompt, max_tokens=500, temperature=0.5)
    return response['choices'][0]['text'].strip()