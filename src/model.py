from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from embedding import get_chrome_client


def get_relevant_docs(user_query, top_k=3):
    "Get the revelant docs from Chroma DB"
    retrieved_docs = collection.query(query_texts=[user_query], n_results=top_k)
    # combine top_k docs 
    retrieved_docs = "\n".join(retrieved_docs.get("documents", [])[0])
    return retrieved_docs


def get_model_response(formatted_prompt, max_length=512, max_new_tokens=60):
    "Encode the prompt and get model response"
    # not using padding as passing single query
    encoded_input = tokenizer(formatted_prompt, 
                              return_tensors='pt',
                              truncation=True, 
                              max_length=max_length).to(device=device)
    with torch.no_grad():
        # leaving everything on default for not 
        # TODO: Experiment with temperature=0.7, top_p=0.9
        model_output = model.generate(**encoded_input, max_new_tokens=max_new_tokens)

    # format model output
    # model_output[0]: as model_output shape is [1, x]
    formatted_model_output = tokenizer.decode(model_output[0], skip_special_tokens=True)
    return formatted_model_output

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    COLLECTION_NAME = "budget_rag"
    PERSIST_DIRECTORY = "/home/shubham/build_in_public/pdf-chatbot/chroma_db"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=device)

    # Chroma client 
    chroma_client, collection = get_chrome_client(PERSIST_DIRECTORY, COLLECTION_NAME)

    user_query = "what is the credit card limit for micro enterprises?"
    retrieved_docs = get_relevant_docs(user_query=user_query)

    # creating prompt for the model
    messages = [
    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents."},
    {"role": "user", "content": f"Context:\n{retrieved_docs}\n\nQuestion: {user_query}\n"},
    ]

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    formatted_model_output = get_model_response(formatted_prompt, max_length=512, max_new_tokens=60)
    print(formatted_model_output)
