import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3ForCausalLM
import torch

# Load Qwen model and tokenizer (you can change to a different Qwen version)
@st.cache_resource
def load_model():
    model_name = r"C:\Users\twang\Work\models\qwen3-0.6b\model\snapshots\e6de91484c29aa9480d55605af694f39b081c455"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3ForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("Chat Demo")
st.write("Welcome and please interact with the chatbot!")

# User input
user_input = st.text_area("Enter your prompt here:", height=200)

if st.button("Generate"):
    if user_input.strip():
        # Tokenize and generate
        inputs = tokenizer(user_input, return_tensors="pt")
        attention_mask = inputs["attention_mask"] 
        generate_ids = model.generate(inputs.input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_length=100)

        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # inputs = tokenizer(user_input, return_tensors="pt")
        # with torch.no_grad():
        #     outputs = model.generate(
        #         inputs.input_ids,
        #         max_new_tokens=256,
        #         do_sample=True,
        #         temperature=0.7,
        #         top_p=0.9,
        #         pad_token_id=tokenizer.eos_token_id,
        #     )
        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Display only the new generation (after prompt)
        st.markdown("**Model Output:**")
        st.write(response[len(user_input):].strip())
    else:
        st.warning("Please enter a prompt to generate a response.")