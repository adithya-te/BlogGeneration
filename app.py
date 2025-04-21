import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Function to get response from LLama 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    if not input_text or not no_words.isdigit():
        return "Please enter a valid topic and number of words."

    no_words = int(no_words)  # Convert to integer

    # Load LLama2 model
    llm = CTransformers(
        model='models/llama-2-7b-chat.Q8_0.gguf',
        model_type='llama',
        config={'max_new_tokens': no_words, 'temperature': 0.01}
    )

    # Prompt Template
    template = """
    Write a blog for {blog_style} job profile on the topic "{input_text}" 
    within {no_words} words.
    """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template=template)
    
    # Generate response
    formatted_prompt = prompt.format_prompt(blog_style=blog_style, input_text=input_text, no_words=no_words).to_string()
    response = llm(formatted_prompt)

    return response

# Streamlit UI
st.set_page_config(
    page_title="Generate Blogs",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Creating two columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')

with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

# Final response
if submit:
    response = getLLamaresponse(input_text, no_words, blog_style)
    st.write(response)
