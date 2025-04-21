📝 AI Blog Generator
Welcome to the AI Blog Generator – a simple yet powerful tool that helps you create blog posts in seconds using the LLaMA 2 model! Whether you're a researcher, data scientist, or just want content in plain English, this app's got you covered.

🔗 Try it Live
https://adithya-te-bloggeneration-app-bspzrt.streamlit.app/

🚀 What does it do?
This app generates a blog post based on a topic you provide. You just enter:

A topic

Desired word count

And select the target audience (like Researchers, Data Scientists, or Common People)

Hit Generate, and voilà – your blog is ready!

🧠 Powered By
Streamlit: For the interactive front-end UI.

LangChain: To build prompt templates.

CTransformers: For running the LLaMA 2 model locally.

⚙️ Model Used: llama-2-7b-chat.Q8_0.gguf

🛠️ How to Run Locally
If you'd like to run the app on your machine:

1. Clone the repository
git clone <your-repo-url>
cd <your-project-folder>

2. Install the required packages
pip install -r requirements.txt

3. Run the app
streamlit run app.py

📌 Features
Clean and minimal UI

Easy prompt customization

Supports different writing tones based on audience

Lightweight local model inference (no cloud APIs needed)

📈 Future Ideas
Here are some potential improvements:

Add more audience types and tones (e.g., technical, persuasive)

Export generated blogs as .txt or .md files

Add image suggestions via AI

Improve formatting and grammar tuning
