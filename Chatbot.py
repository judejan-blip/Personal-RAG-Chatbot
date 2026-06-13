import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("OPENROUTER_API_KEY is not set in your .env file. Please add it and restart.")
    st.stop()

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Read about_me.txt (with error handling)
try:
    with open("about_me.txt", "r", encoding="utf-8") as f:
        ABOUT_ME = f.read()
except FileNotFoundError:
    st.error("about_me.txt not found in the project folder. Please create it.")
    st.stop()

# System prompt
SYSTEM_PROMPT = f"""
You are a chatbot about the Pagalavan.
If user ask's anything, answer in short and crisp tone and be polite.
Answer ONLY using the information below.
If the answer is not present, say you don't know.

Information:
{ABOUT_ME}
"""

# Streamlit page config (optional but nice)
st.set_page_config(page_title="My Personal Chatbot", page_icon="🤖")
st.title("🤖 Pagalavan's Personal Chatbot")
st.caption("Powered by DeepSeek-R1 via OpenRouter")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Pagalavan"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Prepare messages for API (system + full history)
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # Show thinking spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="deepseek/deepseek-r1-0528:free",  # Or "deepseek/deepseek-r1:free"
                    messages=api_messages
                )
                bot_response = response.choices[0].message.content
                st.write(bot_response)

                # Save assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Check your OpenRouter API key, credits, or internet connection.")

# Optional: Add a sidebar with info
with st.sidebar:
    st.header("About This Chatbot")
    st.write("This Chatbot only answer from the Pagalavan's resume.")
    
    st.divider()
    st.write("Model: DeepSeek-R1 (via OpenRouter)")
