# v1.35 Streamlit GUI for Multi-Agent System (Sticky prompt & single input, reverse chronological order, connection messages)
import os
import requests
import json
import uuid
import re
from dotenv import load_dotenv
import streamlit as st

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

from PIL import Image
import base64

# Load the image
sidebar_image = Image.open("multi-agent.png")

def get_image_base64(image):
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

img_base64 = get_image_base64(sidebar_image)

with st.sidebar:
    st.markdown(
        f"""
        <div style='text-align: center; margin-top: -20px;'>
            <img src='data:image/png;base64,{img_base64}' style='max-width: 100%; height: auto;' />
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Select LLM model:",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"],
        index=0
    )

    with st.expander("ℹ️ What is this?"):
        st.markdown(
            """
            This is a multi-agent chatbot interface.  
            - **CustomAgent** handles general queries  
            - **Agentforce** connects to a Salesforce specialist  
            Type keywords like *help* or *issue* to trigger escalation.
            """
        )

    memory_window = st.slider("Memory Window", 1, 20, 10)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

load_dotenv()

ESCALATION_KEYWORDS = ["support", "help", "issue", "problem", "troubleshoot", "fix","case status", "cases"]

### ---- CustomAgent (Groq API) ---- ###
def chat_with_customagent(user_input, selected_model, memory_window):
    if "customagent_memory" not in st.session_state:
        st.session_state.customagent_memory = ConversationBufferWindowMemory(
            k=memory_window, memory_key="chat_history", return_messages=True
        )
    raw_key = os.getenv("GROQ_API_KEY")
    groq_api_key = raw_key.strip().strip('"') if raw_key else ""
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=selected_model)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are CustomAgent, a helpful chatbot that assists users with general inquiries. Please provide complete and comprehensive responses in full sentences."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ])
    conversation = LLMChain(llm=groq_chat, prompt=prompt, memory=st.session_state.customagent_memory)
    response = conversation.predict(human_input=user_input)
    return response

# ......

### ---- Agentforce (Salesforce API) ---- ###
def get_access_token():
    sf_instance = os.getenv("SF_INSTANCE")
    url = f"{sf_instance}/services/oauth2/token"
    payload = {
        'grant_type': 'client_credentials',
        'client_id': os.getenv("SF_CLIENT_ID"),
        'client_secret': os.getenv("SF_CLIENT_SECRET")
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, data=payload, headers=headers)
    if response.status_code != 200:
        st.write(f"[ERROR] Salesforce API responded with: {response.status_code} - {response.text}")
    response.raise_for_status()
    return response.json().get("access_token")

def create_session(access_token):
    url = "https://api.salesforce.com/einstein/ai-agent/v1/agents/0XxWt0000005qu1KAA/sessions"
    session_key = str(uuid.uuid4())
    payload = {
        "externalSessionKey": session_key,
        "instanceConfig": {"endpoint": os.getenv("SF_INSTANCE")},
        "streamingCapabilities": {"chunkTypes": ["Text"]},
        "bypassUser": True
    }
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {access_token}'}
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json().get("sessionId")

def send_message_agentforce(session_id, access_token, user_message):
    if "agentforce_seq" not in st.session_state:
        st.session_state.agentforce_seq = 1
    sequence_id = st.session_state.agentforce_seq
    st.session_state.agentforce_seq += 1

    url = f"https://api.salesforce.com/einstein/ai-agent/v1/sessions/{session_id}/messages/stream"
    payload = {
        "message": {
            "sequenceId": sequence_id,
            "type": "Text",
            "text": user_message
        }
    }
    headers = {
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.post(url, json=payload, headers=headers, stream=True)
    if response.status_code != 200:
        st.write(f"[ERROR] Agentforce API error: {response.status_code} - {response.text}")
        return "Error: Unable to get response from Agentforce."
    raw_response = []
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                try:
                    event = json.loads(decoded_line[6:])
                    raw_response.append(event)
                    if event.get("finish_reason") == "stop":
                        break
                except json.JSONDecodeError:
                    continue
    if not raw_response:
        return "Agentforce: No response message found."
    inform_events = [event for event in raw_response if event.get("message", {}).get("type") == "Inform"]
    if inform_events:
        return inform_events[0]["message"]["message"]
    textchunk_events = [event for event in raw_response if event.get("message", {}).get("type") == "TextChunk"]
    if textchunk_events:
        return textchunk_events[0]["message"]["message"]
    return "Agentforce: No valid response found."

### ---- Streamlit Multi-Agent Controller ---- ###
def main():
    st.markdown(
        """
        <style>
        .sticky-prompt {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 100;
            background-color: white;
            padding: 10px;
            border-bottom: 1px solid #ccc;
        }
        .spacer {
            height: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "mode" not in st.session_state:
        st.session_state.mode = "general"
    if "agentforce_session" not in st.session_state:
        st.session_state.agentforce_session = {"access_token": None, "session_id": None}

    if st.session_state.get("escalation_triggered") and not st.session_state.get("pending_confirmation"):
        try:
            user_msg = st.session_state.pop("escalated_input", "")
            st.session_state.chat_history.append({"sender": "System", "message": "Sending authentication request to Salesforce..."})
            access_token = get_access_token()
            session_id = create_session(access_token)
            st.session_state.agentforce_session = {"access_token": access_token, "session_id": session_id}
            st.session_state.chat_history.append({"sender": "System", "message": "Connected to Salesforce. Specialist agent is now available."})
            if user_msg:
                response = send_message_agentforce(session_id, access_token, user_msg)
                st.session_state.chat_history.append({"sender": "Agentforce", "message": response})
        except requests.exceptions.RequestException as e:
            st.session_state.chat_history.append({"sender": "System", "message": f"Error: {e}"})
        st.session_state.pop("escalation_triggered", None)
        st.rerun()

    with st.container():
        st.markdown('<div class="sticky-prompt">', unsafe_allow_html=True)
        user_input = st.text_input("Your message:", key="user_input")
        if st.button("Send") and user_input:
            st.session_state.chat_history.append({"sender": "User", "message": user_input})
            if user_input.strip().lower() == "exit":
                if st.session_state.mode == "agentforce":
                    st.session_state.chat_history.append({"sender": "System", "message": "Exiting Agentforce mode."})
                    st.session_state.mode = "general"
                else:
                    st.session_state.chat_history.append({"sender": "System", "message": "Session ended."})
            elif st.session_state.mode == "general":
                if any(keyword in user_input.lower() for keyword in ESCALATION_KEYWORDS):
                    st.session_state.chat_history.append({"sender": "CustomAgent", "message": "I will connect you to a specialist agent."})
                    st.session_state.mode = "agentforce"
                    st.session_state["pending_confirmation"] = True
                    st.session_state["escalated_input"] = user_input
                else:
                    response = chat_with_customagent(user_input, selected_model, memory_window)
                    st.session_state.chat_history.append({"sender": "CustomAgent", "message": response})
            elif st.session_state.mode == "agentforce":
                access_token = st.session_state.agentforce_session.get("access_token")
                session_id = st.session_state.agentforce_session.get("session_id")
                if not access_token or not session_id:
                    st.session_state.chat_history.append({"sender": "System", "message": "Agentforce session not initialized."})
                else:
                    response = send_message_agentforce(session_id, access_token, user_input)
                    st.session_state.chat_history.append({"sender": "Agentforce", "message": response})
        st.markdown("</div>", unsafe_allow_html=True)

        # Show confirmation button after CustomAgent message
        if st.session_state.get("pending_confirmation"):
            if st.button("OK - Let's connect to the specialist."):
                st.session_state["escalation_triggered"] = True
                st.session_state.pop("pending_confirmation", None)
                st.rerun()

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    for entry in reversed(st.session_state.chat_history):
        sender = entry["sender"]
        message = entry["message"]

        if sender == "CustomAgent":
            icon = "🤖"
        elif sender == "Agentforce":
            icon = "🛠️"
        elif sender == "User":
            icon = "🧑"
        else:
            icon = "ℹ️"

        st.markdown(f"**{icon} {sender}:** {message}")

if __name__ == "__main__":
    main()
