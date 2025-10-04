### Min Li - AI Agent Front_part
import streamlit as st
import sys
from pathlib import Path

### add project root so we can import agent
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
from agent.commerce_agent import CommerceAgent

### page setup
st.set_page_config(page_title="Palona AI Agent", layout="centered")

### quick CSS tweak for background + chat bubbles
st.markdown(
    """
    <style>
    .stApp { background-color: #EEDFCC; color: #000; }
    .user-bubble { background:#d8cfc4; padding:10px; border-radius:12px; float:right; max-width:70%; }
    .agent-bubble { background:#fff; padding:10px; border-radius:12px; float:left; max-width:70%; border:1px solid #ccc; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Palona AI Agent")

### init agent + history
if "agent" not in st.session_state:
    st.session_state.agent = CommerceAgent()
if "history" not in st.session_state:
    st.session_state.history = []

### render chat history
for role, msg in st.session_state.history:
    bubble = "user-bubble" if role == "user" else "agent-bubble"
    st.markdown(f"<div class='{bubble}'>{msg}</div>", unsafe_allow_html=True)

query = st.chat_input("Enter your query (text, image path, or text | image):")

if query:
    st.session_state.history.append(("user", query))
    try:
        if "|" in query:  # mixed
            text, image = [p.strip() for p in query.split("|")]
            response = st.session_state.agent.handle_mixed_query(text, image)
        elif query.lower().startswith("search:"):  # explicit search
            text = query.split("search:", 1)[1].strip()
            response = st.session_state.agent.handle_query(text)
        elif query.endswith((".jpg", ".png")):  # image
            response = st.session_state.agent.handle_image_query(query)
        else:
            response = st.session_state.agent.handle_general_conversation(query)
    except Exception as e:
        response = f"Error: {e}"

    st.session_state.history.append(("agent", response))
    st.rerun()
### #%#