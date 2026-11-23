"""Acme Telecom Customer Support Assistant -- Streamlit App.

Diagnose service, device, and billing issues using a tool-calling agent
grounded in troubleshooting runbooks, device manuals, and account service
policies.
"""

import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

from agent.telecom_support_agent import TelecomSupportAgent  # noqa: E402

# ---------------------------------------------------------------------------
# Simulated customer profiles
# ---------------------------------------------------------------------------
CUSTOMERS = {
    "Alice Johnson": "customer_alice",
    "Bob Martinez": "customer_bob",
    "Carol Smith": "customer_carol",
}


# ---------------------------------------------------------------------------
# Agent (cached so it is only instantiated once)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_agent() -> TelecomSupportAgent:
    return TelecomSupportAgent()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Acme Telecom Support", layout="wide")
st.title("Acme Telecom Customer Support")
st.caption(
    "Describe your service, device, or billing issue and the agent will diagnose it "
    "using troubleshooting runbooks, device manuals, and account policies."
)

# ---------------------------------------------------------------------------
# Sidebar -- user & session controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Customer & Session")

    selected_name = st.selectbox("Simulated customer", options=list(CUSTOMERS.keys()))
    selected_user_id = CUSTOMERS[selected_name]

    if "user_id" not in st.session_state:
        st.session_state.user_id = selected_user_id
    if st.session_state.user_id != selected_user_id:
        st.session_state.user_id = selected_user_id
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(f"**User ID:** `{st.session_state.user_id}`")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")

    st.divider()
    st.markdown("**Try asking:**")
    st.markdown(
        "- *My calls keep dropping at home*\n"
        "- *The ONT light on my fiber modem is red*\n"
        "- *I was charged for roaming I didn't use*\n"
        "- *My texts aren't going through*\n"
        "- *How do I reset my APN settings?*"
    )

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

question = st.chat_input("Describe your issue or ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Diagnosing issue..."):
            agent = get_agent()
            history = st.session_state.messages[:-1]
            answer = agent.answer_question(question, history=history)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
