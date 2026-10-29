"""Streamlit chat UI for the FAQ bot."""
import streamlit as st
from chatbot import FAQBot

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")
st.title("💬 ShopEase FAQ Chatbot")

@st.cache_resource
def load_bot():
    return FAQBot()

bot = load_bot()

if "history" not in st.session_state:
    st.session_state.history = [("assistant", "Hi! Ask me about orders, shipping, returns, payments or your account.")]

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if q := st.chat_input("Type your question..."):
    answer, matched, score = bot.ask(q)
    st.session_state.history += [("user", q), ("assistant", answer)]
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        st.write(answer)
        if matched:
            st.caption(f"Matched FAQ: “{matched}” (similarity {score:.2f})")

with st.sidebar:
    st.header("Sample questions")
    for f in bot.faqs[:6]:
        st.write("•", f["question"])
