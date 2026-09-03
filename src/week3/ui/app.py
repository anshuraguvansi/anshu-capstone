import streamlit as st
import requests


BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Anshu Capstone UI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.title("Capstone Q&A")
st.caption("Ask anything — the API streams the answer back token by token.")

with st.form(key="ask_form", clear_on_submit=False):
    question = st.text_input(
        "Your question:",
        placeholder="e.g. What is the leave policy?",
    )
    submitted = st.form_submit_button("Ask")

if submitted and question:
    placeholder = st.empty()
    answer = ""
    try:
        with requests.post(
            f"{BASE_URL}/ask",
            json={"question": question},
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(decode_unicode=True):
                if chunk:
                    answer += chunk
                    placeholder.markdown(answer)
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Is uvicorn running on port 8000?")
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
    except requests.exceptions.Timeout:
        st.error("Request timed out after 60 seconds.")
