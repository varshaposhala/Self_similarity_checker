# app.py

import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Smart Semantic Question Analyzer",
    page_icon="🧠",
    layout="wide"
)

# --- Initialize session state for API key validation ---
if "api_key_validated" not in st.session_state:
    st.session_state.api_key_validated = False

# --- SIDEBAR: API KEY ---
st.sidebar.title("🔐 Google Gemini API Key")
api_key_input = st.sidebar.text_input(
    "Enter your Google Gemini API key:",
    type="password",
    help="Get your key from https://makersuite.google.com",
    # Use on_change to re-validate when the user changes the key
    on_change=lambda: st.session_state.update(api_key_validated=False)
)

# --- API KEY VALIDATION LOGIC ---
if api_key_input and not st.session_state.api_key_validated:
    # Use .strip() to remove any accidental whitespace from the pasted key
    cleaned_api_key = api_key_input.strip()
    
    try:
        genai.configure(api_key=cleaned_api_key)
        # Make a lightweight API call to check if the key is valid
        list(genai.list_models())
        st.session_state.api_key_validated = True
        st.sidebar.success("API Key is valid and configured! ✅")
    except Exception as e:
        st.session_state.api_key_validated = False
        st.sidebar.error(f"API key is not valid. Please check and re-enter. Error: {e}")

# Stop the app if the API key is not validated
if not st.session_state.api_key_validated:
    st.warning("Please enter a valid Google Gemini API key in the sidebar to proceed.")
    st.stop()


# --- EMBEDDING FUNCTION (No changes needed, but added a note) ---
@st.cache_data(show_spinner=False)
def get_embeddings(texts):
    # This function will now only be called if the API key has been validated above.
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=texts,
            task_type="RETRIEVAL_DOCUMENT" # Good practice to specify task type
        )
        return np.array(response["embedding"])
    except Exception as e:
        st.error(f"Error during embedding: {e}")
        return np.array([])

# --- MAIN UI (Your existing code from here down is fine) ---
st.title("🧠 Smart Semantic Question Matcher")
# ... (the rest of your code remains unchanged)
# ...
st.markdown("""
Paste your **questions below**. This app supports:
- ✅ One-liner questions (line-by-line)
- ✅ Multi-line/code questions separated by `---` (you add it manually)

The app will automatically choose how to split based on your input.
""")

input_text = st.text_area("✍️ Enter your questions:", height=500)

similarity_threshold = st.slider(
    "🔍 Similarity Score Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.01
)

if st.button("🚀 Analyze Similar Questions", use_container_width=True, type="primary"):
    # --- QUESTION SPLITTING BASED ON USER FORMAT ---
    if "---" in input_text:
        questions = [q.strip() for q in input_text.split('---') if q.strip()]
    else:
        questions = [line.strip() for line in input_text.strip().splitlines() if line.strip()]

    if len(questions) < 2:
        st.warning("Please enter at least two questions.")
    else:
        with st.spinner("Calculating semantic similarities..."):
            embeddings = get_embeddings(questions)

            if embeddings.size > 0:
                sim_matrix = cosine_similarity(embeddings)

                results = []
                for i in range(len(questions)):
                    for j in range(i + 1, len(questions)):
                        score = sim_matrix[i][j]
                        if score >= similarity_threshold:
                            results.append({
                                "Question 1": questions[i],
                                "Question 2": questions[j],
                                "Similarity Score": f"{score:.3f}"
                            })

                if results:
                    df = pd.DataFrame(results)
                    st.success(f"✅ Found {len(results)} similar question pairs")
                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="similar_questions.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No similar questions found above the threshold.")
