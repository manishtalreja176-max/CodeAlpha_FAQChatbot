# CodeAlpha Task 2 - FAQ Chatbot

Matches a user's question to the closest FAQ.

**Pipeline:** lowercase -> clean -> tokenize -> stop-word removal -> lemmatize (NLTK)
-> TF-IDF (unigrams + bigrams) -> cosine similarity -> best answer
(with a confidence threshold fallback).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py        # chat UI
python chatbot.py           # terminal version
```
Edit `faqs.json` to use your own topic or product.
