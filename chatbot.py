"""FAQ matching engine: NLTK preprocessing + TF-IDF + cosine similarity."""
import json
import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_tokenizer = RegexpTokenizer(r"[a-z0-9]+")
_lemma = WordNetLemmatizer()
_stop = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Lowercase -> clean -> tokenize -> remove stopwords -> lemmatize."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = [t for t in _tokenizer.tokenize(text) if t not in _stop]
    return " ".join(_lemma.lemmatize(t) for t in tokens)


class FAQBot:
    def __init__(self, faq_path: str = None, threshold: float = 0.30):
        path = Path(faq_path or Path(__file__).with_name("faqs.json"))
        self.faqs = json.loads(path.read_text(encoding="utf-8"))
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(preprocess(f["question"]) for f in self.faqs)

    def ask(self, user_text: str):
        """Return (answer, matched_question, score)."""
        vec = self.vectorizer.transform([preprocess(user_text)])
        scores = cosine_similarity(vec, self.matrix)[0]
        best = int(scores.argmax())
        if scores[best] < self.threshold:
            return ("Sorry, I couldn't find a good answer. Try rephrasing, or contact support.", None, float(scores[best]))
        return self.faqs[best]["answer"], self.faqs[best]["question"], float(scores[best])


if __name__ == "__main__":  # simple terminal chat
    bot = FAQBot()
    print("FAQ Bot (type 'quit' to exit)")
    while (q := input("You: ").strip().lower()) not in {"quit", "exit"}:
        print("Bot:", bot.ask(q)[0])
