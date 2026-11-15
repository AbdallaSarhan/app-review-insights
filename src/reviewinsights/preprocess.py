"""Text cleaning.

Two different consumers need two different texts:

* Sentiment models (VADER, TextBlob) need the RAW review. VADER in particular
  uses capitalisation, punctuation, emoji and negation ("not good") as signal.
* Topic models need a compact bag of content words, so for them we lowercase,
  drop stopwords and lemmatise.

`legacy_clean` reproduces the pipeline from the first version of this project so
the notebook can measure how much it hurt sentiment accuracy.
"""
from __future__ import annotations

import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

# Words that appear in nearly every review here and carry no topic signal.
DOMAIN_STOPWORDS = {
    "app", "apps", "application", "use", "using", "used",
    "would", "could", "also", "get", "one", "really", "like", "please", "thing",
    "thank", "thanks", "im", "ive", "dont", "doesnt", "cant", "wont", "didnt",
}


def ensure_nltk_data() -> None:
    for pkg, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    ensure_nltk_data()
    base = set(stopwords.words("english"))
    # Contractions lose their apostrophe in our tokens, so add the bare forms too.
    base |= {w.replace("'", "") for w in base}
    return frozenset(base | DOMAIN_STOPWORDS)


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_data()
    return WordNetLemmatizer()


def tokens_for_topics(text: str, min_len: int = 3) -> list[str]:
    """Lowercased, lemmatised content words. Numbers are dropped, not spelled out."""
    if not isinstance(text, str):
        return []
    text = URL_RE.sub(" ", text.lower())
    lemmatize = _lemmatizer().lemmatize
    stop = _stopwords()
    out = []
    for tok in TOKEN_RE.findall(text):
        tok = tok.replace("'", "")
        if tok in stop:
            continue
        lemma = lemmatize(tok)
        if len(lemma) >= min_len and lemma not in stop:
            out.append(lemma)
    return out


def legacy_clean(text: str) -> str:
    """The original preprocessing (punctuation, emoji, case and stopwords all removed).

    Kept only to quantify its effect on sentiment scoring. Do not use for analysis.
    """
    if not isinstance(text, str):
        return ""
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    stop = set(stopwords.words("english"))
    lemmatize = _lemmatizer().lemmatize
    return " ".join(w for w in (lemmatize(t) for t in text.split()) if w not in stop)
