"""LDA topic modelling with the number of topics chosen by coherence, not by hand."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phrases

from .config import SEED


@dataclass
class TopicModel:
    lda: LdaModel
    dictionary: Dictionary
    corpus: list
    docs: list[list[str]]
    doc_index: list[int]  # position of each modelled doc in the input list
    coherence_by_k: dict[int, float]

    @property
    def k(self) -> int:
        return self.lda.num_topics

    def top_words(self, n: int = 8) -> pd.DataFrame:
        rows = [{"topic": t, "words": ", ".join(w for w, _ in self.lda.show_topic(t, n))}
                for t in range(self.k)]
        return pd.DataFrame(rows).set_index("topic")

    def dominant_topic(self) -> pd.DataFrame:
        """Most likely topic per document and how strongly it dominates."""
        out = []
        for bow in self.corpus:
            dist = self.lda.get_document_topics(bow, minimum_probability=0)
            topic, weight = max(dist, key=lambda tw: tw[1])
            out.append((topic, weight))
        return pd.DataFrame(out, columns=["topic", "weight"], index=self.doc_index)


def add_bigrams(docs: list[list[str]], min_count: int = 10) -> list[list[str]]:
    """Join frequent pairs such as `free_trial` or `customer_service` into one token."""
    phrases = Phrases(docs, min_count=min_count, threshold=10, connector_words=ENGLISH_CONNECTOR_WORDS)
    return [phrases[d] for d in docs]


def fit(docs: list[list[str]], k_values=range(4, 13), passes: int = 15) -> TopicModel:
    """Fit one LDA per k and keep the one with the highest c_v coherence."""
    index = [i for i, d in enumerate(docs) if len(d) >= 3]
    docs = add_bigrams([docs[i] for i in index])
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    keep = [j for j, d in enumerate(docs) if len(dictionary.doc2bow(d)) >= 2]
    docs = [docs[j] for j in keep]
    index = [index[j] for j in keep]
    corpus = [dictionary.doc2bow(d) for d in docs]

    best, scores = None, {}
    for k in k_values:
        lda = LdaModel(corpus, num_topics=k, id2word=dictionary, passes=passes,
                       alpha="auto", eta="auto", random_state=SEED)
        scores[k] = CoherenceModel(model=lda, texts=docs, dictionary=dictionary,
                                   coherence="c_v", processes=1).get_coherence()
        if best is None or scores[k] > scores[best.num_topics]:
            best = lda
    return TopicModel(best, dictionary, corpus, docs, index, scores)
