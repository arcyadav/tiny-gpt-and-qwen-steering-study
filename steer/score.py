"""Scoring generated text: how positive is it, and is it still fluent?

- sentiment:  P(positive) - P(negative) from a 3-class (negative / neutral / positive) RoBERTa
              classifier, in [-1, 1]. A neutral class matters: 2-class SST-2 models call ordinary
              neutral text "negative", which hides the effect of steering towards sad.
- perplexity: of the continuation under the *unsteered* model (higher = less natural text)
- distinct-2: share of unique word bigrams. Needed because repetitive loops
              ("despair: despair: despair") can get a deceptively LOW perplexity.
"""
import math

import numpy as np
import torch

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class SentimentScorer:
    def __init__(self, device=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL).to(self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.pos_index, self.neg_index = labels["positive"], labels["negative"]

    @torch.no_grad()
    def __call__(self, texts, batch_size=64):
        """Sentiment score P(positive) - P(negative) for each text, in [-1, 1]."""
        probs = []
        for i in range(0, len(texts), batch_size):
            batch = self.tok([t or "." for t in texts[i : i + batch_size]], return_tensors="pt",
                             padding=True, truncation=True, max_length=128).to(self.device)
            p = self.model(**batch).logits.softmax(-1)
            probs += (p[:, self.pos_index] - p[:, self.neg_index]).tolist()
        return probs


def distinct_2(text):
    words = text.split()
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def score(steerer, sentiment, prompts, continuations):
    """Per-text scores plus a summary for one experimental setting."""
    sent = sentiment(continuations)
    nll = steerer.continuation_nll(prompts, continuations)
    d2 = [distinct_2(c) for c in continuations]
    summary = {
        "sentiment": float(np.mean(sent)),
        "frac_positive": float(np.mean([s > 0.5 for s in sent])),
        "frac_negative": float(np.mean([s < -0.5 for s in sent])),
        "perplexity": math.exp(float(np.mean(nll))),
        "distinct_2": float(np.mean(d2)),
        "n": len(continuations),
    }
    per_text = [{"prompt": p, "text": c, "sentiment": s, "nll": n, "distinct_2": d}
                for p, c, s, n, d in zip(prompts, continuations, sent, nll, d2)]
    return summary, per_text
