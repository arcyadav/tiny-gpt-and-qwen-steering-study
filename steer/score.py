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
import torch.nn.functional as F

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


@torch.no_grad()
def continuation_nll(steerer, prompts, continuations, batch_size=16):
    """Mean negative log-likelihood per token of each continuation given its prompt,
    under the unsteered model."""
    tok, model = steerer.tok, steerer.model
    nlls = []
    for i in range(0, len(prompts), batch_size):
        rows = []
        for p, c in zip(prompts[i : i + batch_size], continuations[i : i + batch_size]):
            p_ids = tok(p).input_ids
            c_ids = tok(" " + c).input_ids if c else []
            rows.append((p_ids + c_ids, len(p_ids)))
        width = max(len(ids) for ids, _ in rows)
        input_ids = torch.full((len(rows), width), tok.pad_token_id)
        attn = torch.zeros((len(rows), width), dtype=torch.long)
        targets = torch.full((len(rows), width), -100)  # -100 = ignored by cross_entropy
        for r, (ids, n_prompt) in enumerate(rows):
            input_ids[r, : len(ids)] = torch.tensor(ids)
            attn[r, : len(ids)] = 1
            targets[r, n_prompt : len(ids)] = torch.tensor(ids[n_prompt:])
        logits = model(input_ids=input_ids.to(steerer.device), attention_mask=attn.to(steerer.device)).logits.float()
        # position t predicts token t+1
        loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), targets[:, 1:].to(steerer.device),
                               reduction="none", ignore_index=-100)
        n_tokens = (targets[:, 1:] != -100).sum(1).clamp(min=1).to(steerer.device)
        nlls += (loss.sum(1) / n_tokens).tolist()
    return nlls


def distinct_2(text):
    words = text.split()
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def score(steerer, sentiment, prompts, continuations):
    """Per-text scores plus a summary for one experimental setting."""
    sent = sentiment(continuations)
    nll = continuation_nll(steerer, prompts, continuations)
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
