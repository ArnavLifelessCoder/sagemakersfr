"""Multilingual cross-encoder pair scorer (runs on a GPU box).

    pip install torch transformers pandas pyarrow
    python cross_encoder.py train --pairs pairs_train.parquet --out ce_model [--model xlm-roberta-base] [--epochs 1]
    python cross_encoder.py score --pairs pairs_test.parquet --model-dir ce_model --out ce_scores.parquet

Input parquet columns: text_a (S1 "name | address | country"), text_b (record), y (train only), is_val (train only),
s1_id, q_id. Output for score: s1_id, q_id, ce (probability of a match).
The model licence must be MIT or Apache-2.0 (xlm-roberta-base and microsoft/mdeberta-v3-base are MIT).
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


class Pairs(Dataset):
    def __init__(self, df, tok, max_len):
        self.a = df.text_a.values
        self.b = df.text_b.values
        self.y = df.y.values.astype(np.float32) if "y" in df else None
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], (self.y[i] if self.y is not None else 0.0)


def collate(tok, max_len):
    def f(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y)
        return enc
    return f


def train(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)
    df = pd.read_parquet(a.pairs)
    tr = df[~df.is_val] if "is_val" in df else df
    va = df[df.is_val] if "is_val" in df else df.iloc[:0]
    if a.max_train and len(tr) > a.max_train:
        tr = tr.sample(a.max_train, random_state=0)
    print(f"train {len(tr)} val {len(va)} pos rate {tr.y.mean():.3f} device {dev}", flush=True)
    dl = DataLoader(Pairs(tr, tok, a.max_len), batch_size=a.bs, shuffle=True, collate_fn=collate(tok, a.max_len), num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(dl)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    model.train()
    step = 0
    for ep in range(a.epochs):
        for batch in dl:
            labels = batch.pop("labels").to(dev)
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            if step % 200 == 0:
                print(f"[{time.time() - t0:7.0f}s] ep {ep} step {step}/{steps} loss {loss.item():.4f}", flush=True)
        os.makedirs(a.out, exist_ok=True)
        model.save_pretrained(a.out)
        tok.save_pretrained(a.out)
        if len(va):
            p = predict(model, tok, va, a.bs * 2, a.max_len, dev)
            y = va.y.values
            ll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
            acc = np.mean((p >= 0.5) == (y == 1))
            print(f"epoch {ep}: val logloss {ll:.4f} acc {acc:.4f}", flush=True)
            model.train()


@torch.no_grad()
def predict(model, tok, df, bs, max_len, dev):
    model.eval()
    dl = DataLoader(Pairs(df, tok, max_len), batch_size=bs, shuffle=False, collate_fn=collate(tok, max_len), num_workers=2)
    out = []
    for batch in dl:
        batch.pop("labels")
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(**batch).logits.squeeze(-1)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def score(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_dir).to(dev)
    df = pd.read_parquet(a.pairs)
    t0 = time.time()
    parts = []
    step = 500_000
    for st in range(0, len(df), step):
        sl = df.iloc[st:st + step]
        p = predict(model, tok, sl, a.bs * 2, a.max_len, dev)
        parts.append(pd.DataFrame({"s1_id": sl.s1_id.values, "q_id": sl.q_id.values, "ce": p.astype(np.float32)}))
        print(f"[{time.time() - t0:7.0f}s] scored {min(st + step, len(df))}/{len(df)}", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(a.out, index=False)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "score"])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-train", type=int, default=1_500_000)
    a = ap.parse_args()
    (train if a.mode == "train" else score)(a)


if __name__ == "__main__":
    main()
