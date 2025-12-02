# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 18:25:48 2025

@author: edasu
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import csv
import time

model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate_summary(text):
    prompt = f"Generate a new scientific abstract for this title:\n{text}"
    inp = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    out = model.generate(**inp, max_length=180, temperature=0.7)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# Daha önce kaydedilenleri yükle
saved = pd.read_csv("ai_partial_657.csv")
ai_outputs = saved[["id", "text", "label"]].values.tolist()

df = pd.read_csv("human_abstracts_clean_3000.csv")
titles = df["text"].tolist()

start_idx = len(saved)

print("Kaldığın yerden devam ediliyor →", start_idx)

for i in range(start_idx, 3000):
    t = titles[i]
    summary = generate_summary(t)
    ai_outputs.append([i+1, summary, "ai"])
    print(f"{i+1}/3000 tamamlandı")

# Yeni kaydet
with open("ai_partial_3000.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "text", "label"])
    w.writerows(ai_outputs)

print("✔ KISMİ KAYIT TAMAMLANDI!")
