# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 18:25:35 2025

@author: edasu
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import csv
import time

# Modeli yükle
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate_summary(text):
    """Bir başlık için yeni akademik özet üretir."""
    prompt = f"Generate a new scientific abstract for this title:\n{text}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    output = model.generate(
        **inputs,
        max_length=180,
        temperature=0.7
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# -------------------------
#      ANA ÇALIŞMA
# -------------------------

df = pd.read_csv("human_abstracts_clean_3000.csv")
titles = df["text"].str[:150].tolist()

ai_outputs = []

print("🚀 3000 AI özeti oluşturuluyor...\n")

for i, title in enumerate(titles):
    try:
        summary = generate_summary(title)
        ai_outputs.append([i+1, summary, "ai"])
        print(f"{i+1}/3000 tamamlandı")
    except:
        print("⚠ Model yanıt vermedi, tekrar…")
        time.sleep(1)

# Kaydet
with open("ai_abstracts_3000_t5.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "text", "label"])
    w.writerows(ai_outputs)

print("\n✔ AI özetleri başarıyla kaydedildi!")

