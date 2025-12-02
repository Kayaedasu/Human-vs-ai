# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 00:42:47 2025

@author: edasu
"""

import requests
from bs4 import BeautifulSoup
import csv
import time

def clean_text(t):
    t = t.replace("\n", " ")
    t = t.replace("\r", " ")
    t = t.replace("  ", " ")
    t = t.replace("Abstract: ", "")
    return t.strip()

def fetch_arxiv_abstracts(query="computer science", total=3000):
    abstracts = []
    size = 200           # arXiv sayfa başı max 200 sonuç
    pages = total // size

    for p in range(pages):
        start = p * size
        url = (
            f"https://arxiv.org/search/?query={query}"
            f"&searchtype=all&abstracts=show&order=-announced_date_first"
            f"&size={size}&start={start}"
        )

        print(f"📌 {p+1}. sayfa çekiliyor...")
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")

        results = soup.find_all("li", class_="arxiv-result")

        for item in results:
            abs_tag = item.find("span", class_="abstract-full")
            if abs_tag:
                abstract = clean_text(abs_tag.text)
                abstracts.append(abstract)

        time.sleep(1)

    return abstracts


# 3000 temiz human abstract çek
abstracts = fetch_arxiv_abstracts(total=3000)

# CSV'ye kaydet
with open("human_abstracts_clean_3000.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "text", "label"])

    for i, text in enumerate(abstracts):
        writer.writerow([i+1, text, "human"])

print("✔ 3000 temiz human abstract 'human_abstracts_clean_3000.csv' dosyasına kaydedildi!")
