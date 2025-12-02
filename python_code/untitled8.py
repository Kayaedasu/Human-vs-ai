# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 18:26:01 2025

@author: edasu
"""

import pandas as pd

human = pd.read_csv("human_abstracts_clean_3000.csv")
ai = pd.read_csv("ai_partial_2554.csv")

human["label"] = "human"
ai["label"] = "ai"

combined = pd.concat([human, ai], ignore_index=True)

combined = combined.sample(frac=1).reset_index(drop=True)

combined.to_csv("final_dataset_5554.csv", index=False, encoding="utf-8")

print("✔ final_dataset_5554.csv oluşturuldu!")
