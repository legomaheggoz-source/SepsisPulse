---
title: SepsisPulse
emoji: 💓
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
license: mit
---

# 💓 SepsisPulse

Clinical Utility & Lead-Time Auditor for Sepsis Early Warning

Audits three sepsis prediction approaches using the PhysioNet Challenge 2019 dataset:
- qSOFA (baseline heuristic)
- XGBoost-TS (gradient boosted trees)
- TFT-Lite (lightweight transformer)

Primary metric: Clinical Utility Score (rewards early detection 6hrs before onset)
