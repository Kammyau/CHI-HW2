# -*- coding: utf-8 -*-
"""對《紅樓夢》黛玉葬花段落進行情感預測：載入已訓練模型，逐句預測並輸出結果"""
import os
import re

import jieba
import joblib
import pandas as pd

MODEL_PATH = 'output/sentiment_model.joblib'
TEXT_PATH = 'data/analysis_text.txt'
OUT_PATH = 'output/analysis_predictions.csv'

os.makedirs('output', exist_ok=True)

# 1. 載入已訓練好的模型（dict：{'model', 'vectorizer'}）
bundle = joblib.load(MODEL_PATH)
model = bundle['model']
vectorizer = bundle['vectorizer']

# 2. 讀取段落文字
with open(TEXT_PATH, encoding='utf-8') as f:
    lines = [line.strip().strip('　').strip() for line in f]
lines = [line for line in lines if line]

# 3. 切句：依斷行，再以 。！？；… 等標點切分
SENT_END = r'。！？!?；;…'
sentences = []
for line in lines:
    for sent in re.findall(rf'[^{SENT_END}]+(?:[{SENT_END}]+|$)', line):
        sent = sent.strip()
        # 過濾掉只剩標點或太短的碎片
        if len(re.sub(rf'[{SENT_END}，「」『』：:\s、…]', '', sent)) >= 2:
            sentences.append(sent)

# 4. 逐句預測（與訓練時相同：jieba 分詞 → TF-IDF → LogisticRegression）
texts = [' '.join(jieba.cut(s)) for s in sentences]
X = vectorizer.transform(texts)
labels = model.predict(X)
probas = model.predict_proba(X)
pos_idx = list(model.classes_).index('positive')
pos_probs = probas[:, pos_idx]

# 5. 儲存 CSV
df = pd.DataFrame({
    'text': sentences,
    'predicted_label': labels,
    'positive_prob': pos_probs,
})
df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
print(f'共 {len(df)} 句，已儲存至 {OUT_PATH}\n')

# 6. 印出所有預測結果
for i, (text, label, prob) in enumerate(
        zip(sentences, labels, pos_probs), start=1):
    print(f'{i:02d} [{label:>8s}] positive_prob={prob:.4f}  {text}')
