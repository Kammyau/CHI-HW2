# -*- coding: utf-8 -*-
"""訓練《紅樓夢》情感分類器：jieba 分詞 + TF-IDF + LogisticRegression"""
import os

import joblib
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

DATA = 'data/sentiment_balanced.csv'
OUT_DIR = 'output'

os.makedirs(OUT_DIR, exist_ok=True)

# 1. 讀取資料
df = pd.read_csv(DATA)
print(f'讀取 {DATA}：{len(df)} 筆')
print(df['label'].value_counts())

# 2. jieba 分詞
texts = [' '.join(jieba.cut(t)) for t in df['text']]
labels = df['label']

# 3. TF-IDF（最多 2000 特徵）
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(texts)

# 4. 切分資料（80% 訓練 / 20% 測試，stratified）
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, stratify=labels, random_state=42)
print(f'訓練集 {X_train.shape[0]} 筆，測試集 {X_test.shape[0]} 筆')

# 5. 訓練 LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 6. 評估
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, digits=4)
conf = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])
acc = accuracy_score(y_test, y_pred)

lines = [
    f'Accuracy: {acc:.4f}',
    '',
    'Classification Report:',
    report,
    '',
    f'Confusion Matrix (labels={["positive", "negative"]}):',
    str(conf),
    '',
    f'訓練集筆數: {X_train.shape[0]}，測試集筆數: {X_test.shape[0]}',
    f'特徵維度: {X.shape[1]}',
]
evaluation = '\n'.join(lines)
print(evaluation)

with open(os.path.join(OUT_DIR, 'evaluation.txt'), 'w', encoding='utf-8') as f:
    f.write(evaluation + '\n')

# 7. 保存模型與 vectorizer
joblib.dump({'model': model, 'vectorizer': vectorizer},
            os.path.join(OUT_DIR, 'sentiment_model.joblib'))
print(f'已儲存模型至 {OUT_DIR}/sentiment_model.joblib')
