# Titanic-ML-Project
My first experience with data analysis<br><br>
Kaggle Titanic Survival Prediction (Logistic Regression model)
- **Submission data**(https://github.com/k-ohyeah/Titanic-ML-Project/blob/main/titanic-competition-submission2.ipynb)
- **Learning data**
  - **Data Visualization**(https://github.com/k-ohyeah/Titanic-ML-Project/blob/main/titanic-competition.ipynb)
  - **Data Preprocessing and Modeling**(https://github.com/k-ohyeah/Titanic-ML-Project/blob/main/titanic-competition-base-model.ipynb)
  
# 🚢 Titanic - 生存予測モデル構築プロジェクト

Kaggle Titanic データセットを用いた生存予測モデル構築。  
ロジスティック回帰を中心に、特徴量設計と前処理を通じて予測精度の向上を検証。  
最終スコアは **0.76794 (Submission)**、ROC AUC **0.869** を達成。

---

## 🎯 目的

Titanic データセットを題材に、基本的な機械学習プロセスの理解および  
特徴量エンジニアリング・モデルチューニングの実践を目的とした。  
単純な分類問題を通じ、分析設計から評価までの一連の流れを体系的に整理。

---

## ⚙️ データ前処理

- 欠損値の補完（`Age`、`Embarked`、`Fare`）  
- カテゴリ変数のラベルエンコーディングおよびOne-Hot化  
- 新規特徴量の導入  
  - `Title`（氏名から抽出した敬称）  
  - `FamilySize`（家族人数）  
  - `IsAlone`（単身フラグ）  
  - `Fare_per_person`（1人当たり運賃）  

💡 **気づき:**  
単なる欠損補完や正規化ではなく、**意味のある特徴量の構築が精度向上に寄与**することを確認。  
特に `Title` と `FamilySize` は生存率に強く影響する特徴量として機能。

---

## 🧩 モデル構築

ロジスティック回帰を中心としたベースモデルを構築。  
GridSearchCVを用いたハイパーパラメータ最適化を実施。

| モデル | 特徴 | パラメータ |
|--------|------|------------|
| **Logistic Regression** | シンプルかつ解釈性が高い | `C`, `penalty` を GridSearchCV で探索 |
| **Validation Split** | 80/20 分割 | ROC AUC による評価 |

💡 **気づき:**  
パラメータ探索により汎化性能が向上。  
ロジスティック回帰のような単純モデルでも、**適切な特徴量設計と正則化設定により高い安定性を発揮**。

---

## 📊 結果

| 指標 | 値 |
|------|------|
| ROC AUC | **0.869** |
| Kaggle Submission Score | **0.76794** |

💡 **気づき:**  
単一モデルでも適切な特徴量エンジニアリングにより、  
Kaggle初学者水準を上回るスコアを達成。  
モデル選択よりも、前処理と特徴量構築の重要性を再認識。

---

## 💬 取り組み方

分析は ChatGPT および GitHub Copilot の提案を参考に進行。  
各ステップを独自に再現・検証し、結果の妥当性を確認。  
提案コードをそのまま適用せず、動作意図を把握しながら修正を加えることで理解を深化。

---

## 🧾 Summary

| 項目 | 内容 |
|------|------|
| モデル | Logistic Regression |
| 主な特徴量 | Title, FamilySize, IsAlone, Fare_per_person |
| 最適化手法 | GridSearchCV |
| 評価指標 | ROC AUC, Kaggle Submission |
| 最終スコア | ROC AUC 0.869 / Submission 0.76794 |

---

💡 **学習の要約**  
- 特徴量設計がモデル精度を決定づける要素であることを確認。  
- モデルチューニングよりも前処理・データ理解の重要性を認識。  
- 自動生成コードを参考にしつつ、自ら検証する姿勢を徹底。  
- データサイエンスの基礎工程を一通り再現し、分析プロセス理解を深化。

---
---
# (English ver)
# 🚢 Titanic Survival Prediction (Kaggle)

A beginner-friendly machine learning project using the Titanic dataset.

## 🎯 Objective
Predict passenger survival using logistic regression and feature engineering.

## 📊 Key Features
- Feature engineering with `Title`, `FamilySize`, `IsAlone`, `Fare_per_person`
- Missing value imputation by Title
- Model tuning via `GridSearchCV`
- Final ROC AUC: **0.869**, Submission Score: **0.76794**

## 🧩 Tech Stack
- Python (pandas, numpy, scikit-learn, matplotlib)
- Jupyter Notebook
- GitHub for version control

## 🧠 Learning Outcome
- Gained understanding of end-to-end ML workflow
- Improved feature engineering intuition
- Experienced hyperparameter tuning

---
⭐️ This project was completed as part of a 7-day Kaggle learning plan.
