# 🌍 BiLingual Translator

A sequence-to-sequence (Seq2Seq) translation system using Transformer models for bidirectional translation between English and French. This project implements two models: one for English-to-French translation and another for French-to-English translation, with evaluation through back-translation and metrics like BLEU and ROUGE.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-orange.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 Overview

This project builds two Transformer-based Seq2Seq models:
- **English-to-French Model** 🇬🇧➡️🇫🇷: Translates English sentences to French.
- **French-to-English Model** 🇫🇷➡️🇬🇧: Translates French sentences to English.

The models are trained on parallel corpora and evaluated by back-translating sentences to ensure meaning preservation. Performance is assessed using BLEU and ROUGE scores.

## 🚀 Features

- **Bidirectional Translation**: Translate seamlessly between English and French.
- **Transformer Architecture**: Utilizes PyTorch's Transformer implementation for efficient sequence modeling.
- **Preprocessing**: Leverages spaCy for tokenization and torchtext for vocabulary building.
- **Evaluation Metrics**: Includes BLEU and ROUGE scores for translation quality assessment.
- **Back-Translation**: Validates translation consistency by translating back to the original language.

## 📋 Requirements

To run this project, install the following dependencies:

```bash
pip install torch==2.0.0 torchtext==0.15.1 torchdata==0.6.0
pip install portalocker>=2.0.0
pip install numpy==1.23.5
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
pip install nltk rouge-score
```
## ⚙️ Tech Stack

PyTorch (Transformer Architecture)  
spaCy (Tokenization)  
NLTK (BLEU Score)   
Rouge-Score (ROUGE Metrics) 

## 🛠️ Installation  

### Clone the repo
```bash
git clone https://github.com/yourusername/Bilingual-Transformer.git
cd Bilingual-Transformer
```
### Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

## 🧠 Model Training

```python
# Train English-to-French Model
train(model_en_fr, en_fr_loader, criterion, optimizer, epochs=10)

# Train French-to-English Model
train(model_fr_en, fr_en_loader, criterion, optimizer, epochs=10)
```
## 🔍 Evaluation Metrics

✅ Test Case 1

Input (EN): "How was the movie yesterday?"  
Output (FR): "Comment était le film hier ?"  
Back-Translated (EN): "How was the movie yesterday?"  

✅ Test Case 2  

Input (EN): "She loves chocolate."  
Output (FR): "Elle adore le chocolat."  
Back-Translated (EN): "She loves chocolate."  


## 📊 **Evaluation Metrics**
| Metric       | Sentence 1 (`How was the movie yesterday?`) | Sentence 2 (`She loves chocolate.`) |
|--------------|---------------------------------------------|-------------------------------------|
| **BLEU**     | 1.0                                         | 1.0                                 |
| **ROUGE-L**  | F1=1.0                                      | F1=1.0                              |

## 🎯 Future Improvements

### Larger Dataset (e.g., WMT14)
### Beam Search Decoding 🏹
### HuggingFace Integration 🤗


## 📌 License

This project is intended for academic and research use only.
