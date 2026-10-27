# JordSub: Arabic Sub-Dialect Identification

The first annotated dataset for intra-Jordanian Arabic dialect identification, plus classifiers that tell four Jordanian sub-dialects apart. Built as a Pattern Recognition course project at the University of Jordan.

## Overview

JordSub contains 8,030 sentences across four dialect classes:

- Urban (Ammani)
- Bedouin
- Rural / Fallahi
- Karaki (Southern)

I compared TF-IDF features with SVM, KNN, Random Forest, and MLP classifiers, then added contextual embeddings from three Arabic language models (AraBERT, CAMeL-BERT, and MARBERT). The best system, an SVM on fused TF-IDF and LLM embeddings, reaches **82.75% accuracy** and a **0.7807 F1 on Fallahi**, the hardest class.

## Dataset

- **Raw data:** 8,696 sentences (Ammani 1,999, Karaki 1,992, Fallahi 2,705, Bedouin 2,000)
- **Source:** public social media comments (YouTube, Twitter/X, Facebook groups, TikTok), with URLs, mentions, hashtags, emojis, and personal information removed
- **Labels:** a dialect class and an annotator confidence score (High / Medium / Low), assigned by native Jordanian speakers, with regional affiliation checked through geo-tags, profile locations, and region-specific pages
- **Preprocessing:** Arabic character normalization, word-count filtering, and deduplication

I tested three minimum word-length filters. The 5-50 filter suggested in the project guidelines removes 68% of Fallahi samples, so I adopted 3-50.

| Filter | Total samples | Fallahi kept |
|---|---|---|
| **3-50 (adopted)** | 8,030 | 80.4% |
| 4-50 | 7,233 | 55.0% |
| 5-50 | 6,172 | 32.2% |

## Repository structure

```
├── data/                                  # the four raw dialect files (.xlsx)
├── 3-50 constraint.py                     # TF-IDF models, 3-50 word filter (adopted)
├── 4-50/                                  # same pipeline with the 4-50 filter
├── 5-50/                                  # same pipeline with the 5-50 filter
├── with llm embeddings/                   # TF-IDF + AraBERT / CAMeL-BERT / MARBERT (best model)
├── final_dialect_dataset.csv              # cleaned, filtered dataset (3-50)
├── cleaned_jordanian_dialects_fixed.csv   # all four dialects combined, labels standardized
├── full_results_3words.json               # results of the 3-50 run
└── confusion_matrix_svm_combined.png
```

Each folder also holds the output files (results, confusion matrices) of its own run.

## Results (3-50 filter)

| Classifier | Accuracy | Fallahi F1 |
|---|---|---|
| SVM (baseline) | 75.53% | 0.7119 |
| SVM (character 3-5-grams) | 76.03% | 0.6959 |
| KNN | 70.86% | 0.6237 |
| Random Forest | 72.23% | 0.7191 |
| MLP | 74.28% | 0.6857 |
| SVM (combined TF-IDF + confidence weights) | 79.20% | 0.7373 |
| SVM (AraBERT) | 76.40% | 0.7281 |
| SVM (CAMeL-BERT) | 78.33% | 0.7247 |
| SVM (MARBERT) | 80.32% | 0.7506 |
| SVM (fused LLM embeddings) | 81.44% | 0.7703 |
| **SVM (TF-IDF + fused LLM embeddings)** | **82.75%** | **0.7807** |

Main findings:

- A uniform minimum word length hurts Fallahi most, because its sentences are shorter. The 5-50 filter gets the highest overall accuracy under TF-IDF (80.16%) but a Fallahi F1 of only 0.5476.
- Using annotator confidence as sample weights (High = 1.0, Medium = 0.7, Low = 0.4) improved accuracy by 3.67 points over the baseline.
- Fusing TF-IDF with averaged LLM embeddings gave the best result, 3.55 points above the best TF-IDF-only model.

## How to run

1. Install the libraries:
   ```
   pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl
   ```
   The embeddings script also needs `torch` and `transformers`.
2. At the top of each script, change the file paths to the location of the `data/` folder on your computer.
3. Run a script, for example:
   ```
   python "3-50 constraint.py"
   ```

The embeddings script downloads three models from Hugging Face on its first run (`aubmindlab/bert-base-arabertv02`, `CAMeL-Lab/bert-base-arabic-camelbert-da`, and `UBC-NLP/MARBERT`). It runs much faster on a GPU.

## Limitations

- Social media text may not fully represent spoken dialect
- Some regions (such as Aqaba and the northern Badia) are underrepresented
- Inter-annotator agreement was not measured, due to limited resources
- The LLM embeddings were used as fixed features, with no fine-tuning, so fine-tuning would likely improve results

## Author

Ahmad Abu Zayyad, Data Science, University of Jordan
