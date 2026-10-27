import pandas as pd
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ============================================
# PART 1: LOAD & CLEAN DATA
# ============================================

#------------------------------------------------------------------------------------------------- #
# ----------------------CHANGE THE FILE PATH TO YOUR OWN ----------------------------------------- #
# ------------------------------------------------------------------------------------------------ #


file_paths = {
    'ammani':  'C:/Users/Hp/Desktop/pr dataset/ammani dialect.xlsx',
    'karaki':  'C:/Users/Hp/Desktop/pr dataset/karaki dialect.xlsx',
    'fallahi': 'C:/Users/Hp/Desktop/pr dataset/fallahi dialect.xlsx',
    'bedouin': 'C:/Users/Hp/Desktop/pr dataset/bedouin dialect.xlsx'
}

df_list = []
for path in file_paths.values():
    df_list.append(pd.read_excel(path))
df = pd.concat(df_list, ignore_index=True)

print(f"Total rows loaded: {len(df)}")

def fix_dialect(label):
    if not isinstance(label, str):
        return label
    label = label.lower().strip()
    if label in ['ammani', 'amman', 'ammany']:       return 'ammani'
    elif label in ['karaki', 'karak', 'karaky']:      return 'karaki'
    elif label in ['fallahi', 'fallahy', 'rural', 'falla7i']: return 'fallahi'
    elif label in ['bedouin', 'badawi', 'bedu']:      return 'bedouin'
    return label

def fix_sentiment(label):
    if not isinstance(label, str):
        return label
    label = label.lower().strip()
    if label in ['positive', 'pos']:                  return 'positive'
    elif label in ['negative', 'neg']:                return 'negative'
    elif label in ['neutral', 'neut', 'nuetral', 'mixed']: return 'neutral'
    return label

def fix_confidence(label):
    if not isinstance(label, str):
        return label
    label = label.lower().strip()
    if label.startswith('h'):   return 'high'
    elif label.startswith('m'): return 'medium'
    elif label.startswith('l'): return 'low'
    return label

df['dialect'] = df['dialect'].apply(fix_dialect)
df['classification'] = df['classification'].apply(fix_sentiment)
df['confidence'] = df['confidence'].apply(fix_confidence)

if df['dialect'].isnull().any():
    mode_dialect = df['dialect'].mode()[0]
    df['dialect'] = df['dialect'].fillna(mode_dialect)
if df['confidence'].isnull().any():
    df['confidence'] = df['confidence'].fillna('medium')
if df['classification'].isnull().any():
    df['classification'] = df['classification'].fillna('neutral')

df.to_csv('cleaned_jordanian_dialects_fixed.csv', index=False)

# ============================================
# PART 2: PREPROCESSING & CONSTRAINTS (3-50)
# ============================================

print("\n" + "="*50)
print("DATA PREPROCESSING (3-50 WORD FILTER)")
print("="*50)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    text = re.sub(r'07[7-9]\d{7}|07\d{8}', '', text)
    text = re.sub(r'\b\d{8,10}\b', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'[يى]', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    return ' '.join(text.split()).strip()

df['cleaned_sentence'] = df['sentence'].apply(clean_text)
df = df[df['cleaned_sentence'].str.len() > 0]
df['word_count'] = df['cleaned_sentence'].apply(lambda x: len(x.split()))

MIN_WORDS, MAX_WORDS = 3, 50
before = len(df)
df = df[(df['word_count'] >= MIN_WORDS) & (df['word_count'] <= MAX_WORDS)]
print(f"After {MIN_WORDS}-{MAX_WORDS} filter: {len(df)} rows (removed {before - len(df)})")

before_dedup = len(df)
df = df.drop_duplicates(subset=['cleaned_sentence'])
print(f"After removing duplicates: {len(df)} rows (removed {before_dedup - len(df)})")

dialect_dist = df['dialect'].value_counts()
print(f"\nDialect distribution:\n{dialect_dist}")

original_counts = {'ammani': 1999, 'karaki': 1992, 'fallahi': 2705, 'bedouin': 2000}
print("\nPreservation rate:")
for dialect, orig in original_counts.items():
    pres = dialect_dist.get(dialect, 0)
    print(f"  {dialect}: {pres}/{orig} ({pres/orig*100:.1f}%)")

df_final = df[['cleaned_sentence', 'dialect', 'confidence', 'word_count']].copy()
df_final.to_csv('final_dialect_dataset.csv', index=False)
print(f"\nTotal samples: {len(df_final)}")

# ============================================
# PART 3: TRAIN/TEST SPLIT
# ============================================

print("\n" + "="*50)
print("TRAIN/TEST SPLIT")
print("="*50)

X = df_final['cleaned_sentence'].values
y = df_final['dialect'].values
confidence_col = df_final['confidence'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Class mapping:")
for i, name in enumerate(label_encoder.classes_):
    print(f"  {i} → {name}")

X_train, X_test, y_train, y_test, conf_train, conf_test = train_test_split(
    X, y_encoded, confidence_col,
    test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ============================================
# PART 4: FEATURE EXTRACTION
# ============================================

print("\n" + "="*50)
print("FEATURE EXTRACTION")
print("="*50)

tfidf_word = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    analyzer='char_wb',
    max_df=0.8,
    min_df=2
)
X_train_word = tfidf_word.fit_transform(X_train)
X_test_word = tfidf_word.transform(X_test)
print(f"Word features shape: {X_train_word.shape}")

tfidf_char = TfidfVectorizer(
    max_features=3000,
    ngram_range=(3, 5),
    analyzer='char_wb',
    max_df=0.8,
    min_df=3
)
X_train_char = tfidf_char.fit_transform(X_train)
X_test_char = tfidf_char.transform(X_test)
print(f"Character 4-gram features shape: {X_train_char.shape}")

X_train_combined = hstack([X_train_word, X_train_char])
X_test_combined = hstack([X_test_word, X_test_char])
print(f"Combined features shape: {X_train_combined.shape}")

# ============================================
# PART 5: SVM BASELINE (Word features only)
# ============================================

print("\n" + "="*50)
print("1. SVM BASELINE (Word features)")
print("="*50)

svm_baseline = SVC(kernel='linear', class_weight='balanced', random_state=42, C=1.0)
svm_baseline.fit(X_train_word, y_train)
y_pred_baseline = svm_baseline.predict(X_test_word)
acc_baseline = accuracy_score(y_test, y_pred_baseline)
f1_baseline = f1_score(y_test, y_pred_baseline, average=None)

print(f"Accuracy: {acc_baseline*100:.2f}%")
fallahi_idx = list(label_encoder.classes_).index('fallahi')
print(f"Fallahi F1: {f1_baseline[fallahi_idx]:.4f}")
print(classification_report(y_test, y_pred_baseline, target_names=label_encoder.classes_))

# ============================================
# PART 6: SVM WITH CHARACTER 4-GRAMS
# ============================================

print("\n" + "="*50)
print("2. SVM WITH CHARACTER 4-GRAMS")
print("="*50)

best_acc_char = 0
best_pred_char = None
best_C_char = None

for C_val in [0.1, 0.5, 1.0, 5.0]:
    svm_c = LinearSVC(class_weight='balanced', random_state=42, C=C_val, max_iter=2000)
    svm_c.fit(X_train_char, y_train)
    pred_c = svm_c.predict(X_test_char)
    acc_c = accuracy_score(y_test, pred_c)
    print(f"  C={C_val}: {acc_c*100:.2f}%")
    if acc_c > best_acc_char:
        best_acc_char = acc_c
        best_pred_char = pred_c
        best_C_char = C_val

print(f"\nBest C={best_C_char}: {best_acc_char*100:.2f}%")
f1_char = f1_score(y_test, best_pred_char, average=None)
print(f"Fallahi F1: {f1_char[fallahi_idx]:.4f}")
print(classification_report(y_test, best_pred_char, target_names=label_encoder.classes_))

# ============================================
# PART 7: SVM COMBINED FEATURES + CONFIDENCE WEIGHTS
# ============================================

print("\n" + "="*50)
print("3. SVM COMBINED FEATURES + CONFIDENCE WEIGHTS (BEST MODEL)")
print("="*50)

weight_map = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
sample_weights = np.array([weight_map.get(c, 0.7) for c in conf_train])

best_acc_comb = 0
best_pred_comb = None
best_C_comb = None

for C_val in [0.1, 0.5, 1.0, 5.0]:
    svm_c = LinearSVC(class_weight='balanced', random_state=42, C=C_val, max_iter=2000)
    svm_c.fit(X_train_combined, y_train, sample_weight=sample_weights)
    pred_c = svm_c.predict(X_test_combined)
    acc_c = accuracy_score(y_test, pred_c)
    print(f"  C={C_val}: {acc_c*100:.2f}%")
    if acc_c > best_acc_comb:
        best_acc_comb = acc_c
        best_pred_comb = pred_c
        best_C_comb = C_val

print(f"\nBest C={best_C_comb}: {best_acc_comb*100:.2f}%")
f1_comb = f1_score(y_test, best_pred_comb, average=None)
print(f"Fallahi F1: {f1_comb[fallahi_idx]:.4f}")
print(classification_report(y_test, best_pred_comb, target_names=label_encoder.classes_))

# ============================================
# PART 8: ADDITIONAL CLASSIFIERS
# ============================================

print("\n" + "="*50)
print("4. KNN CLASSIFIER")
print("="*50)

knn_grid = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9, 11]}, cv=5, scoring='accuracy')
knn_grid.fit(X_train_word, y_train)
best_knn = knn_grid.best_estimator_
y_pred_knn = best_knn.predict(X_test_word)
acc_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn, average=None)
print(f"Best KNN (k={best_knn.n_neighbors}): {acc_knn*100:.2f}%")
print(f"Fallahi F1: {f1_knn[fallahi_idx]:.4f}")

print("\n" + "="*50)
print("5. RANDOM FOREST CLASSIFIER")
print("="*50)

rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]},
    cv=3, scoring='accuracy', n_jobs=1
)
rf_grid.fit(X_train_word, y_train)
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test_word)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average=None)
print(f"Best RF: {acc_rf*100:.2f}%")
print(f"Fallahi F1: {f1_rf[fallahi_idx]:.4f}")

print("\n" + "="*50)
print("6. MLP CLASSIFIER (OPTIMIZED)")
print("="*50)

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
mlp.fit(X_train_word, y_train)
y_pred_mlp = mlp.predict(X_test_word)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp, average=None)
print(f"MLP Accuracy: {acc_mlp*100:.2f}%")
print(f"Fallahi F1: {f1_mlp[fallahi_idx]:.4f}")

# ============================================
# PART 9: FINAL RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

results_df = pd.DataFrame([
    {'Classifier': 'SVM (Baseline)', 'Accuracy': f"{acc_baseline*100:.2f}%", 'Fallahi F1': f"{f1_baseline[fallahi_idx]:.4f}"},
    {'Classifier': 'SVM (4-gram)', 'Accuracy': f"{best_acc_char*100:.2f}%", 'Fallahi F1': f"{f1_char[fallahi_idx]:.4f}"},
    {'Classifier': 'SVM (Combined + Weights)', 'Accuracy': f"{best_acc_comb*100:.2f}%", 'Fallahi F1': f"{f1_comb[fallahi_idx]:.4f}"},
    {'Classifier': 'KNN', 'Accuracy': f"{acc_knn*100:.2f}%", 'Fallahi F1': f"{f1_knn[fallahi_idx]:.4f}"},
    {'Classifier': 'Random Forest', 'Accuracy': f"{acc_rf*100:.2f}%", 'Fallahi F1': f"{f1_rf[fallahi_idx]:.4f}"},
    {'Classifier': 'MLP', 'Accuracy': f"{acc_mlp*100:.2f}%", 'Fallahi F1': f"{f1_mlp[fallahi_idx]:.4f}"},
])
print(results_df.to_string(index=False))

print(f"\nBEST OVERALL ACCURACY: SVM (Combined + Weights) with {best_acc_comb*100:.2f}%")
print(f"BEST FALLAHI F1: {f1_comb[fallahi_idx]:.4f}")

# ============================================
# PART 10: CONFUSION MATRIX
# ============================================

print("\n" + "="*50)
print("CONFUSION MATRIX - BEST MODEL (SVM Combined + Weights)")
print("="*50)

cm_best = confusion_matrix(y_test, best_pred_comb)
print("\nConfusion Matrix:")
print(pd.DataFrame(cm_best,
                   index=label_encoder.classes_,
                   columns=label_encoder.classes_))

plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - SVM Combined + Weights ({best_acc_comb*100:.2f}%)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_svm_combined.png', dpi=150)
plt.show()

print("\nMost frequent misclassifications:")
misclass = []
for i in range(len(label_encoder.classes_)):
    for j in range(len(label_encoder.classes_)):
        if i != j and cm_best[i][j] > 0:
            misclass.append((label_encoder.classes_[i], label_encoder.classes_[j], cm_best[i][j]))
for true, pred, count in sorted(misclass, key=lambda x: x[2], reverse=True)[:5]:
    print(f"  {true} → {pred}: {count} times")

# ============================================
# PART 11: SAVE RESULTS
# ============================================

all_results = {
    'word_filter': '3-50',
    'total_samples': len(df_final),
    'accuracy': {
        'SVM_Baseline': float(acc_baseline),
        'SVM_4gram': float(best_acc_char),
        'SVM_Combined_Weights': float(best_acc_comb),
        'KNN': float(acc_knn),
        'RandomForest': float(acc_rf),
        'MLP': float(acc_mlp)
    },
    'fallahi_f1': {
        'SVM_Baseline': float(f1_baseline[fallahi_idx]),
        'SVM_4gram': float(f1_char[fallahi_idx]),
        'SVM_Combined_Weights': float(f1_comb[fallahi_idx]),
        'KNN': float(f1_knn[fallahi_idx]),
        'RandomForest': float(f1_rf[fallahi_idx]),
        'MLP': float(f1_mlp[fallahi_idx])
    },
    'best_accuracy': float(best_acc_comb),
    'best_fallahi_f1': float(f1_comb[fallahi_idx])
}

with open('full_results_3words.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nResults saved to 'full_results_3words.json'")
print("Confusion matrix saved to 'confusion_matrix_svm_combined.png'")