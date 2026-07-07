# ============================================================
# FAKE NEWS DETECTION - KAGGLE MANUAL TESTING APPROACH
# ============================================================
# This follows the Kaggle project methodology:
# 1. Remove last 10 rows from Fake.csv & True.csv as test data
# 2. Keep rest for training
# 3. Add status column to track train/test data
# 4. Train model on training data
# 5. Evaluate on manual test data
# ============================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# STEP 1: LOAD AND SPLIT DATA MANUALLY (KAGGLE APPROACH)
# ============================================================

print("="*70)
print("KAGGLE MANUAL TESTING APPROACH")
print("="*70)

# Load fake news dataset
fake_df = pd.read_csv('Fake.csv')
print(f"\n✓ Loaded Fake.csv: {len(fake_df)} articles")

# Load real news dataset  
true_df = pd.read_csv('True.csv')
print(f"✓ Loaded True.csv: {len(true_df)} articles")

# ============================================================
# STEP 2: SPLIT - LAST 10 ROWS AS TEST DATA
# ============================================================

print("\n" + "-"*70)
print("SPLITTING DATA MANUALLY")
print("-"*70)

# Split fake news: last 10 for testing, rest for training
fake_test = fake_df.tail(10).copy()
fake_train = fake_df.iloc[:-10].copy()

# Split true news: last 10 for testing, rest for training
true_test = true_df.tail(10).copy()
true_train = true_df.iloc[:-10].copy()

print(f"\nFAKE NEWS:")
print(f"  Training: {len(fake_train)} articles")
print(f"  Testing:  {len(fake_test)} articles (last 10)")

print(f"\nTRUE NEWS:")
print(f"  Training: {len(true_train)} articles")
print(f"  Testing:  {len(true_test)} articles (last 10)")

# ============================================================
# STEP 3: ADD STATUS COLUMN
# ============================================================

# Add label column (0 = fake, 1 = real)
fake_train['label'] = 0
fake_test['label'] = 0
true_train['label'] = 1
true_test['label'] = 1

# Add status column (train or test)
fake_train['status'] = 'train'
fake_test['status'] = 'test'
true_train['status'] = 'train'
true_test['status'] = 'test'

print("\n" + "-"*70)
print("ADDED COLUMNS")
print("-"*70)
print("\nColumns in each dataset:")
print(fake_train.columns.tolist())

# Combine train and test data (separately)
train_df = pd.concat([fake_train, true_train], ignore_index=True)
test_df = pd.concat([fake_test, true_test], ignore_index=True)

print(f"\nTraining dataset combined: {len(train_df)} articles")
print(f"  - Fake (label=0): {len(fake_train)}")
print(f"  - Real (label=1): {len(true_train)}")

print(f"\nTest dataset combined: {len(test_df)} articles")
print(f"  - Fake (label=0): {len(fake_test)}")
print(f"  - Real (label=1): {len(true_test)}")

# ============================================================
# STEP 4: PREPARE TEXT DATA
# ============================================================

print("\n" + "-"*70)
print("PREPARING TEXT DATA")
print("-"*70)

# Combine title and text (if available)
# Note: adjust column names based on your CSV structure
# Common columns: 'title', 'text', 'content'

def combine_text(row):
    """Combine title and text columns"""
    title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
    text = str(row.get('text', '')) if pd.notna(row.get('text')) else ''
    
    # Use content if text not available
    if not text:
        text = str(row.get('content', '')) if pd.notna(row.get('content')) else ''
    
    combined = (title + ' ' + text).strip()
    return combined if combined else 'empty'

# Create combined text column
train_df['combined_text'] = train_df.apply(combine_text, axis=1)
test_df['combined_text'] = test_df.apply(combine_text, axis=1)

print("✓ Combined title and text for each article")
print(f"  Sample training text: {train_df['combined_text'].iloc[0][:100]}...")
print(f"  Sample test text: {test_df['combined_text'].iloc[0][:100]}...")

# ============================================================
# STEP 5: FEATURE ENGINEERING (TF-IDF)
# ============================================================

print("\n" + "-"*70)
print("FEATURE ENGINEERING - TF-IDF VECTORIZATION")
print("-"*70)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,        # Top 5000 words
    ngram_range=(1, 2),       # Unigrams + bigrams
    lowercase=True,
    stop_words='english',
    min_df=5,                 # Ignore words appearing in < 5 docs
    max_df=0.8                # Ignore words appearing in > 80% of docs
)

print("\nVectorizer configuration:")
print(f"  - Max features: 5000")
print(f"  - N-gram range: (1, 2)")
print(f"  - Stop words: English")
print(f"  - Min document frequency: 5")
print(f"  - Max document frequency: 80%")

# Fit on training data, transform both
X_train = vectorizer.fit_transform(train_df['combined_text'])
X_test = vectorizer.transform(test_df['combined_text'])

y_train = train_df['label'].values
y_test = test_df['label'].values

print(f"\n✓ Training data shape: {X_train.shape}")
print(f"  (Articles × Features)")
print(f"✓ Test data shape: {X_test.shape}")

# ============================================================
# STEP 6: TRAIN MODEL
# ============================================================

print("\n" + "-"*70)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("-"*70)

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    C=1.0  # Regularization strength
)

print("Model: Logistic Regression")
print("  - Max iterations: 1000")
print("  - Regularization (C): 1.0")

model.fit(X_train, y_train)
print("\n✓ Model trained successfully!")

# ============================================================
# STEP 7: PREDICTIONS ON TEST DATA
# ============================================================

print("\n" + "-"*70)
print("MAKING PREDICTIONS")
print("-"*70)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"✓ Predictions made on {len(y_test)} test articles")
print(f"  Confidence scores calculated")

# ============================================================
# STEP 8: EVALUATION METRICS
# ============================================================

print("\n" + "="*70)
print("MODEL EVALUATION ON TEST DATA")
print("="*70)

accuracy = accuracy_score(y_test, y_pred)
precision_fake = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
recall_fake = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
f1_fake = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

precision_real = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall_real = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1_real = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"\n{'OVERALL METRICS':^70}")
print("-"*70)
print(f"Accuracy:                    {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\n{'FAKE NEWS (Label = 0)':^70}")
print("-"*70)
print(f"Precision (True Positive Rate):  {precision_fake:.4f}")
print(f"  → Of articles flagged as FAKE, how many actually were")
print(f"Recall (Sensitivity):            {recall_fake:.4f}")
print(f"  → Of actual fake articles, how many did we catch")
print(f"F1-Score:                        {f1_fake:.4f}")
print(f"  → Balanced metric")

print(f"\n{'REAL NEWS (Label = 1)':^70}")
print("-"*70)
print(f"Precision:                   {precision_real:.4f}")
print(f"Recall:                      {recall_real:.4f}")
print(f"F1-Score:                    {f1_real:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n{'CONFUSION MATRIX':^70}")
print("-"*70)
print(f"\n                  Predicted Fake    Predicted Real")
print(f"Actual Fake:           {cm[0,0]:3d}               {cm[0,1]:3d}")
print(f"Actual Real:           {cm[1,0]:3d}               {cm[1,1]:3d}")

# Breakdown
correct_fake = cm[0, 0]
incorrect_fake = cm[0, 1]
correct_real = cm[1, 1]
incorrect_real = cm[1, 0]

print(f"\nBreakdown:")
print(f"  ✓ Correctly identified FAKE: {correct_fake}/10 ({correct_fake*10:.0f}%)")
print(f"  ✗ Incorrectly classified as REAL: {incorrect_fake}/10")
print(f"  ✓ Correctly identified REAL: {correct_real}/10 ({correct_real*10:.0f}%)")
print(f"  ✗ Incorrectly classified as FAKE: {incorrect_real}/10")

# ============================================================
# STEP 9: VISUALIZE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fake', 'Real'], 
            yticklabels=['Fake', 'Real'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 14})
plt.title('Confusion Matrix - Fake News Detection\n(Manual Test Data: Last 10 from each class)', 
          fontsize=14, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Confusion matrix saved to 'confusion_matrix.png'")

# ============================================================
# STEP 10: DETAILED TEST RESULTS WITH STATUS
# ============================================================

print("\n" + "="*70)
print("DETAILED TEST RESULTS")
print("="*70)

# Add predictions back to test dataframe
test_df['predicted_label'] = y_pred
test_df['predicted_proba_fake'] = y_pred_proba[:, 0]
test_df['predicted_proba_real'] = y_pred_proba[:, 1]

# Determine if prediction was correct
test_df['correct'] = test_df['label'] == test_df['predicted_label']

# Show results for fake news (first 5)
print(f"\nFAKE NEWS TEST RESULTS (showing first 5 of 10):")
print("-"*70)
fake_results = test_df[test_df['label'] == 0].head(5)
for idx, (_, row) in enumerate(fake_results.iterrows(), 1):
    actual = "FAKE"
    predicted = "FAKE" if row['predicted_label'] == 0 else "REAL"
    confidence = row['predicted_proba_fake'] if row['predicted_label'] == 0 else row['predicted_proba_real']
    correct = "✓" if row['correct'] else "✗"
    
    print(f"\n{idx}. {correct} Actual: {actual} → Predicted: {predicted} ({confidence*100:.1f}%)")
    print(f"   Title: {row['title'][:60]}...")

# Show results for real news (first 5)
print(f"\n\nREAL NEWS TEST RESULTS (showing first 5 of 10):")
print("-"*70)
real_results = test_df[test_df['label'] == 1].head(5)
for idx, (_, row) in enumerate(real_results.iterrows(), 1):
    actual = "REAL"
    predicted = "REAL" if row['predicted_label'] == 1 else "FAKE"
    confidence = row['predicted_proba_real'] if row['predicted_label'] == 1 else row['predicted_proba_fake']
    correct = "✓" if row['correct'] else "✗"
    
    print(f"\n{idx}. {correct} Actual: {actual} → Predicted: {predicted} ({confidence*100:.1f}%)")
    print(f"   Title: {row['title'][:60]}...")

# ============================================================
# STEP 11: MOST IMPORTANT FEATURES (WORDS)
# ============================================================

print("\n" + "="*70)
print("MOST IMPORTANT FEATURES (WORDS INDICATING FAKE VS REAL)")
print("="*70)

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Top 15 most indicative of FAKE news (most negative coefficients)
fake_indices = np.argsort(coefficients)[:15]
# Top 15 most indicative of REAL news (most positive coefficients)
real_indices = np.argsort(coefficients)[-15:]

print(f"\nTOP 15 WORDS INDICATING FAKE NEWS:")
print("-"*70)
for rank, idx in enumerate(fake_indices, 1):
    word = feature_names[idx]
    coef = coefficients[idx]
    print(f"{rank:2d}. '{word}' (coefficient: {coef:7.4f})")

print(f"\nTOP 15 WORDS INDICATING REAL NEWS:")
print("-"*70)
for rank, idx in enumerate(reversed(real_indices), 1):
    word = feature_names[idx]
    coef = coefficients[idx]
    print(f"{rank:2d}. '{word}' (coefficient: {coef:7.4f})")

# ============================================================
# STEP 12: PREDICTIONS ON NEW ARTICLES
# ============================================================

print("\n" + "="*70)
print("TESTING ON NEW ARTICLES")
print("="*70)

new_articles = {
    "Sensational Clickbait": "SHOCKING! Celebrity reveals secret you won't BELIEVE! Click here NOW!",
    "Scientific Study": "New research published in Nature journal shows correlation between variables in controlled study",
    "Fake Conspiracy": "Government EXPOSED in massive cover-up conspiracy CONFIRMED!!!",
    "Real News": "Local government announces new infrastructure project for community development"
}

print("\nMaking predictions on new articles:\n")

for article_name, text in new_articles.items():
    # Transform text
    text_tfidf = vectorizer.transform([text])
    
    # Predict
    pred = model.predict(text_tfidf)[0]
    proba = model.predict_proba(text_tfidf)[0]
    
    pred_label = "FAKE" if pred == 0 else "REAL"
    confidence = proba[pred] * 100
    
    print(f"Article: {article_name}")
    print(f"Text: {text[:70]}...")
    print(f"Prediction: {pred_label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"(Fake: {proba[0]*100:.1f}%, Real: {proba[1]*100:.1f}%)")
    print()

# ============================================================
# STEP 13: SAVE MODEL AND VECTORIZER
# ============================================================

print("="*70)
print("SAVING MODEL")
print("="*70)

import pickle

# Save model
with open('/mnt/user-data/outputs/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved to 'fake_news_model.pkl'")

# Save vectorizer
with open('/mnt/user-data/outputs/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✓ Vectorizer saved to 'vectorizer.pkl'")

# Save test results
test_df.to_csv('/mnt/user-data/outputs/test_results.csv', index=False)
print("✓ Test results saved to 'test_results.csv'")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print("\nYou can now:")
print("  1. Load the model with: pickle.load(open('fake_news_model.pkl', 'rb'))")
print("  2. Review test results in 'test_results.csv'")
print("  3. View confusion matrix in 'confusion_matrix.png'")
print("="*70)
