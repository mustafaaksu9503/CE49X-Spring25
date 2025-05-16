import json
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC

# 1. Load the data
with open("../../labs/lab6/construction_documents.json", "r") as f:
    docs = json.load(f)
df = pd.DataFrame(docs)


# 2. ---------------- Feature Engineering Practice ----------------
# 2a. Handle missing metadata
meta_cols = ['project_phase', 'author_role']
imputer = SimpleImputer(strategy='constant', fill_value='missing')
df[meta_cols] = imputer.fit_transform(df[meta_cols])

# 2b. One-hot encode metadata with DictVectorizer
meta_dicts = df[meta_cols].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
meta_features = dv.fit_transform(meta_dicts)
print(f"Metadata features shape: {meta_features.shape}")

# 2c. Visualize distribution of document types across project phases
df['project_phase'].value_counts().plot(kind='bar')
plt.title('Documents per Project Phase')
plt.xlabel('Project Phase')
plt.ylabel('Count')
plt.show()

# 2d. Text preprocessing: remove common abbreviations and measurements
def preprocess_text(text):
    text = text.lower()
    # remove abbreviations (e.g., 'mm', 'kg', 'm2')
    text = re.sub(r"\b(?:mm|cm|m|kg|t|m2|m3)\b", '', text)
    # remove digits
    text = re.sub(r"[0-9]+", ' ', text)
    # remove punctuation
    text = re.sub(r"[^\w\s]+", ' ', text)
    return text

# Apply preprocessing
df['clean_text'] = df['content'].apply(preprocess_text)

# 2e. Convert text to numeric features
# Count Vectorizer
count_vec = CountVectorizer(stop_words='english', min_df=2)
X_count = count_vec.fit_transform(df['clean_text'])
print(f"Count features shape: {X_count.shape}")
# TF-IDF Vectorizer
tfidf_vec = TfidfVectorizer(stop_words='english', min_df=2)
X_tfidf = tfidf_vec.fit_transform(df['clean_text'])
print(f"TF-IDF features shape: {X_tfidf.shape}")

# 3. ---------------- Document Classification ----------------
# 3a. Choose which text features to use (here: TF-IDF)
X_text = X_tfidf

y = df['document_type']

# 3b. Combine metadata and text features
X = hstack([meta_features, X_text])

# 3c. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3d. Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# 3e. Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 3f. Plot confusion matrix with red-to-yellow heatmap
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix (Red to Yellow)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.show()

# 4. ---------------- Advanced Analysis ----------------
# 4a. Temporal pattern analysis
# First convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

plt.figure(figsize=(12, 6))
df.groupby([df['date'].dt.month, 'document_type']).size().unstack().plot(kind='bar')
plt.title('Document Types by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(title='Document Type')
plt.show()

# 4b. Feature importance analysis (for Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature names - remove .tolist() since get_feature_names() already returns a list
feature_names = dv.get_feature_names() + tfidf_vec.get_feature_names()
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot top 20 important features
plt.figure(figsize=(12, 6))
feature_importance.head(20).plot(kind='bar', x='feature', y='importance')
plt.title('Top 20 Important Features')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print top 10 most important features
print("\nTop 10 most important features:")
print(feature_importance.head(10))

# 5. ---------------- Cross Validation ----------------

# Define the models to evaluate
models = {
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Linear SVM': LinearSVC(random_state=42)
}

# Define scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted'
}

# Perform k-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Cross-validation results:")
print("-" * 50)

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 30)
    
    # Perform cross-validation for each metric
    for metric_name, metric in scoring.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring=metric)
        print(f"{metric_name.capitalize()}:")
        print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        print(f"Individual scores: {scores}")

# Visualize cross-validation results
plt.figure(figsize=(12, 6))
results = []

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    results.append({
        'Model': name,
        'Mean Accuracy': scores.mean(),
        'Std Accuracy': scores.std()
    })

results_df = pd.DataFrame(results)
results_df.plot(kind='bar', x='Model', y='Mean Accuracy', yerr='Std Accuracy', capsize=5)
plt.title('Cross-validation Results (Accuracy)')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print detailed results table
print("\nDetailed Cross-validation Results:")
print(results_df.to_string(index=False))