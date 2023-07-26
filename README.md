
# Fake News Detection

A machine learning project to detect fake news articles using TF-IDF vectorization and the PassiveAggressiveClassifier algorithm.

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd fake-news-detection
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Import the required libraries:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
```

2. Initialize a TfidfVectorizer:

```python
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
```

3. Fit and transform the training set, and transform the test set:

```python
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)
```

4. Initialize a PassiveAggressiveClassifier:

```python
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
```

5. Predict on the test set and calculate accuracy:

```python
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 2)}%')
```

## Data

The project uses a dataset containing labeled news articles, where each article is labeled as either 'FAKE' or 'REAL'.

## Model

The model used is the PassiveAggressiveClassifier combined with TF-IDF vectorization for text classification.

## Evaluation

The model's performance is evaluated using accuracy, precision, recall, F1-score, ROC curve, and AUC score.

## License

MIT License

Please feel free to add more details or modify the content as per your specific project. A concise README.md provides the essential information to users and contributors in a straightforward manner.
