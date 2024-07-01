# Reviews Classifier

This project aims to classify reviews into positive, neutral, or negative sentiments using a Support Vector Machine (SVM) classifier and TF-IDF vectorization.

## Sentiment Labels

- **1**: Positive review
- **0**: Neutral review
- **-1**: Negative review

## Prerequisites

Before running the project, ensure you have the following packages installed:

- pandas
- scikit-learn
- joblib

You can install these packages using pip:

```sh
pip install pandas scikit-learn joblib
```

## Run

```sh
python3 main.py
```

### How to Expand the Dataset

To expand the dataset, simply add more reviews to the reviews_dataset.csv file. Each review should be on a new line, and the format should be as follows:

```code
review,label
"New review text",label
```

### Adapting for Other Use Cases

This classifier can be adapted for other text classification tasks by modifying the dataset and labels. Here are some examples:

Spam Detection: Classify emails or messages as spam or not spam.

- Labels: 1 for spam, 0 for not spam.
  Dataset Example:
- Dataset Example:

```code
review,label
"Congratulations! You've won a $1000 gift card.",1
"Can we schedule a meeting for tomorrow?",0
```

### Expanded Use Case for Aspect-Based Review Classification

In addition to classifying reviews into positive, neutral, or negative sentiments, this project can also categorize reviews based on specific aspects such as food quality, service, ambiance, pricing, etc. This allows for a more detailed analysis of customer opinions and can provide valuable insights for businesses to understand which areas require improvement or are highly valued by their customers.

### Examples of Application

1. Restaurant Reviews:

- Food Quality (Label 0): "The restaurant has delicious food, but the service was lacking."
- Service (Label 1): "The service was exceptional, but the food was average."
- Ambiance (Label -1): "The ambiance was cozy, but the prices were too high."

2. Movie Reviews:

- Plot (Label 0): "The movie had an intriguing plot, but the acting fell short."
- Acting (Label 1): "The acting was superb, but the storyline was confusing."
- Direction (Label -1): "The direction was impressive, but the soundtrack was distracting."
