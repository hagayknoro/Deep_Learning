import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("tweets.csv")

# Inspect the dataset structure (uncomment to view)
# print(data.head())

# Assuming the dataset has columns 'text' and 'target'
# 'target' is the label (0 or 1), and 'text' contains the tweets

# Split the dataset into train and test sets (80/20 split)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Compute the proportion of each class in the training data
majority_class = train_data['target'].value_counts().idxmax()  # Class with the highest frequency
print(f"Majority class in training data: {majority_class}")

# Predict the majority class for all test samples
test_data['baseline_prediction'] = majority_class

# Compute the evaluation metrics
accuracy = accuracy_score(test_data['target'], test_data['baseline_prediction'])
precision = precision_score(test_data['target'], test_data['baseline_prediction'], zero_division=0)
recall = recall_score(test_data['target'], test_data['baseline_prediction'], zero_division=0)

# Print the baseline metrics
print("Baseline Model Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
