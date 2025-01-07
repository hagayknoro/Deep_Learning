import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# Step 1: Load the data
file_path = 'tweets.csv'  # Change this to your file path
data = pd.read_csv(file_path)

# Ensure 'text' and 'target' columns exist
data = data[['text', 'target']].dropna()

# Step 2: Convert text to numeric vectors (Bag of Words)
vectorizer = CountVectorizer(max_features=10000)  # Limit to 5000 most common words
X = vectorizer.fit_transform(data['text']).toarray()
y = data['target'].values

# Define the number of folds for cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize lists to store metrics for each fold
accuracy_list = []
precision_list = []
recall_list = []

# First, split the data into train and temp (validation + test) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# Then, split the temp set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Cross-validation loop
for train_index, val_index in kf.split(X):
    # Split the data into training and validation sets
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Step 4: Define the logistic regression model
    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            return self.sigmoid(self.linear(x))

    # Initialize the model
    input_dim = X_train.shape[1]
    model = LogisticRegressionModel(input_dim)

    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Step 5: Train the model
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Step 6: Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_val_tensor)
        predictions = (predictions >= 0.5).float()  # Threshold at 0.5

    # Convert predictions and true labels to numpy for evaluation
    y_pred = predictions.numpy().flatten()
    y_true = y_val_tensor.numpy().flatten()

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Store metrics
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)

# Calculate average metrics across all folds
avg_accuracy = np.mean(accuracy_list)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)

print("\nCross-Validation Results:")
print(f"Average Accuracy: {avg_accuracy:.2f}")
print(f"Average Precision: {avg_precision:.2f}")
print(f"Average Recall: {avg_recall:.2f}")
