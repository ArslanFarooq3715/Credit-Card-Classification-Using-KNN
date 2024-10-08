# Credit Card Fraud Detection

## Overview

This project focuses on detecting fraudulent credit card transactions using a K-Nearest Neighbors (KNN) classifier. The goal is to preprocess the credit card dataset, apply the KNN algorithm, and evaluate the model's performance using cross-validation and confusion matrix analysis.

## Dataset Description

The dataset, `CreditCard.csv`, contains information about credit card transactions. Key features selected for analysis include:

- **Age**: Age of the cardholder.
- **Income**: Annual income of the cardholder.
- **Load**: Amount of credit used.

### Data Loading and Exploration

1. **Load the Dataset**: The `CreditCard.csv` file is loaded into a DataFrame for analysis.
   
   ```python
   import pandas as pd

   data = pd.read_csv('CreditCard.csv')
   ```

2. **Describe the Dataset**: Basic statistics and information about the dataset are obtained.

   ```python
   print(data.describe())
   ```

## Feature Selection

The following features are selected for training the model:

- **Age**
- **Income**
- **Load**

## Data Splitting

The dataset is split into training and testing sets to evaluate the performance of the model.

```python
from sklearn.model_selection import train_test_split

X = data[['Age', 'Income', 'Load']]
y = data['Fraudulent']  # Assuming 'Fraudulent' is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Model Training

1. **KNN Classifier**: A KNN classifier with `k=4` is applied to fit the model.

   ```python
   from sklearn.neighbors import KNeighborsClassifier

   knn = KNeighborsClassifier(n_neighbors=4)
   knn.fit(X_train, y_train)
   ```

2. **Model Prediction**: The model is used to predict outcomes on the test set.

   ```python
   predictions = knn.predict(X_test)
   ```

## Cross-Validation and Optimal K

To determine the optimal value of `k`, cross-validation is performed. The optimal `k` is found to be `28`.

```python
from sklearn.model_selection import cross_val_score

k_values = range(1, 30)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10)
    cv_scores.append(scores.mean())

optimal_k = k_values[cv_scores.index(max(cv_scores))]
```

## Model Evaluation

### Confusion Matrix

The confusion matrix is generated as follows:

```
[[522   1]
 [  8  69]]
```

### Accuracy

The model achieves an accuracy of **0.985**.

```python
from sklearn.metrics import confusion_matrix, accuracy_score

conf_matrix = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
```

## Conclusion

This project demonstrates the use of KNN for detecting fraudulent credit card transactions. The model achieved a high accuracy of 98.5%, indicating its effectiveness in distinguishing between legitimate and fraudulent transactions. Future work could involve testing additional classifiers or fine-tuning model parameters to improve performance further.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

Feel free to modify any sections as needed, and fill in any additional details relevant to your project!
