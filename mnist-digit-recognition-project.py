import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

print(" Loading MNIST Dataset (Handwritten Digits)... Please wait.")

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

X = mnist.data  
y = mnist.target.astype(int)  

print(f" Dataset Loaded: {X.shape[0]} total samples with {X.shape[1]} features (pixels).")

X = X / 255.0

print(" Splitting dataset into Training (80%) and Testing (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  )

print(" Training Random Forest Classifier on MNIST...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(" Model Training Completed!")

print("\n Evaluating Model Performance on Test Data...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n Overall Model Test Accuracy: {accuracy * 100:.2f}%\n")

print(" Detailed Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("MNIST Digit Recognition - Confusion Matrix")
plt.xlabel("Predicted Digit Label")
plt.ylabel("Actual Digit Label")
plt.tight_layout()
plt.show()