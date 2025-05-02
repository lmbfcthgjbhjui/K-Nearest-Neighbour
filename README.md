 K-Nearest Neighbors (KNN) project! It's a really intuitive algorithm, kind of like making a decision based on what the majority of your neighbors think. Clicks the "click here to download dataset" link. Ah, the Iris dataset! A classic for classification, perfect for visualizing those decision boundaries later on.

Let's get our Python environment set up with the necessary libraries:
Here is the python code below:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap
import seaborn as sns
Okay, all the tools are ready. Now, let's load the Iris dataset:

Python

# Load the Iris dataset
data = pd.read_csv('iris.csv') # Assuming the file is in the same directory

# Separate features (X) and target (y)
X = data.drop('species', axis=1)
y = data['species']

# Convert species names to numerical labels for easier handling
species_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
y_numerical = y.map(species_mapping)
We've loaded the data and separated the features (sepal length, sepal width, petal length, petal width) from the target variable ('species'). I've also converted the species names into numerical labels (0, 1, 2) which is often helpful for machine learning tasks.

Now, the first hint: choose a classification dataset and normalize features. We've got the Iris dataset. Let's split it into training and testing sets and then normalize the features. Normalization is important for KNN because it relies on distance calculations, and features with larger scales could disproportionately influence the results.
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_numerical, test_size=0.3, random_state=42)

# Feature normalization using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Data split and features scaled – check! Now, let's move on to the second hint: use KNeighborsClassifier from sklearn and experiment with different values of K. 'K' is the number of nearest neighbors we'll consider when making a prediction.   

Python

# Experiment with different values of K
k_values = range(1, 21)
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Plot the accuracy for different values of K
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('Accuracy vs. K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Let's choose a K value based on the plot (e.g., the one with the highest accuracy)
best_k = k_values[np.argmax(accuracy_scores)]
print(f"\nThe best K value based on the plot is: {best_k}")

# Train the KNN classifier with the best K
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred_best_knn = best_knn.predict(X_test_scaled)
We've experimented with K values from 1 to 20, trained a KNN classifier for each, and recorded the accuracy on the test set. The plot helps us visualize how the choice of K affects the model's performance. Based on the plot, we've selected a 'best' K and trained a final KNN model with that value.

Now for the fourth hint: evaluate the model using accuracy and the confusion matrix. We've already calculated the accuracy. Let's look at the confusion matrix and a classification report for a more detailed evaluation.
# Evaluate the model with the best K
accuracy = accuracy_score(y_test, y_pred_best_knn)
conf_matrix = confusion_matrix(y_test, y_pred_best_knn)
class_report = classification_report(y_test, y_pred_best_knn, target_names=species_mapping.keys())

print(f"\nAccuracy of KNN with K={best_k}: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Visualize the confusion matrix using seaborn for better readability
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=species_mapping.keys(), yticklabels=species_mapping.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix for KNN (K={best_k})')
plt.show()
# Experiment with different values of K
k_values = range(1, 21)
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Plot the accuracy for different values of K
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('Accuracy vs. K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Let's choose a K value based on the plot (e.g., the one with the highest accuracy)
best_k = k_values[np.argmax(accuracy_scores)]
print(f"\nThe best K value based on the plot is: {best_k}")

# Train the KNN classifier with the best K
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred_best_knn = best_knn.predict(X_test_scaled)
We've experimented with K values from 1 to 20, trained a KNN classifier for each, and recorded the accuracy on the test set. The plot helps us visualize how the choice of K affects the model's performance. Based on the plot, we've selected a 'best' K and trained a final KNN model with that value.

Now for the fourth hint: evaluate the model using accuracy and the confusion matrix. We've already calculated the accuracy. Let's look at the confusion matrix and a classification report for a more detailed evaluation.

Python

# Evaluate the model with the best K
accuracy = accuracy_score(y_test, y_pred_best_knn)
conf_matrix = confusion_matrix(y_test, y_pred_best_knn)
class_report = classification_report(y_test, y_pred_best_knn, target_names=species_mapping.keys())

print(f"\nAccuracy of KNN with K={best_k}: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Visualize the confusion matrix using seaborn for better readability
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=species_mapping.keys(), yticklabels=species_mapping.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix for KNN (K={best_k})')
plt.show()
We've printed the accuracy, the confusion matrix (which shows the counts of true positives, true negatives, and false positives/negatives for each class), and a classification report (which includes precision, recall, F1-score, and support for each class). The heatmap provides a more visually intuitive representation of the confusion matrix.

Finally, the fifth hint: visualize decision boundaries. Since the Iris dataset has four features, visualizing decision boundaries in 4D is tricky. However, we can visualize the decision boundaries using only two features at a time to get an idea of how the KNN classifier separates the classes. Let's use the first two features (sepal length and sepal width) for this visualization.

Python

# Visualize decision boundaries (using the first two features for simplicity)
X_vis = X[['sepal_length', 'sepal_width']].values
y_vis = y_numerical.values

# Split the visualization data
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y_vis, test_size=0.3, random_state=42)

# Scale the visualization features
scaler_vis = StandardScaler()
X_train_scaled_vis = scaler_vis.fit_transform(X_train_vis)
X_test_scaled_vis = scaler_vis.transform(X_test_vis)

# Train KNN with the best K on the visualization data
knn_vis = KNeighborsClassifier(n_neighbors=best_k)
knn_vis.fit(X_train_scaled_vis, y_train_vis)

# Create a meshgrid of points to plot the decision boundaries
x_min, x_max = X_scaled_vis[:, 0].min() - 1, X_scaled_vis[:, 0].max() + 1
y_min, y_max = X_scaled_vis[:, 1].min() - 1, X_scaled_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a colormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plot the decision boundaries
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)

# Plot the training points
scatter = plt.scatter(X_scaled_vis[:, 0], X_scaled_vis[:, 1], c=y_vis, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.title(f'Decision Boundaries of KNN (K={best_k}) using Sepal Length and Width')
plt.legend(*scatter.legend_elements(), title="Species", labels=species_mapping.keys())
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
This visualization shows how the KNN classifier separates the Iris species based on just the sepal length and sepal width features. The colored regions represent the decision boundaries for each class.
