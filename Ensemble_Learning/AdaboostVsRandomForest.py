import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Step 1: Data Preparation

# Load the dataset
file_path = 'indian_liver_patient.csv'
data = pd.read_csv(file_path)

# Renaming columns based on standard liver patient dataset column names
data.columns = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
    'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase', 'Total_Proteins',
    'Albumin', 'Albumin_and_Globulin_Ratio', 'Liver_Disease'
]

# Handle missing values by imputing with the mean of the column
data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean())

# Encode categorical data
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Re-encode the target variable to be {0, 1}
data['Liver_Disease'] = data['Liver_Disease'].map({2: 0, 1: 1})

# Split the data into features and target variable
X = data.drop('Liver_Disease', axis=1)
y = data['Liver_Disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Model Training and Evaluation

# Initialize the models
ada_model = AdaBoostClassifier(random_state=42, algorithm='SAMME')
rf_model = RandomForestClassifier(random_state=42)

# Train the models
ada_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Make predictions
ada_predictions = ada_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Calculate performance metrics
ada_accuracy = accuracy_score(y_test, ada_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

ada_precision = precision_score(y_test, ada_predictions)
rf_precision = precision_score(y_test, rf_predictions)

ada_recall = recall_score(y_test, ada_predictions)
rf_recall = recall_score(y_test, rf_predictions)

ada_f1 = f1_score(y_test, ada_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

# Calculate ROC curve and AUC
ada_fpr, ada_tpr, _ = roc_curve(y_test, ada_predictions)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_predictions)

ada_auc = auc(ada_fpr, ada_tpr)
rf_auc = auc(rf_fpr, rf_tpr)

# Print the performance metrics
print(f'AdaBoost - Accuracy: {ada_accuracy:.2f}, Precision: {ada_precision:.2f}, Recall: {ada_recall:.2f}, F1 Score: {ada_f1:.2f}, AUC: {ada_auc:.2f}')
print(f'Random Forest - Accuracy: {rf_accuracy:.2f}, Precision: {rf_precision:.2f}, Recall: {rf_recall:.2f}, F1 Score: {rf_f1:.2f}, AUC: {rf_auc:.2f}')

# Step 3: Visualization

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

plot_confusion_matrix(y_test, ada_predictions, 'AdaBoost')
plot_confusion_matrix(y_test, rf_predictions, 'Random Forest')

# Plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(ada_fpr, ada_tpr, label=f'AdaBoost (AUC = {ada_auc:.2f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
