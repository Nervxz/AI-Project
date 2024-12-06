# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample  

# Load data
df = pd.read_csv('D:/BCU - SS2/AI and machine learning/Report/Dataset/Electric_Vehicle_Population_Data.csv')  

# Preprocessing
print(df.info())
print(df.isnull().sum())

# Handle missing values
df['Electric Range'] = df['Electric Range'].fillna(df['Electric Range'].median())
df['Base MSRP'] = df['Base MSRP'].fillna(df['Base MSRP'].median())
df['County'] = df['County'].fillna(df['County'].mode()[0])
df['City'] = df['City'].fillna(df['City'].mode()[0])
df['Postal Code'] = df['Postal Code'].fillna(df['Postal Code'].mode()[0])
df['Electric Utility'] = df['Electric Utility'].fillna(df['Electric Utility'].mode()[0])
df['Vehicle Location'] = df['Vehicle Location'].fillna(df['Vehicle Location'].mode()[0])


# Check for class imbalance
print(df['Electric Vehicle Type'].value_counts())

# Handle class imbalance by oversampling the minority class
# Separate majority and minority classes
df_majority = df[df['Electric Vehicle Type'] == df['Electric Vehicle Type'].mode()[0]]
df_minority = df[df['Electric Vehicle Type'] != df['Electric Vehicle Type'].mode()[0]]

# Upsample the minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,         # Sample with replacement
                                 n_samples=len(df_majority),  # Match the number of majority class samples
                                 random_state=42)      # For reproducibility

# Combine the majority class with the upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Recheck class distribution
print(df_upsampled['Electric Vehicle Type'].value_counts())

# Select features and target variable
X = df_upsampled[['Model Year', 'Electric Range']]  # Using fewer features
y = df_upsampled['Electric Vehicle Type']

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree model with constraints
dt_model = DecisionTreeClassifier(
    max_depth=2,                # Shallow depth to avoid overfitting
    min_samples_split=50,       # Require at least 50 samples to split an internal node
    min_samples_leaf=20,        # Require at least 20 samples in each leaf node
    random_state=42
)

# Train the model
dt_model.fit(X_train, y_train)

# Predictions
y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

# Calculate and display accuracies for both train and test sets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Classification report for more detailed evaluation
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Cross-validation to check model stability
cv_scores = cross_val_score(dt_model, X, y, cv=5)  # 5-fold cross-validation
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean()}")

# Plotting the Decision Tree (optional)
from sklearn.tree import plot_tree
plt.figure(figsize=(12, 8))
plot_tree(dt_model, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True)
plt.title('Decision Tree')
plt.show()
