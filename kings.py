import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('globalAirPol.csv', encoding="utf-8")

# Clean up column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Use all 5 features to match your sample input.
features = ['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
X = df[features].values
y = df['PM2.5 AQI Category']

# Split the data into training and test sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training data only
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define the RandomForestClassifier
base_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

# Train the model on the balanced training data
base_model.fit(X_train, y_train)

# Evaluate the model performance on the test set
y_pred = base_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Run the test on the specific input sample: 58, 2, 21, 8, 42
sample_input = np.array([[75, 2, 16, 8, 126]])
sample_prediction = base_model.predict(sample_input)
print("For the test input [75, 2, 16, 8, 126], the predicted category is:", sample_prediction)

# Optionally, compute and display a correlation matrix for the features and encoded target
df_numeric = df.copy()
df_numeric['PM2.5 AQI Category'] = pd.factorize(df_numeric['PM2.5 AQI Category'])[0]
corr = df_numeric[['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Category']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
