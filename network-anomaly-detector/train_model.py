import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load the dataset
df = pd.read_csv('upload/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Drop columns with too many missing values
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.dropna(inplace=True)

# Drop non-numeric columns (optional, or use df.select_dtypes)
X = df.drop('Label', axis=1)
y = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
