import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Load the data
data = pd.read_csv('financial_fraud_detection.csv')

# Define features and target
X = data.drop('Fraud', axis=1)
y = data['Fraud']

# Preprocessing for numerical and categorical features
numeric_features = ['Amount', 'AccountAge', 'TransactionSpeed', 'NumberOfLogins']
categorical_features = ['TransactionType', 'CustomerLocation', 'DeviceType', 'DayOfWeek', 'CardType', 'MerchantCategory', 'Country']

# Create transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data
X_processed = preprocessor.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Train models
for name, model in models.items():
    model.fit(X_train_pca, y_train)

# Save the preprocessor, PCA, and models
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(models['Logistic Regression'], 'logistic_regression.pkl')
joblib.dump(models['Random Forest'], 'random_forest.pkl')
joblib.dump(models['SVM'], 'svm.pkl')

print("Preprocessor, PCA, and models have been saved successfully.")
