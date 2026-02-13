import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data():
    # Load dataset (Modify this path according to your dataset location)
    data = pd.read_csv('path_to_your_dataset.csv')
    return data

def preprocess_data(data):
    # Split features and target variable (modify as needed)
    X = data.drop('target', axis=1)
    y = data['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def save_model(model, scaler):
    joblib.dump(model, 'student_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model, scaler = train_model(X_train, y_train)
    save_model(model, scaler)
