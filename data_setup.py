import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def split_data():
    url="https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
    pima = pd.read_csv(url)
    
    X = pima.drop('Outcome', axis=1)
    y = pima['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Separate features and target variable in the training set
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    # Separate features and target variable in the test set
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
    # Print the shapes of the resulting arrays or dataframes
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
split_data()

