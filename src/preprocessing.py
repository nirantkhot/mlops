import pandas as pd
import numpy as np 
import pickle
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def load_data(df_path):
    # Load data
    df = pd.read_csv(df_path)
    return df

def prepare_data(df, impute_strategy='median'): 
    df = df.drop(columns='Id')
    df_y = df['Class']
    df = df.drop(columns='Class')

    train_X, test_X, train_y, test_y = train_test_split(df, df_y, test_size=0.2, random_state=42)

    numeric_cols = train_X.select_dtypes(include=['int', 'float']).columns

    num_imputer = SimpleImputer(strategy='median')
    train_num_imputed = pd.DataFrame(num_imputer.fit_transform(train_X[numeric_cols]), columns=numeric_cols)
    test_num_imputed = pd.DataFrame(num_imputer.transform(test_X[numeric_cols]), columns=numeric_cols)

    scaler = StandardScaler()
    train_num_scaled = pd.DataFrame(scaler.fit_transform(train_num_imputed), columns=numeric_cols)
    test_num_scaled = pd.DataFrame(scaler.transform(test_num_imputed), columns=numeric_cols)

    return train_num_scaled, test_num_scaled, train_y, test_y


def save_data(train_new, test_new, train_name, test_name):
    train_new.to_csv(train_name)
    test_new.to_csv(test_name)

if __name__=="__main__":
    df = load_data('../data/train.csv')
    X_train, X_test, y_train, y_test = prepare_data(df)