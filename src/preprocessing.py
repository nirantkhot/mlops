import pandas as pd
import numpy as np 
import pickle
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train.csv')
df = df.drop(columns='Id')
df_y = df['Class']
df = df.drop(columns='Class')

train_X, test_X, train_y, test_y = train_test_split(df, df_y, test_size=0.2, random_state=42)

# Create pipeline for imputing and scaling numeric variables
# one-hot encoding categorical variables, and select features based on chi-squared value
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_include = ['int', 'float'])),
        ("cat", categorical_transformer, make_column_selector(dtype_exclude = ['int', 'float'])),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor)]
)

# Create new train and test data using the pipeline
clf.fit(train_X, train_y)
train_new = clf.transform(train_X)
test_new = clf.transform(test_X)

# Transform to dataframe and save as a csv
train_new = pd.DataFrame(train_new)
test_new = pd.DataFrame(test_new)
train_new['y'] = train_y
test_new['y'] = test_y

train_new.to_csv('data/processed_train_data.csv')
test_new.to_csv('data/processed_test_data.csv')

# Save pipeline
with open('data/pipeline.pkl','wb') as f:
    pickle.dump(clf,f)
