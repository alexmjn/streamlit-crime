import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
import category_encoders as ce

def load_data():
    return pd.read_pickle("data_file.bz2")

df = load_data()

train, test = train_test_split(df, test_size=.2, random_state=42)
target = "cleared"
features = train.drop([target], axis = 1).columns.to_list()
X_train = train[features]
y_train = train[target]
X_val = test[features]
y_val = test[target]


encoder = ce.OrdinalEncoder()
X_train_encoded = encoder.fit_transform(X_train)
X_val_encoded = encoder.transform(X_val)

model = XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=.2,
    n_jobs=-1,)


eval_set = [(X_train_encoded, y_train),
            (X_val_encoded, y_val)]

model.fit(X_train_encoded,
          y_train,
          eval_set=eval_set,
          eval_metric='error',
          early_stopping_rounds=90
          )

model.save_model("xgb_model.txt")
