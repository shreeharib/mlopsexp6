from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pandas as pd
import numpy as np
import joblib
import os
iris_df = load_iris()
X = iris_df.data
y = iris_df.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

model = LogisticRegression()
model.fit(X_train, y_train)

os.makedirs("model",exist_ok=True)
model_path = os.path.join("model","model.joblib")

joblib.dump(model,model_path)

with open("metrics.txt","w") as fw:
  fw.write(f"\nAccuracy: {accuracy_score(y_test,model.predict(X_test))}")
  fw.write(f"\n{classification_report(y_test,model.predict(X_test))}")

print("Training Completed")
