import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
data=pd.read_csv("health_data.csv")
df=pd.DataFrame(data)
df.drop(columns=['Unnamed: 0'],inplace=True)
df.drop(columns=['id'],inplace=True)
df['age']=df['age']/365
df=df[(df['ap_hi'] > 0) & (df['ap_lo'] > 0)]
df=df[(df['ap_hi'] < 250) & (df['ap_lo'] < 200)]
df = df[(df['height'] > 100) & (df['height'] < 220)]
df = df[(df['weight'] > 30) & (df['weight'] < 200)]
x=df.drop('cardio', axis=1)
y=df['cardio']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

model = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)
grid.fit(x_train, y_train)
pickle.dump(grid.best_estimator_, open('model.pkl', 'wb'))

print(" Model trained and saved as model.pkl")
