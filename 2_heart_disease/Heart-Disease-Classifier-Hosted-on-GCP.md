# Predict heart disease with a RandomForestClassifier

This [heart disease data comes from Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) and I used it to predict if a person has heart disease using a RandomForestClassifier.

[You can see how I'd host this model at a low-cost/no-cost way on GCP](https://github.com/scottfrasso/host-a-model)


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import pandas as pd
heart_disease = pd.read_csv("heart-disease.csv")

heart_disease.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Shuffle and split the data


```python
np.random.seed(42)

# Shuffle the data
heart_disease_shuffled = heart_disease.sample(frac=1)

# Split into X & y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]
```

## We'll RandomForestClassifier and a GridSearchCV to tune the hyperparameters

Using GridSearchCV allows us to automatically tune the hypterparamters for the RandomForestClassifier to get the 
best possible prediction.


```python
%%capture
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

gs_grid_search_params = {
     'n_estimators': [100, 200, 500],
     'max_depth': [None],
     'max_features': ['auto', 'sqrt'],
     'min_samples_split': [6],
     'min_samples_leaf': [1, 2]
}

# Split into X & y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate RandomForestClassifier
clf = RandomForestClassifier(n_jobs=1)

# Setup RandomizedSearchCV
gs_clf = GridSearchCV(estimator=clf,
                      param_grid=gs_grid_search_params, 
                      cv=5,
                      verbose=2)

# Fit the RandomizedSearchCV version of clf
gs_clf.fit(X_train, y_train);
```

## The best parameters that GridSearchCV found


```python
gs_clf.best_params_
```




    {'max_depth': None,
     'max_features': 'sqrt',
     'min_samples_leaf': 1,
     'min_samples_split': 6,
     'n_estimators': 100}



## The Results

Using just the heart disease data tuning the hyperparameters gives a decent accuracy score of 81.97%


```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

gs_y_preds = gs_clf.predict(X_test)

accuracy = accuracy_score(y_test, gs_y_preds)
print(f"Acc: {accuracy * 100:.2f}%")

precision = precision_score(y_test, gs_y_preds)
print(f"Precision: {precision:.2f}")

recall = recall_score(y_test, gs_y_preds)
print(f"Recall: {recall:.2f}")

f1 = f1_score(y_test, gs_y_preds)
print(f"F1 score: {f1:.2f}")
```

    Acc: 81.97%
    Precision: 0.77
    Recall: 0.86
    F1 score: 0.81

