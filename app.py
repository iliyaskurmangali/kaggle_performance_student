import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance


df=pd.read_csv('data/Student_performance_data _.csv')


# As we can see we don't have any missing values
df.isnull().sum()/len(df)

y=df[['GPA']]
X=df.drop(columns=['StudentID','GradeClass','GPA','Sports','Music','Age','Ethnicity','ParentalEducation','Gender','Volunteering'])

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])


# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols)
    ])

# Split data
X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.1, random_state=42)

# Fit the preprocessor
preprocessor.fit(X_train)

# Model
model1 = RandomForestRegressor(random_state=42, n_jobs=-1)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model1)
])

# GridSearchCV for hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 150],
    'model__max_depth': [None, 10, 15],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Best Model Parameters: {grid_search.best_params_}')
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')


# Perform the permutation
permutation_score = permutation_importance(best_model, X_train, y_train, n_repeats=10)

# Unstack results showing the decrease in performance after shuffling features
importance_df = pd.DataFrame(np.vstack((X_train.columns,
                                        permutation_score.importances_mean)).T)
importance_df.columns=['feature','score decrease']

# Show the important features
print(importance_df.sort_values(by="score decrease", ascending = False))

import joblib
# Save the preprocessor and the best model
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(best_model, 'best_model.joblib')
