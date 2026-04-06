import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# importing models and metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# importing cross validation
from sklearn.model_selection import cross_val_score

 

# 1. load dataset
housing = pd.read_csv('housing.csv')

# 2. create stratified set
housing['income_cat'] = pd.cut(housing['median_income'], 
                                bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                labels=[1, 2, 3, 4, 5])

# Ensure strat_train_set is defined before use
# 3. stratified splitiong
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis=1)
    strat_test_set = housing.loc[test_index].drop('income_cat', axis=1)

# we will copy the trainig set to avoid any changes to the original data
housing = strat_train_set.copy() # type: ignore

#  4. separating labels from features 
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)
# print(housing_labels)


#  5. numerical pipeline
num_attributes = housing.select_dtypes(include=[np.number]).columns.tolist()
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# 6. categorical pipeline
cat_attributes = ['ocean_proximity']
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder(handle_unknown='ignore')),
])


# 7. full pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes),
])

housing_prepared = full_pipeline.fit_transform(housing)
print(f"Shape of prepared housing data: {housing_prepared.shape}")


#* Now we are ready to train our model using housing_prepared and housing_labels

# linear regression model
linear_reg = LinearRegression()
linear_reg.fit(housing_prepared, housing_labels)

# decision tree model
decision_tree = DecisionTreeRegressor()
decision_tree.fit(housing_prepared, housing_labels)

# random forest model
random_forest = RandomForestRegressor()
random_forest.fit(housing_prepared, housing_labels)


# 8. evaluating models
def evaluate_model(model, features, labels):
    predictions = model.predict(features)
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    return rmse

print("Evaluating Models:")
print(f"Linear regression RMSE: {evaluate_model(linear_reg, housing_prepared, housing_labels)}")
print(f"Decision tree RMSE: {evaluate_model(decision_tree, housing_prepared, housing_labels)}")
print(f"Random forest RMSE: {evaluate_model(random_forest, housing_prepared, housing_labels)}")


# Evaluate Decision Tree with cross-validation
# WARNING: Scikit-Learn’s scoring uses utility functions (higher is better), so RMSE is returned as negative.
# We use minus (-) to convert it back to positive RMSE.
decision_tree_rmses = -cross_val_score(
    decision_tree,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

# ! this is performimg best from the three models
random_forest_rmses = -cross_val_score(
    random_forest,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
) 

print("\nCross-Validation Performance (Decision Tree):")
print("Decision Tree CV RMSEs:", decision_tree_rmses)
print(pd.Series(decision_tree_rmses).describe())


print("\nCross-Validation Performance (Random Forest):")
print("Random Forest CV RMSEs:", random_forest_rmses)
print(pd.Series(random_forest_rmses).describe())  

