import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the original time arrays
original_times15 = [
    [766, 846, 685, 694, 721, 783],
    [766, 718, 685, 694, 721, 783],
    [766, 718, 685, 694, 721, 718],
    [727, 718, 685, 694, 721, 718],
    [718, 718, 685, 694, 721, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [727, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718],
    [718, 718, 685, 694, 718, 718]
]

original_times11 = [
    [760, 662, 797, 738, 764, 646],
    [760, 662, 658, 738, 764, 646],
    [760, 662, 658, 738, 658, 646],
    [694, 662, 658, 738, 658, 646],
    [694, 662, 658, 658, 658, 646],
    [658, 662, 658, 658, 658, 646],
    [658, 694, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [694, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [694, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [714, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [694, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646]
]

original_times1 = [
    [678, 660, 752, 766, 701, 697],
    [678, 660, 752, 660, 701, 697],
    [678, 660, 660, 660, 701, 697],
    [678, 660, 660, 660, 660, 697],
    [678, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660],
    [660, 660, 660, 660, 660, 660]
]

original_times5 = [
    [800, 841, 734, 680, 679, 702],
    [800, 649, 734, 680, 679, 702],
    [655, 649, 734, 680, 679, 702],
    [655, 649, 655, 680, 679, 702],
    [655, 649, 655, 680, 679, 680],
    [655, 649, 655, 655, 679, 680],
    [655, 649, 655, 655, 679, 680],
    [655, 649, 655, 655, 679, 680],
    [655, 649, 655, 655, 679, 655],
    [655, 649, 655, 655, 680, 655],
    [655, 649, 655, 655, 680, 655],
    [655, 649, 655, 655, 680, 655],
    [655, 649, 655, 655, 655, 655],
    [655, 649, 655, 655, 655, 655],
    [655, 649, 655, 655, 655, 655],
    [655, 649, 655, 655, 655, 655],
    [655, 649, 655, 655, 655, 655],
    [655, 649, 655, 655, 655, 655],
    [655, 649, 655, 655, 655, 655],
    [655, 649, 655, 655, 655, 655],
    [649, 649, 655, 655, 655, 655]
]


class CustomDummyRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, population_size=50, temperature=11):
        self.population_size = population_size
        self.temperature = temperature

    def fit(self, X, y=None):
        # DummyRegressor does not require fitting, so we do nothing here
        return self

    def predict(self, X):
        # DummyRegressor always predicts the mean of the target variable
        # Here, we return the population size and temperature as the prediction
        return np.array([[self.population_size, self.temperature]] * len(X))

# Define a function to evaluate the performance of the genetic algorithm
def evaluate_performance(estimator, X, y=None):
    # Calculate the min of all values in the dataset
    return np.min(X)


# Instantiate CustomDummyRegressor as a placeholder estimator
dummy_regressor = CustomDummyRegressor()

# Define parameters grid for dataset 11
parameters_grid_11 = {
    'population_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Example values for population size
    'temperature': [11],  # Temperature is 11 for dataset 11
}

# Define parameters grid for dataset 1
parameters_grid_1 = {
    'population_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Example values for population size
    'temperature': [1],  # Temperature is 1 for dataset 1
}

# Define parameters grid for dataset 5
parameters_grid_5 = {
    'population_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Example values for population size
    'temperature': [5],  # Temperature is 5 for dataset 5
}

# Define parameters grid for dataset 15
parameters_grid_15 = {
    'population_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Example values for population size
    'temperature': [15],  # Temperature is 15 for dataset 15
}

# Perform grid search for dataset 11
grid_search_11 = GridSearchCV(estimator=dummy_regressor, param_grid=parameters_grid_11, cv=5, scoring=evaluate_performance)
grid_search_11.fit(original_times11)

# Perform grid search for dataset 1
grid_search_1 = GridSearchCV(estimator=dummy_regressor, param_grid=parameters_grid_1, cv=5, scoring=evaluate_performance)
grid_search_1.fit(original_times1)

# Perform grid search for dataset 5
grid_search_5 = GridSearchCV(estimator=dummy_regressor, param_grid=parameters_grid_5, cv=5, scoring=evaluate_performance)
grid_search_5.fit(original_times5)

# Perform grid search for dataset 15
grid_search_15 = GridSearchCV(estimator=dummy_regressor, param_grid=parameters_grid_15, cv=5, scoring=evaluate_performance)
grid_search_15.fit(original_times15)

# Access the results
print("Best parameters for dataset 11:", grid_search_11.best_params_)
print("Best score for dataset 11:", grid_search_11.best_score_)

print("Best parameters for dataset 1:", grid_search_1.best_params_)
print("Best score for dataset 1:", grid_search_1.best_score_)

print("Best parameters for dataset 5:", grid_search_5.best_params_)
print("Best score for dataset 5:", grid_search_5.best_score_)

print("Best parameters for dataset 15:", grid_search_15.best_params_)
print("Best score for dataset 15:", grid_search_15.best_score_)

# Extract grid search results for dataset 11
results_df_11 = pd.DataFrame(grid_search_11.cv_results_)

# Extract grid search results for dataset 1
results_df_1 = pd.DataFrame(grid_search_1.cv_results_)

# Extract grid search results for dataset 5
results_df_5 = pd.DataFrame(grid_search_5.cv_results_)

# Extract grid search results for dataset 15
results_df_15 = pd.DataFrame(grid_search_15.cv_results_)

# Combine the results of all datasets into one DataFrame
combined_results_df = pd.concat([results_df_11, results_df_1, results_df_5, results_df_15])

# Select relevant columns
relevant_cols_combined = ['param_population_size', 'param_temperature', 'mean_test_score']

# Create a pivot table for visualization
pivot_table_combined = combined_results_df.pivot_table(index='param_population_size', columns='param_temperature', values='mean_test_score')

# Create a heatmap for all datasets combined
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table_combined, annot=True, cmap='viridis', fmt=".2f")
plt.title('Grid Search Results for Selected Datasets')
plt.xlabel('Temperature')
plt.ylabel('Population Size')
plt.show()
