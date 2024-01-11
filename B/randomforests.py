from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def randomforest_tuning(param_grid, xtrain, ytrain):
    forest = RandomForestClassifier(n_jobs=2)
    grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(xtrain, ytrain)
    best_classifier = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_results = grid_search.cv_results_
    return best_classifier, best_params, cv_results
