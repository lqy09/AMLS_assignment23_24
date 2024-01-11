import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score

def KNN(xtrain, ytrain, xtest, ytest, k=69):
    """
    create a KNN model
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xtrain, ytrain)
    accuracy = accuracy_score(ytest, knn.predict(xtest))
    return accuracy

def KNN_k_tuning_plot(xtrain, ytrain):
    """
    For KNN hyperparameter tuning, we first search the space for the sensitive param n_neighbors to find a good range of it
    Then we create a plot for the n_neighbors
    """
    k_range = list(range(1, 81))
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=2)
        score = cross_val_score(knn, xtrain, ytrain, cv=5, scoring='accuracy')
        scores.append(score.mean())
    plt.plot(k_range, scores)
    plt.xticks(range(0, 81, 5))
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.title('hyperparameter k tuning for KNN model')
    plt.grid(ls=':')
    if not os.path.exists('./A/figures'):
        os.makedirs('./A/figures')
    plt.savefig('./A/figures/KNN_k_tuning_plot.png')

def KNN_tuning(params, xtrain, ytrain):
    """
    Tune all the hyperparameters of KNN using 5 fold cross-validation. Metric: accuracy
    """
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, params, cv=5, n_jobs=2)
    grid_search.fit(xtrain, ytrain)
    best_classifier = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_results = grid_search.cv_results_
    return best_classifier, best_params, cv_results

def KNN_tuning_results_plot(cv_results):
    """ plot the result of KNN hyperparameter tuning using heat map to visualize """
    
    df = pd.DataFrame(cv_results)
    weight_values = df['param_weights'].unique()
    num_weights = len(weight_values)
    fig, ax = plt.subplots(1, num_weights, figsize=(20, 8))
    for i, weight in enumerate(weight_values):
        df_sub = df[df['param_weights']==weight]
        piv_sub = df_sub.pivot(index='param_n_neighbors', columns='param_metric', values='mean_test_score')
        sns.heatmap(piv_sub,annot=True, fmt=".3f", cmap="YlGnBu",  ax=ax[i])
        ax[i].set_title(f"weight param = {weight}")
        ax[i].set_xlabel("metric")
        ax[i].set_ylabel("n_neighbors")
    plt.tight_layout()
    plt.savefig("./A/figures/KNN_hyperparamter_heatmap")