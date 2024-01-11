import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score

def SVM_C_tuning_plot(xtrain, ytrain, xval, yval):
    """ search the space for the hyperparameter C in SVM to get a reasonable range for the following grid search """
    c_range = list(range(1, 51))
    scores = []
    # accuracy = []
    for C in range(1, 51):
        svm = SVC(C=C, kernel='rbf')
        svm.fit(xtrain, ytrain)
        score = cross_val_score(svm, xtrain, ytrain, cv=5, scoring='accuracy')
        # acc = accuracy_score(yval, svm.predict(xval))
        scores.append(score.mean())
        # accuracy.append(acc)
    
    # plt.plot(c_range, accuracy, label='validation accuracy')
    plt.plot(c_range, scores, label='average cv accuracy')
    plt.xlabel('Value of C for SVM')
    plt.ylabel('Cross Validation Accuracy')
    plt.title('hyperparameter C tuning for SVM model')
    plt.xticks(range(0,51,5))
    # plt.legend()
    plt.grid(ls=':')
    plt.savefig("A/figures/SVM_C_tuning_plot.png")

def SVM_tuning(params, xtrain, ytrain):
    svm = SVC()
    grid_search = GridSearchCV(svm, params, scoring='accuracy', cv=5)
    grid_search.fit(xtrain, ytrain)
    best_classifier = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_results = grid_search.cv_results_
    return best_classifier, best_params, cv_results

