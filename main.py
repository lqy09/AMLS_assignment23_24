import A.main
import B.main

if __name__ == '__main__':
    A.main.train_all_models()
    A.main.show_tuning_process(show_knn=False, show_svm=False, show_cnn=False)
    B.main.train_all_models(train_randomforest=True, train_cnn=True, train_resnet=True)
    B.main.show_tuning_process(show_forest=False, show_cnn=False, show_resnet=False)
