import A.main
import B.main

if __name__ == '__main__':
    A.main.train_all_models()
    A.main.show_tuning_process(show_knn=False, show_svm=False, show_cnn=False)
    B.main_train_all_models()
    B.main_show_tuning_process()
