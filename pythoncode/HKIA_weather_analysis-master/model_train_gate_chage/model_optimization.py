#  Aothor: Ethan Lee
#  E-mail: ethan2lee.coder@gmail.com
import warnings

warnings.filterwarnings('ignore')
import sys
from model_train_gate_chage.model_train_gate_change import model_assessment
from model_train_gate_chage.model_predict_gate_change import test_set_assessment
import numpy as np
import time
from multiprocessing.managers import BaseManager
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#### 重载处理好训练集和测试集
train_test_dataset = np.load("./dataset/train_test_dataset.npz")


def model_train_output(estimate, train_test_dataset, estimate_name):
    try:
        X_train = train_test_dataset['X_train']
        Y_train_gate_change = train_test_dataset['Y_train_gate_change']
        X_test = train_test_dataset['X_test']
        Y_test_gate_change = train_test_dataset['Y_test_gate_change']

        # 模型训练
        start = time.time()
        estimate.fit(X_train, Y_train_gate_change)
        end = time.time()

        ### 模型评估
        model_assessment(estimate, X_test, Y_test_gate_change)
        test_set_assessment(estimate, estimate_name, X_test, X_test, Y_test_gate_change,
                            "./model_output/model_optimization_output.txt")
        with open("./model_output/model_optimization_output.txt", "a+", encoding="utf-8") as file:
            file.write("训练集数据量： {} \n".format(len(X_train)))
            file.write("测试集数据量： {} \n".format(len(X_test)))
            file.write("模型训练时间： {} s\n".format(end - start))
        print("预测结果保存在model_output/model_optimization_output.txt，请前往查看")

    except Exception:
        print("error in {}".format(estimate_name))


def model_grid(model, param_grid, train_test_dataset=train_test_dataset):
    X_train = train_test_dataset['X_train']
    Y_train_gate_change = train_test_dataset['Y_train_gate_change']
    X_test = train_test_dataset['X_test']
    Y_test_gate_change = train_test_dataset['Y_test_gate_change']

    grid_search = GridSearchCV(model, param_grid,
                               scoring="accuracy", return_train_score=True)
    grid_search.fit(X_train, Y_train_gate_change)
    savedStdout = sys.stdout  # 保存标准输出流
    with open("model_grid_output.txt", 'a+', encoding="utf-8") as file:
        sys.stdout = file
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(mean_score, params)
        print("准确率最高的超参数:{}".format(grid_search.best_params_))
        model_assessment(grid_search.best_estimator_, X_test, Y_test_gate_change)
        print("\n")
    sys.stdout = savedStdout  # 恢复标准输出流
    return grid_search.best_estimator_


class multiprocess_model_train():
    """
    构建进程管理对象，子进程中需要引用父进程对象时，
    必须要包包含在进程管理对象中，否则会出现序列化错误。
    """
    sgd_clf_ = SGDClassifier(loss="log", max_iter=100, random_state=42, n_jobs=1)

    svm_clf_ = SVC(gamma="auto", random_state=42, )

    log_clf_ = LogisticRegression(solver="liblinear", random_state=42, )

    knn_clf_ = KNeighborsClassifier(n_jobs=1, weights='distance', n_neighbors=4)

    bag_clf_ = BaggingClassifier(
        DecisionTreeClassifier(random_state=42, ), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=1, random_state=42)

    tree_clf_ = DecisionTreeClassifier(random_state=42)

    rnd_clf_ = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    svm_clf_ = SVC(gamma="auto", random_state=42)

    voting_clf_h = VotingClassifier(
        estimators=[('lr', log_clf_), ('rf', rnd_clf_), ('sgd', sgd_clf_)],
        voting='hard')

    voting_clf_s = VotingClassifier(
        estimators=[('lr', log_clf_), ('rf', rnd_clf_), ('sgd', sgd_clf_)],
        voting='soft')

    ada_clf_ = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42)

    # xgb_clf_ = XGBClassifier(random_state=42)

    def sgd_clf(self):
        model_train_output(self.sgd_clf_, train_test_dataset, "sgd_clf")

    def svm_clf(self):
        model_train_output(self.svm_clf_, train_test_dataset, "SVC")

    def knn_clf(self):
        model_train_output(self.knn_clf_, train_test_dataset, "knn_clf")

    def log_clf(self):
        model_train_output(self.log_clf_, train_test_dataset, "log_clf")

    def rnd_clf(self):
        model_train_output(self.rnd_clf_, train_test_dataset, "rnd_clf")

    def bag_clf(self):
        model_train_output(self.bag_clf_, train_test_dataset, "bag_clf")

    def tree_clf(self):
        model_train_output(self.tree_clf_, train_test_dataset, "tree_clf")

    def voting_clf_hard(self):
        model_train_output(self.voting_clf_h, train_test_dataset, "voting_clf_hard")

    def voting_clf_soft(self):
        model_train_output(self.voting_clf_s, train_test_dataset, "voting_clf_soft")

    def ada_clf(self):
        model_train_output(self.ada_clf_, train_test_dataset, "ada_clf")

    def xgb_clg(self):
        model_train_output(self.xgb_clf_, train_test_dataset, "xgb_clg")


class MyManager(BaseManager):
    pass


MyManager.register('multiprocess_model_train', multiprocess_model_train)


def model_grid_search():
    """
    对三个模型进行参数搜素调优
    :return:
    """
    estimators = {
        "sgd": SGDClassifier(random_state=42, n_jobs=1),
        "log_clf": LogisticRegression(solver="liblinear", random_state=42, ),
        "tree_clf": DecisionTreeClassifier(random_state=42)}

    param_grids = {
        "sgd": [{"loss": ["log", "perceptron", "squared_hinge"], "max_iter": [5, 10, 100, 200],
                 "penalty": ['l2', "l1", "None", "elasticnet"],
                 "alpha": [0.001, 0.001, 0.01], "learning_rate": ['optimal', 'adaptive'],
                 "eta0": [0.01],
                 "early_stopping": [True]
                 }, {"loss": ["log"], "max_iter": [100]}],
        "log_clf": {"C": [10., 30., 100., 300., 1000., 3000],
                    "intercept_scaling": [1], "max_iter": [5, 10, 100], "multi_class": ['warn'],
                    "penalty": ['l2', "l1"], "solver": ['liblinear'], },
        "tree_clf": {"class_weight": [None, "balanced"], "criterion": ['gini'], "max_depth": [None, 8, 9, 10, 11, 12],
                     "max_features": [None, "auto", "sqrt", "log2"], "max_leaf_nodes": [None],
                     "min_impurity_decrease": [0.0], "min_impurity_split": [None],
                     "min_samples_leaf": [1, 2, 3, 4, 5], "min_samples_split": [2, 4, 6, 8],
                     "min_weight_fraction_leaf": [0.0], "presort": [False], "random_state": [42],
                     "splitter": ['best', "random"]}}
    for name, value in estimators.items():
        model_grid(estimators[name], param_grids[name])
        pass

    # print(estimators)


def pool_train():
    """
    开辟进程池，多进程训练多个分类器。
    :return:
    """
    manager = MyManager()
    manager.start()
    f1 = manager.multiprocess_model_train()
    pool = Pool(processes=6)
    pool.apply_async(f1.sgd_clf())
    pool.apply_async(f1.svm_clf)
    pool.apply_async(f1.knn_clf)
    pool.apply_async(f1.log_clf)
    # pool.apply_async(f1.bag_clf)
    # pool.apply_async(f1.tree_clf)
    # pool.apply_async(f1.rnd_clf)
    # pool.apply_async(f1.voting_clf_hard)
    # pool.apply_async(f1.voting_clf_soft)
    # pool.apply_async(f1.ada_clf)
    # pool.apply_async(f1.xgb_clg)
    pool.close()
    pool.join()


def main():
    pool_train()
    # model_grid_search()
    pass


if __name__ == "__main__":
    main()
