"""
KNN algorithm tests
"""
from knn import evaluation as eval
from knn import export as exp
from knn import loader as ldr
from knn import visualization as vis
import pandas
from sklearn import preprocessing

def main():
    run_tests()


def run_tests():
    # x, y, _ = ldr.load_dataset('iris')
    #
    # scores = eval.evaluate_knn_all_options(x, y, fold_nums=list(range(2, 10)))
    # print(scores)
    # #exp.results_to_latex('iris', scores)
    # #exp.results_to_file('iris', scores)
    # vis.make_scoring_plots('iris', scores)

    dataset_names = ('wine',)  #('diabetes', 'glass', 'seeds', 'wine')
    for name in dataset_names:
        print('Dataset:', name)
        x, y, _ = ldr.load_dataset(name)

        xx = x.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(xx)
        x = pandas.DataFrame(x_scaled)

        scores = eval.evaluate_knn_all_options(x, y, fold_nums=[5])
        print(scores)
        vis.make_scoring_barplots(name, scores)


if __name__ == "__main__":
    main()
