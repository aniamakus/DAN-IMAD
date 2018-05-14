"""
KNN algorithm tests
"""
from knn import evaluation as eval
from knn import export as exp
from knn import loader as ldr
from knn import visualization as vis


def main():
    run_tests()


def run_tests():
    # x, y, _ = ldr.load_dataset('iris')
    # scores = eval.evaluate_knn_all_options(x, y, fold_nums=list(range(2, 10)))
    # print(scores)
    # exp.results_to_latex('iris', scores)
    # #exp.results_to_file('iris', scores)
    # #vis.make_scoring_plots('iris', scores)

    dataset_names = ('diabetes', 'glass', 'seeds', 'wine')
    for name in dataset_names:
        print('Dataset:', name)
        x, y, _ = ldr.load_dataset(name)

        scores = eval.evaluate_knn_all_options(x, y, fold_nums=[5])
        print(scores)
        vis.make_scoring_barplots(name, scores)


if __name__ == "__main__":
    main()
