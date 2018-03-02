from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


def main():
    iris = datasets.load_iris()

    bc = GaussianNB()
    bc = bc.fit(iris.data, iris.target)

    for method in ('accuracy', 'f1'):
        score = cross_val_score(bc, iris.data, iris.target, scoring=method)
        print(method, '=', score)


if __name__ == "__main__":
    main()