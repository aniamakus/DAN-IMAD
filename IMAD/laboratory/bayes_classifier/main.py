from bc.evaluation import evaluate_classifier
from bc import loader


def main():
    x, y = loader.load_iris()
    cv_min = 2
    cv_max = 10

    result = evaluate_classifier(x, y, discretize=("caim", y),#discretize=False,
                                 cv_min=cv_min, cv_max=cv_max)

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, plots = plt.subplots(1, 4, sharex=True, sharey=True)
    i = 0

    for key, val in result.items():
        if key == 'cnf_matrix':
            continue
        print(key, val)
        sns.pointplot(list(range(cv_min, cv_max + 1)), val, ax=plots[i])
        plots[i].set(title=key)
        i += 1
    plt.show()
    #evaluate_classifier(x, y, discretize=("ew", 50), cv=9)


if __name__ == "__main__":
    main()
