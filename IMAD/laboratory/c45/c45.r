source('utils.r')
library(RWeka)
library(caret)
library(gridExtra)

filepath <- "data/glass2.data.txt"
dataset <- loadDataset(filepath)
is_binary <- FALSE


#plot_tree(dataset, Weka_control(), savepath="out_plots/tree_default.png")

# plot_tree(dataset, Weka_control(R = TRUE, N = 2), savepath="out_plots/tree_pruned_low_folds.png")
# plot_tree(dataset, Weka_control(R = TRUE, N = 50), savepath="out_plots/tree_pruned_high_folds.png")

# plot_tree(dataset, Weka_control(M = 1), savepath="out_plots/tree_low_leaf.png")
# plot_tree(dataset, Weka_control(M = 10), savepath="out_plots/tree_high_leaf.png")

# plot_tree(dataset, Weka_control(C = 0.01), savepath="out_plots/tree_low_conf.png")
# plot_tree(dataset, Weka_control(C = 0.5), savepath="out_plots/tree_low_conf.png")
break

min_nb_folds <- 2
max_nb_folds <- 9
stratified <- FALSE

# WEKA_CONTROL PARAMETERS
# Use reduced error pruning: R = TRUE, FALSE  --> (DEFAULT: FALSE)
# Set number of folds for reduced error pruning: N = 2 | 10 --> (DEFAULT: 3)

# Set minimum number of instances per leaf: M = 1 | 10 ---> (DEFAULT: 2)

# Set confidence threshold for pruning: C = 0..1 --> (DEFAULT: 0.25)
options_metrics_tuples <- list(
  "C1"=list(
    "tree_options"=Weka_control(),
    "Accuracy"=list(),
    "Precision"=list(),
    "Recall"=list(),
    "F1"=list()
  ),
  "C2"=list(
    "tree_options"=Weka_control(M=10),
    "Accuracy"=list(),
    "Precision"=list(),
    "Recall"=list(),
    "F1"=list()
  )
)

options_metrics_tuples <- exploreCrossvalidationParams(options_metrics_tuples, is_binary)

x11()
makePlots(options_metrics_tuples, min_nb_folds, max_nb_folds)
