# Clustering in R
library(amap)
library(dismo)
library(clusterCrit)
library(cluster)
library(fpc)
library(xtable)

source('utils.r')

set.seed(as.integer(Sys.time()))

dataset_name <- 'seeds'
dataset <- loadDataset(sprintf('data/%s.data.txt', dataset_name))

dataset_scaled <- scale(dataset[, names(dataset) != 'Class'])
classes <- as.integer(dataset$Class)
min_nb_cluster <- 1
max_nb_cluster <- 10
nb_classes <- length(unique(classes))


algorithms <- list(
  c('KMeans-Euc', function(ds, k) Kmeans(ds, k, method='euclidean')),
  c('KMeans-Man', function(ds, k) Kmeans(ds, k, method='manhattan')),
  c('PAM-Euc', function(ds, k) pam(ds, k, metric='euclidean')),
  c('PAM-Man', function(ds, k) pam(ds, k, metric='manhattan'))
)

algorithm_metrics_tuples <- evaluateAllAlgorithms(dataset_scaled, classes,
                                                  min_nb_cluster, max_nb_cluster,
                                                  algorithms)

latex_table <- make_latex_table(algorithm_metrics_tuples, algorithms) 
print(latex_table, 
      include.rownames=FALSE, 
      file=sprintf('out_tables/%s_table.tex', dataset_name))

make_clust_metric_plots(algorithm_metrics_tuples, 
                        min_nb_cluster, max_nb_cluster, 
                        sprintf('out_plots/%s_metrics.png', dataset_name))

for(alg in algorithms) {
  make_scatter_plot(dataset_scaled, classes, 
                    alg[[2]], nb_classes,
                    savepath=sprintf('out_plots/%s_%s_scatter.png', dataset_name, alg[[1]]))
  make_cluster_plot(dataset_scaled, classes,
                    alg[[2]], nb_classes,
                    title=alg[[1]],
                    savepath=sprintf('out_plots/%s_%s_cluster.png', dataset_name, alg[[1]]))
}