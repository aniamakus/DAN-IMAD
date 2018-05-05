loadDataset <- function(filepath) {
  return(read.csv(filepath,
                  header=TRUE,
                  colClasses = c("Class"="factor")));
}

evaluateAllAlgorithms <- function(dataset_scaled, classes, min_nb_cluster, max_nb_cluster, algorithms) {
  algorithm_metrics_tuples <- list()
  
  for(alg in algorithms) {
    alg_name <- alg[[1]]
    alg_fn <- alg[[2]]
    
    metrics <- evaluateClustAlg(dataset_scaled, classes,
                                min_nb_cluster, max_nb_cluster,
                                alg_fn)
    
    curr_metrics <- list(metrics)
    names(curr_metrics) <- alg_name
    algorithm_metrics_tuples <- append(algorithm_metrics_tuples, curr_metrics)
  }
  
  return(algorithm_metrics_tuples)
}

evaluateClustAlg <- function(dataset, classes, min_k, max_k, algorithm) {
  metrics <- data.frame()
  
  for(k in min_k:max_k) {
    fit <- algorithm(dataset_scaled, k)
    
    m <- data.frame(
      intCriteria(dataset_scaled, fit$cluster, c('Davies_Bouldin', 'Dunn')),
      extCriteria(classes, fit$cluster, c('Rand')),
      list('purity'=calc_purity(fit$cluster, classes))
    )
    
    metrics <- rbind(metrics, m)
  }
  
  rownames(metrics) <- min_k:max_k
  return(metrics)
}

make_clust_metric_plots <- function(algorithm_metrics_tuples, min_x, max_x, savepath) {
  png(savepath, width=1024, height=768)
  #x11(width=10, height=7)
  
  metric_names <- c("davies_bouldin", "dunn", "rand", "purity")
  colors <- c("red", "green", "blue", "purple", "orange")
  par(mfrow=c(1, 4))
  
  for(mn in metric_names) {
    plot(1, type="n", main=mn, xlab="", ylab="", xlim=c(min_x, max_x), ylim=c(0,2))
    i <- 1
    for(amt in algorithm_metrics_tuples) {
      values_to_plot <- amt[[mn]]
      values_to_plot[values_to_plot > 2] <- 3
      points(min_x:max_x, values_to_plot, col=colors[[i]], pch=20)
      lines(min_x:max_x, values_to_plot, col=colors[[i]], lty=2)
      i <- i + 1
    }
  }
  
  legend(x="topright", legend=names(algorithm_metrics_tuples), col=colors, lwd=2)
  dev.off()
}

calc_purity <- function(clusters, classes) {
  sum(apply(table(classes, clusters), 2, max)) / length(clusters)  
}

make_scatter_plot <- function(dataset_scaled, classes, algorithm, k, savepath) {
  png(savepath, width=1024, height=768)
  #x11(width=10, height=7)
  
  colors <- rainbow(k)
  fit <- algorithm(dataset_scaled, k)
  
  pairs(dataset_scaled, pch=19, lower.panel=NULL, col=colors[classes], cex=0.7)
  dev.off()
}

make_cluster_plot <- function(dataset_scaled, classes, algorithm, k, title, savepath) {
  png(savepath, width=1024, height=768)
  #x11(width=10, height=7)
  
  fit <- algorithm(dataset_scaled, k)
  
  clusplot(dataset_scaled, fit$cluster, main=sprintf('Cluster plot: %s', title))
  dev.off()
}

make_latex_table <- function(algorithm_metric_tuples, algorithms) {
  summaries <- data.frame()
  for(alg in algorithms) {
    alg_name <- alg[[1]]
    
    df <- data.frame(round(algorithm_metrics_tuples[[alg_name]], 3))
    df[df > 3] <- 'INF'
    names(df) <- names(algorithm_metrics_tuples[[alg_name]])
    
    df <- data.frame(list('alg'=alg_name), 
                     list('nb_clus'=rownames(algorithm_metrics_tuples[[alg_name]])), 
                     df)
    
    summaries <- rbind(summaries, df)
  }
  return(xtable(summaries))
}