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

predict_via_clustering <- function(data, classes, clusters, centroids) {
  # Find most occuring class in cluster and assing it to the cluster
  cluster_labels <- c()
  for(cluster_id in unique(clusters)) {
    cluster_classes <- classes[clusters == cluster_id]
    current_cluster_label <- names(which.max(table(cluster_classes)))
    cluster_labels <- append(cluster_labels, current_cluster_label)
  }
  
  y_pred <- c()
  for(idx in 1:nrow(data)) {
    dp <- data[idx]
    # Calc data point <-> centroids distances
    distances <- as.matrix(dist(rbind(dp, centroids)))[1, -1]
    # Assign the nearest cluster label as the predicted y
    current_y <- cluster_labels[[which.min(distances)]]
    
    y_pred <- append(y_pred, current_y)
  }
  
  return(y_pred)
} 

crossvalidate_clustering <- function(data, algorithm, nb_folds) {
  folds <- createFolds(1:nrow(data), k=nb_folds)
  scores_sum <- numeric(11)
  
  for(fold in folds) {
    train_x <- data[-fold, ]
    train_y <- train_x[, 'Class']
    train_x <- scale(train_x[names(train_x) != 'Class'])
    
    test_x <- data[fold,]
    y_true <- test_x[, 'Class']
    test_x <- scale(test_x[names(test_x) != 'Class'])
    
    fit <- algorithm(train_x)
    y_pred <- predict_via_clustering(test_x, train_y, fit$cluster, fit$centers)
    
    cm <- confusionMatrix(as.factor(y_true), as.factor(y_pred))$byClass
    cm[is.na(cm)] <- 0
    cm <- if (class(cm) == "matrix") colMeans(cm) else cm
    
    scores_sum <- scores_sum + cm
  }
  
  scores <- scores_sum / nb_folds
  
  return(scores)
}

perform_and_plot_cv <- function(dataset) {
  algorithm <- function(ds) Kmeans(ds, nb_classes, method='euclidean')
  
  min_x <- 2
  max_x <- 10
  
  accs <- c()
  precisions <- c()
  recalls <- c()
  f1s <- c()
  
  for(nb_folds in min_x:max_x) {
    print(sprintf('Crossvalidation for nb_folds=%d', nb_folds))
    scores <- crossvalidate_clustering(dataset, algorithm, nb_folds)
    
    accs <- append(accs, scores['Balanced Accuracy'])
    precisions <- append(precisions, scores['Precision'])
    recalls <- append(recalls, scores['Recall'])
    f1s <- append(f1s, scores['F1'])
  }
  
  x11()
  colors <- c("red", "green", "blue", "purple", "orange")
  par(mfrow=c(1, 4))
  
  plot(1, type="n", main='Balanced Accuracy', xlab="", ylab="", xlim=c(min_x, max_x), ylim=c(0,1))
  points(min_x:max_x, accs, col=colors[[1]], pch=20)
  lines(min_x:max_x, accs, col=colors[[1]], lty=2)
  text(min_x:max_x, accs, round(accs, 2), pos=3)
  
  plot(1, type="n", main='Precision', xlab="", ylab="", xlim=c(min_x, max_x), ylim=c(0,1))
  points(min_x:max_x, precisions, col=colors[[2]], pch=20)
  lines(min_x:max_x, precisions, col=colors[[2]], lty=2)
  text(min_x:max_x, precisions, round(precisions, 2), pos=3)
  
  plot(1, type="n", main='Recall', xlab="", ylab="", xlim=c(min_x, max_x), ylim=c(0,1))
  points(min_x:max_x, recalls, col=colors[[3]], pch=20)
  lines(min_x:max_x, recalls, col=colors[[3]], lty=2)
  text(min_x:max_x, recalls, round(recalls, 2), pos=3)
  
  plot(1, type="n", main='F1', xlab="", ylab="", xlim=c(min_x, max_x), ylim=c(0,1))
  points(min_x:max_x, f1s, col=colors[[4]], pch=20)
  lines(min_x:max_x, f1s, col=colors[[4]], lty=2)
  text(min_x:max_x, f1s, round(f1s, 2), pos=3)
}