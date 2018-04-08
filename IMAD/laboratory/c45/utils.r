library(dismo)

loadDataset <- function(filepath) {
  return(read.csv(filepath,
                  header=TRUE,
                  colClasses = c("Class"="factor")));
}

makeFoldIndexes <- function(dataset, nb_folds, stratified) {
  if(stratified) {
    return(kfold(dataset, k = nb_folds, by = dataset$Class));
  }
  
  return(kfold(dataset, k = nb_folds));
}

crossValidateTree <- function(dataset, nb_folds, stratified, tree_options, is_binary) {
  accuracies = list()
  precisions = list()
  recalls = list()
  f1s = list()
  
  foldIndexes <- makeFoldIndexes(dataset, nb_folds, stratified)
  
  for(i in 1:nb_folds) {
    currFoldIndexes <- which(foldIndexes==i, arr.ind=TRUE)
    
    testData <- dataset[currFoldIndexes, ]
    trainData <- dataset[-currFoldIndexes, ]
    
    tree <- J48(Class~.,
                data=trainData,
                control=tree_options)
    
    y_pred <- predict(tree, newdata = testData)
    cm <- confusionMatrix(y_pred, testData$Class)
    
    accuracies <- append(accuracies, cm$overall["Accuracy"])
    
    if(is_binary) {
      p <- cm$byClass["Precision"]
      r <- cm$byClass["Recall"]
      f1 <- cm$byClass["F1"]
    }
    else {
      p <- cm$byClass[, "Precision"]
      r <- cm$byClass[, "Recall"]
      f1 <- cm$byClass[, "F1"]
    }
    
    precisions <- append(precisions, mean(p, na.rm=TRUE))
    recalls <- append(recalls, mean(r, na.rm=TRUE))
    f1s <- append(f1s, mean(f1, na.rm=TRUE))
  }
  
  cv_data = list(
    "Accuracy"=mean(unlist(accuracies)),
    "Precision"=mean(unlist(precisions)),
    "Recall"=mean(unlist(recalls)),
    "F1"=mean(unlist(f1s))
  )
  
  return(cv_data)
}

exploreCrossvalidationParams <- function(options_metrics_tuples, is_binary) {
  for(n in names(options_metrics_tuples)) {
    omt <- options_metrics_tuples[[n]]
    print(omt$tree_options)
    
    for(i in min_nb_folds:max_nb_folds) {
      print(sprintf("Crossvalidation K=%d", i))
      cv_data <- crossValidateTree(dataset, i, stratified, omt$tree_options, is_binary)
      
      omt$Accuracy <- append(omt$Accuracy, cv_data$Accuracy)
      omt$Precision <- append(omt$Precision, cv_data$Precision)
      omt$Recall <- append(omt$Recall, cv_data$Recall)
      omt$F1 <- append(omt$F1, cv_data$F1)
    }
    
    options_metrics_tuples[[n]] <- omt
  }
  return(options_metrics_tuples)
}

makePlots <- function(options_metrics_tuples, min_x, max_x) {
  metric_names <- c("Accuracy", "Precision", "Recall", "F1")
  colors <- c("red", "green", "blue", "purple", "orange")
  par(mfrow=c(1, 4))
  
  for(mn in metric_names) {
    print(sprintf("Plotting metric: %s", mn))
    
    plot(1, type="n", main=mn, xlab="", ylab="", xlim=c(min_x, max_x), ylim=c(0,1))
    i <- 1
    for(omt in options_metrics_tuples) {
      print(sprintf("Processing: %d", i))
      points(min_x:max_x, omt[[mn]], col=colors[[i]], pch=20)
      lines(min_x:max_x, omt[[mn]], col=colors[[i]])
      i <- i + 1
    }
  }
  
  legend(x="right", legend=names(options_metrics_tuples), col=colors, lwd=2)
}