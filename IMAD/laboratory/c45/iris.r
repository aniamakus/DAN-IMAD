library(RWeka)

#iris <- read.csv('data/iris.data.txt', header=TRUE)
#tree <- J48(Class~., data=iris)
#plot(tree)
#summary(tree)
#evaluate_Weka_classifier(tree, numFolds=2)


make_tree <- function(filepath) {
  dataset <- read.csv(filepath, 
                      header=TRUE,
                      colClasses = c("Class"="factor"))
  tree <- J48(Class~., data=dataset)
  
  x11()
  plot(tree)
  
  summary(tree)
}


for(ds_filepath in c("glass", "iris", "pima-indians-diabetes", "wine")) {
  make_tree(paste("data/", ds_filepath, ".data.txt", sep=''))  
}
