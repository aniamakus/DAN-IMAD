library(RWeka)

make_tree <- function(filepath) {
  dataset <- read.csv(filepath,
                      header=TRUE,
                      colClasses = c("Class"="factor"))
  tree <- J48(Class~.,
              data=dataset,
              control=Weka_control())
  # WEKA_CONTROL PARAMETERS
  # Use reduced error pruning: R = TRUE, FALSE  --> (DEFAULT: FALSE)
  # Set number of folds for reduced error pruning: N = 2 | 10 --> (DEFAULT: 3)
  
  # Set minimum number of instances per leaf: M = 1 | 10 ---> (DEFAULT: 2)
  
  # Set confidence threshold for pruning: C = 0..1 --> (DEFAULT: 0.25)
  
  #x11()
  #plot(tree)
  
  summary(tree)
  print(tree)
  
  e <- evaluate_Weka_classifier(tree, numFolds = 8, class = TRUE)
  summary(e)
  print(e)
}


#for(ds_filepath in c("glass", "iris", "pima-indians-diabetes", "wine")) {
#  make_tree(paste("data/", ds_filepath, ".data.txt", sep=''))  
#}

make_tree("data/iris.data.txt")

