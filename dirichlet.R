
library(smotefamily)
library(caret)
library(ROSE)
library(MASS)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(rattle)
library(MCMCpack)
library(ROCR)
library(patchwork)
library(kknn)
library(class) # Ensure the class library is loaded for kNN


# our new function, there have to be at least  
# 2 points within the rare class
SMOTE.DIRICHLET <- function (X, target, K = 3, dup_size = 0) 
{
  ncD = ncol(X)
  n_target = table(target)
  minority_class = names(which.min(n_target))
  # points of the rare class
  minority_set = subset(X, target == names(which.min(n_target)))[sample(min(n_target)), 
  ]
  # points of the majority class
  majority_set = subset(X, target != names(which.min(n_target)))
  
  P_class = rep(names(which.min(n_target)), nrow(minority_set))
  N_class = target[target != names(which.min(n_target))]
  size_minority_class = nrow(minority_set)
  K <- min(size_minority_class - 1, K) #such that if we have <5 rare cases it works
  size_majority_class = nrow(majority_set)
  knear = knearest(minority_set, minority_set, K)
  
  
  
  # sum_dup is the number of new points to be generated for each point
  # If dup_size is zero, it returns the number of rounds 
  # to duplicate positive to nearly equal to the number of negative instances
  # (50% rare, 50% common)
  sum_dup = n_dup_max(size_minority_class + size_majority_class, size_minority_class, size_majority_class, dup_size)
  syn_dat = NULL
  for (i in 1:size_minority_class) {
    
    # matrix of sum_dup rows
    # each row is a vector of k weights that sum to 1
    g <- matrix(0, nrow = sum_dup, ncol = K)
    for(j in 1:sum_dup) {
      # Why use 4?
      g[j, ] <- MCMCpack::rdirichlet(1, rep(4, K))
    }
    
    # multiplies the sum_dup weights for the k neighbors
    # in this way I obtain sum_dup new points
    # I put the check in the case of K=1
    if (K==1){
      syn_i = g %*% matrix(as.matrix(minority_set[knear[i],]), ncol = ncD, byrow = FALSE)
    }else{
      syn_i = g %*% matrix(as.matrix(minority_set[knear[i, ],]), ncol = ncD, byrow = FALSE)
    }
    
    syn_dat = rbind(syn_dat, syn_i)
  }
  
  minority_set[, ncD + 1] = P_class
  colnames(minority_set) = c(colnames(X), "class")
  majority_set[, ncD + 1] = N_class
  colnames(majority_set) = c(colnames(X), "class")
  rownames(syn_dat) = NULL
  syn_dat = data.frame(syn_dat)
  syn_dat[, ncD + 1] = rep(names(which.min(n_target)), nrow(syn_dat))
  colnames(syn_dat) = c(colnames(X), "class")
  NewD = rbind(minority_set, syn_dat, majority_set)
  rownames(NewD) = NULL
  D_result = list(data = NewD, syn_data = syn_dat, orig_N = majority_set, 
                  orig_P = minority_set, K = K, K_all = NULL, dup_size = sum_dup, 
                  outcast = NULL, eps = NULL, method = "SMOTE")
  class(D_result) = "gen_data"
  return(D_result)
}

data_generator <- function(pi, train_size, mu_0, cov_matrix_0, mu_1, cov_matrix_1, seed = 17){
  set.seed(seed)
  trainsets <- list()
  for (prob in pi) {
    for (size in train_size) {
      # Calculate number of samples per class
      n_0 <- round(size * (1 - prob))  # Class 0
      n_1 <- round(size * prob)       # Class 1
      
      # Generate data for class 0
      x_0 <- mvrnorm(n_0, mu_0, cov_matrix_0)
      x_0 <- data.frame(x_0)
      x_0$y <- rep(0, n_0)
      
      # Generate data for class 1
      x_1 <- mvrnorm(n_1, mu_1, cov_matrix_1)
      x_1 <- data.frame(x_1)
      x_1$y <- rep(1, n_1)
      
      # Combine and shuffle
      combined_data <- rbind(x_0, x_1)
      
      # Save train dataset in the list
      dataset_name <- paste0("train_", size, "_", gsub("\\.", "", as.character(prob)))
      trainsets[[dataset_name]] <- combined_data
    }
  }
  return(trainsets)
}

y <- c(0, 1)
train_size <- c(600, 1000, 5000)
pi <- c(0.1, 0.05, 0.025, 0.01)

# Parameters of distribution of the two features
mu_0 <- c(0, 0)
cov_matrix_0 <- matrix(c(1, 0, 0, 1), nrow = 2)

mu_1 <- c(1, 1)
cov_matrix_1 <- matrix(c(1, -0.5, -0.5, 1), nrow = 2)

data <- data_generator(pi, train_size, mu_0, cov_matrix_0, mu_1, cov_matrix_1)

data <- data$train_600_001  



x <- data[,1:2]
y <- data$y
dirichlet <- SMOTE.DIRICHLET(x, y, K=3)
syn_data <- dirichlet$syn_data


p1 <- ggplot(data, aes(x = X1, y = X2, color = factor(y))) + 
  geom_point(aes(size = factor(y)), alpha = 0.8) +  # Map size to factor(y)
  scale_color_manual(values = c("grey", "blue")) +  # Assign grey and blue colors
  geom_point(data = syn_data, aes(x = X1, y = X2, color = factor(class)), shape = 17, alpha = 0.4) +
  scale_size_manual(values = c(1, 4)) +  # Smaller size for grey, larger for blue
  labs(title = "Train Dataset", x = "Feature 1", y = "Feature 2", color = "Class", size = "Class") +
  theme_minimal()

print(p1)



