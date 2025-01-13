#install.packages("smotefamily")
#install.packages("caret")
#install.packages("ROSE")
#install.packages("MCMCpack")
#install.packages("rpart.plot")
library(smotefamily)
library(caret)
library(ROSE)
library(MASS)
library(ggplot2)
#library(reshape)
library(rpart)
library(rpart.plot)
library(rattle)
library(MCMCpack)

# our new function, for now it doesn't work if there are less than K 
# points within the rare class
SMOTE.DIRICHLET <- function (X, target, K = 5, dup_size = 0) 
{
  ncD = ncol(X)
  n_target = table(target)
  classP = names(which.min(n_target))
  # points of the rare class
  P_set = subset(X, target == names(which.min(n_target)))[sample(min(n_target)), 
  ]
  N_set = subset(X, target != names(which.min(n_target)))
  P_class = rep(names(which.min(n_target)), nrow(P_set))
  N_class = target[target != names(which.min(n_target))]
  sizeP = nrow(P_set)
  sizeN = nrow(N_set)
  knear = knearest(P_set, P_set, K)
  # sum_dup is the number of new points to be generated for each point
  sum_dup = n_dup_max(sizeP + sizeN, sizeP, sizeN, dup_size)
  syn_dat = NULL
  for (i in 1:sizeP) {
    
    # matrix of sum_dup rows
    # each row is a vector of k weights that sum to 1
    g <- matrix(0, nrow = sum_dup, ncol = K)
    for(i in 1:sum_dup) {
      g[i, ] <- MCMCpack::rdirichlet(1, rep(1, K))
    }
    
    # multiplies the sum_dup weights for the k neighbors
    # in this way I obtain sum_dup new points
    syn_i = g %*% matrix(as.matrix(P_set[knear[i, ],]), ncol = ncD, byrow = TRUE)
    
    syn_dat = rbind(syn_dat, syn_i)
  }
  
  P_set[, ncD + 1] = P_class
  colnames(P_set) = c(colnames(X), "class")
  N_set[, ncD + 1] = N_class
  colnames(N_set) = c(colnames(X), "class")
  rownames(syn_dat) = NULL
  syn_dat = data.frame(syn_dat)
  syn_dat[, ncD + 1] = rep(names(which.min(n_target)), nrow(syn_dat))
  colnames(syn_dat) = c(colnames(X), "class")
  NewD = rbind(P_set, syn_dat, N_set)
  rownames(NewD) = NULL
  D_result = list(data = NewD, syn_data = syn_dat, orig_N = N_set, 
                  orig_P = P_set, K = K, K_all = NULL, dup_size = sum_dup, 
                  outcast = NULL, eps = NULL, method = "SMOTE")
  class(D_result) = "gen_data"
  return(D_result)
}


###############################################################################

# simulate data
# where each dataset has different parameters
# (size of the dataset and proportion of rare examples over all examples)

# Parameters
# y = 0 most frequent class
# y = 1 less frequent class
y <- c(0, 1)
train_size <- c(250, 1000, 5000)
pi <- c(0.1, 0.05, 0.025, 0.01)

# Parameters of distribution of the two features
mu_0 <- c(0, 0)
cov_matrix_0 <- matrix(c(1, 0, 0, 1), nrow = 2)

mu_1 <- c(1, 1)
cov_matrix_1 <- matrix(c(1, -0.5, -0.5, 1), nrow = 2)

set.seed(123)

# Compute all train datasets (and corresponding test sets)
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
    shuffled_data <- combined_data[sample(nrow(combined_data)), ]
    
    # Dynamically name the dataframe
    df_name <- paste0("data_", size, "_", gsub("\\.", "", as.character(prob)))
    assign(df_name, shuffled_data)
  }
  # With the same method and same parameter pi, compute test set
  n_0 <- round(250 * (1 - prob))  
  n_1 <- round(250 * prob)       
  x_0 <- mvrnorm(n_0, mu_0, cov_matrix_0)
  x_0 <- data.frame(x_0)
  x_0$y <- rep(0, n_0)
  x_1 <- mvrnorm(n_1, mu_1, cov_matrix_1)
  x_1 <- data.frame(x_1)
  x_1$y <- rep(1, n_1)
  combined_data <- rbind(x_0, x_1)
  shuffled_data <- combined_data[sample(nrow(combined_data)), ]
  df_name <- paste0("test_", gsub("\\.", "", as.character(prob)))
  assign(df_name, shuffled_data)
}

###############################################################################
# For now it is tested only on one dataset

# (train sets will be different,
# but test set is the same for all methods)
data_5000_01$y <- as.factor(data_5000_01$y) # for some reason it needs this???
# balance dataset with SMOTE
IR <- nrow(data_5000_01[data_5000_01$y == 1, ]) / nrow(data_5000_01)
IR
set.seed(123) # do we need to put it everytime?
data.smote <- SMOTE(data_5000_01[,-3], data_5000_01[,3], K = 5, dup_size = 0)$data
data.smote$class <- factor(data.smote$class) #?????
new_IR <- nrow(data.smote[data.smote$class == 1, ]) / nrow(data.smote)
new_IR

# balance dataset with SMOTE variant
set.seed(123)
data.smote.dirichlet <- SMOTE.DIRICHLET(data_5000_01[,-3], data_5000_01[,3], K = 5, dup_size = 0)$data
data.smote.dirichlet$class <- factor(data.smote.dirichlet$class) #?????
new_IR <- nrow(data.smote.dirichlet[data.smote.dirichlet$class == 1, ]) / nrow(data.smote.dirichlet)
new_IR



###############################################################################
#apply models (tree and logistic regression) to all train sets

# model trained on original unbalanced data
tree <- rpart(y ~ ., data = data_5000_01)
#par(mfrow=c(1,1))
#plot(tree) # this doesn't work if the tree is formed by only the root
#text(tree, cex=.5)
#fancyRpartPlot(tree)

# model trained on data balanced using smote
tree.smote <- rpart(class ~ ., data = data.smote)

# model trained on data balanced using smote dirichlet
tree.smote.dirichlet <- rpart(class ~ ., data = data.smote.dirichlet)

# predict all models
pred.tree <- predict(tree, newdata = test_01,type = "prob")
pred.tree.smote <- predict(tree.smote, newdata = test_01,type = "prob")
pred.tree.smote.dirichlet <- predict(tree.smote.dirichlet, newdata = test_01)




fit <- glm(y ~ . , data = data_5000_01, family = binomial(link = "logit"))
fit.smote <- glm(class ~ . , data = data.smote, family = binomial(link = "logit"))
fit.smote.dirichlet <- glm(class ~ . , data = data.smote.dirichlet, family = binomial(link = "logit"))
prob_class_1 <- predict(fit, newdata = test_01, type= "response")      
prob_class_0 <- 1 - prob_class_1   
pred.fit <- cbind(prob_class_0, prob_class_1)

prob_class_1 <- predict(fit.smote, newdata = test_01, type="response")      
prob_class_0 <- 1 - prob_class_1   
pred.fit.smote <- cbind(prob_class_0, prob_class_1)

prob_class_1 <- predict(fit.smote.dirichlet, newdata = test_01, type="response")      
prob_class_0 <- 1 - prob_class_1   
pred.fit.smote.dirichlet <- cbind(prob_class_0, prob_class_1)


###############################################################################
#compute metrics to compare our variant to the original technique

accuracy.meas(response = test_01$y, predicted = pred.tree[,2])
accuracy.meas(response = test_01$y, predicted = pred.tree.smote[,2])
accuracy.meas(response = test_01$y, predicted = pred.tree.smote.dirichlet[,2])

roc.curve(test_01$y, pred.tree[,2])
roc.curve(test_01$y, pred.tree.smote[,2],add.roc = TRUE, col = 2)
roc.curve(test_01$y, pred.tree.smote.dirichlet[,2], add.roc = TRUE, col = 3)



accuracy.meas(response = test_01$y, predicted = pred.fit[,2])
accuracy.meas(response = test_01$y, predicted = pred.fit.smote[,2])
accuracy.meas(response = test_01$y, predicted = pred.fit.smote.dirichlet[,2])

roc.curve(test_01$y, pred.fit[,2])
roc.curve(test_01$y, pred.fit.smote[,2],add.roc = TRUE, col = 2)
roc.curve(test_01$y, pred.fit.smote.dirichlet[,2], add.roc = TRUE, col = 3)