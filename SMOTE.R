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
library(rpart)
library(rpart.plot)
library(rattle)
library(MCMCpack)
library(ROCR)
library(patchwork)


# our new function, there have to be at least  
# 2 points within the rare class
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
  K <- min(sizeP - 1, K)
  sizeN = nrow(N_set)
  knear = knearest(P_set, P_set, K)
  
  
  # sum_dup is the number of new points to be generated for each point
  # If dup_size is zero, it returns the number of rounds 
  # to duplicate positive to nearly equal to the number of negative instances
  # (50% rare, 50% common)
  sum_dup = n_dup_max(sizeP + sizeN, sizeP, sizeN, dup_size)
  syn_dat = NULL
  for (i in 1:sizeP) {
    
    # matrix of sum_dup rows
    # each row is a vector of k weights that sum to 1
    g <- matrix(0, nrow = sum_dup, ncol = K)
    for(j in 1:sum_dup) {
      g[j, ] <- MCMCpack::rdirichlet(1, rep(2, K))
    }
    
    # multiplies the sum_dup weights for the k neighbors
    # in this way I obtain sum_dup new points
    # I put the check in the case of K=1
    if (K==1){
      syn_i = g %*% matrix(as.matrix(P_set[knear[i],]), ncol = ncD, byrow = F)
    }else{
      syn_i = g %*% matrix(as.matrix(P_set[knear[i, ],]), ncol = ncD, byrow = F)
    }
    
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

# Parameters
# y = 0 most frequent class
# y = 1 less frequent class
y <- c(0, 1)
train_size <- c(600, 1000, 5000)
pi <- c(0.1, 0.05, 0.025)

# Parameters of distribution of the two features
mu_0 <- c(0, 0)
cov_matrix_0 <- matrix(c(1, 0, 0, 1), nrow = 2)

mu_1 <- c(1, 1)
cov_matrix_1 <- matrix(c(1, -0.5, -0.5, 1), nrow = 2)

mu_out <- c(-3,-3)
cov_matrix_out <- matrix(c(1,-0.5,0.5,1), nrow = 2)


# Create a list of 9 lists to store results for each trainset
results <- vector("list", 9)  # One entry for each trainset
results_outliers <- vector("list", 9)
names(results) <- paste0("Trainset_", 1:9)
names(results_outliers) <- paste0("Trainset_", 1:9)

### NUMBER OF SIMULATIONS
n_simulations = 100


# Loop through each trainset and initialize model results
for (l in 1:9) {
  results[[l]] <- list(
    logistic_regressor = vector("list", 3),         # 3 versions for KNN
    decision_tree = vector("list", 3)        # 3 versions for decision tree
  )
  results_outliers[[l]] <- list(
    logistic_regressor = vector("list", 3),         # 3 versions for KNN
    decision_tree = vector("list", 3)        # 3 versions for decision tree
  )
  
  # Loop through each model type and each version to initialize the sublists
  for (model_type in c("logistic_regressor", "decision_tree")) {
    for (version in 1:3) {
      results[[l]][[model_type]][[version]] <- list(
        auc = numeric(n_simulations),        # AUC values
        precision = numeric(n_simulations),   # Precision values
        recall = numeric(n_simulations),      # Recall values
        f1 = numeric(n_simulations)           # F1 values
      )
    }
  }
  for (model_type in c("logistic_regressor", "decision_tree")) {
    for (version in 1:3) {
      results_outliers[[l]][[model_type]][[version]] <- list(
        auc = numeric(n_simulations),        # AUC values
        precision = numeric(n_simulations),   # Precision values
        recall = numeric(n_simulations),      # Recall values
        f1 = numeric(n_simulations)           # F1 values
      )
    }
  }
}


for (k in 1:n_simulations){
  trainsets <- list()
  testsets <- list()
  trainsets_outliers <- list()
  testsets_outliers <- list()
  
  force_matrix <- function(x) {
    if (is.vector(x)) {
      x <- t(x)  # Convert vector to single-row matrix
    }
    return(as.data.frame(x))
  }
  
  safe_mvrnorm <- function(n, mu, sigma) {
    # Check if 'mu' is a valid vector of length 2
    if (length(mu) != 2) stop("mu must be a 2-dimensional vector!")
    
    # Check if 'sigma' is a valid 2x2 matrix
    if (ncol(sigma) != 2 || nrow(sigma) != 2) stop("sigma must be a 2x2 matrix!")
    
    if (n <= 0) return(data.frame(matrix(nrow = 0, ncol = length(mu))))
    
    MASS::mvrnorm(n, mu, sigma)
  }
  
  # Simulate data for all train datasets (and corresponding test sets)
  for (prob in pi) {
    for (size in train_size) {
      # Calculate number of samples per class
      n_0 <- round(size * (1 - prob))  # Class 0
      n_1 <- round(size * prob)       # Class 1
      
      # Generate data for class 0
      x_0 <- safe_mvrnorm(n_0, mu_0, cov_matrix_0)
      x_0 <- data.frame(x_0)
      x_0$y <- rep(0, n_0)
      
      # Generate data for class 1
      x_1 <- safe_mvrnorm(n_1, mu_1, cov_matrix_1)
      x_1 <- data.frame(x_1)
      x_1$y <- rep(1, n_1)
      
      # Combine
      combined_data <- rbind(x_0, x_1)
      
      # Save train dataset in the list
      dataset_name <- paste0("train_", size, "_", gsub("\\.", "", as.character(prob)))
      trainsets[[dataset_name]] <- combined_data
      
      
      # outliers ----------------------------------------------------------------
      
      n_0_outliers <- round(size * (1-prob))
      n_1_outliers <- round(size * prob)
      
      prop <- 0.04
      n_1_non_out <- round(n_1_outliers * (1-prop))
      n_1_out <- round(n_1_outliers * prop)
      
      x_0_outliers <- if (n_0 > 0) {
        data.frame(safe_mvrnorm(n_0, mu_0, cov_matrix_0), y = rep(0, n_0))
      } else {
        data.frame(matrix(nrow = 0, ncol = 3))  # Ensure 3 columns: X1, X2, y
      }
      x_1_non_out <- force_matrix(safe_mvrnorm(n_1_non_out, mu_1, cov_matrix_1))
      x_1_out <- force_matrix(safe_mvrnorm(n_1_out, mu_out, cov_matrix_out))
      
      colnames(x_1_non_out) <- colnames(x_1_out) <- c('X1', 'X2')
      
      # Combine class 1 data
      x_1_outliers <- rbind(x_1_non_out, x_1_out)
      x_1_outliers$y <- rep(1, n_1_outliers)
      
      combined_data_outliers <- rbind(x_0_outliers, x_1_outliers)
      
      dataset_name_outliers <- paste0("train_", size, "_", gsub("\\.", "", as.character(prob)))
      trainsets_outliers[[dataset_name_outliers]] <- combined_data_outliers
    }
    
    
    # testsets ----------------------------------------------------------------
    
    # With the same method and same parameter pi, compute test set
    n_0 <- round(600 * (1 - prob))  
    n_1 <- round(600 * prob)       
    x_0 <- safe_mvrnorm(n_0, mu_0, cov_matrix_0)
    x_0 <- data.frame(x_0)
    x_0$y <- rep(0, n_0)
    x_1 <- safe_mvrnorm(n_1, mu_1, cov_matrix_1)
    x_1 <- data.frame(x_1)
    x_1$y <- rep(1, n_1)
    combined_data <- rbind(x_0, x_1)
    
    # Save test dataset in the list
    dataset_name <- paste0("test_", gsub("\\.", "", as.character(prob)))
    testsets[[dataset_name]] <- combined_data
    
    
    # outliers ----------------------------------------------------------------
    
    n_0_outliers <- round(600 * (1-prob))
    n_1_outliers <- round(600 * prob)
    n_1_non_out <- round(n_1_outliers * (1-prop))
    n_1_out <- round(n_1_outliers * prop)
    x_0_outliers <- if (n_0 > 0) {
      data.frame(safe_mvrnorm(n_0, mu_0, cov_matrix_0), y = rep(0, n_0))
    } else {
      data.frame(matrix(nrow = 0, ncol = 3))  # Ensure 3 columns: X1, X2, y
    }
    x_1_non_out <- force_matrix(safe_mvrnorm(n_1_non_out, mu_1, cov_matrix_1))
    x_1_out <- force_matrix(safe_mvrnorm(n_1_out, mu_out, cov_matrix_out))
    
    colnames(x_1_non_out) <- colnames(x_1_out) <- c('X1', 'X2')
    
    # Combine class 1 data
    x_1_outliers <- rbind(x_1_non_out, x_1_out)
    x_1_outliers$y <- rep(1, n_1_outliers)
    
    combined_data_outliers <- rbind(x_0_outliers, x_1_outliers)
    
    dataset_name_outliers <- paste0("test_", "_", gsub("\\.", "", as.character(prob)))
    testsets_outliers[[dataset_name_outliers]] <- combined_data_outliers
    
  }
  
  ###############################################################################
  
  for (i in 1:9){
    trainset <- trainsets[[i]]
    trainset_name <- names(trainsets)[i]
    trainset_outliers <- trainsets_outliers[[i]]
    trainset_outliers_name <- trainsets_outliers[[i]]
    
    p <- ggplot(trainset, aes(x = X1, y = X2, color = factor(y))) + 
      geom_point(aes(size = factor(y)), alpha = 1, show.legend = c(color = TRUE, size = FALSE)) + 
      scale_color_manual(values = c("grey", "darkgreen")) + 
      scale_size_manual(values = c(1, 1.5)) +
      labs(title = "SMOTE w/outliers", x = "Feature 1", y = "Feature 2", color = "Class") +
      theme_minimal()
    #print(p)
    
    p_outliers <- ggplot(trainset_outliers, aes(x = X1, y = X2, color = factor(y))) + 
      geom_point(aes(size = factor(y)), alpha = 1, show.legend = c(color = TRUE, size = FALSE)) + 
      scale_color_manual(values = c("grey", "darkgreen")) + 
      scale_size_manual(values = c(1, 1.5)) +
      labs(title = "SMOTE w/outliers", x = "Feature 1", y = "Feature 2", color = "Class") +
      theme_minimal()
    #print(p_outliers)
    
    combined_data <- p + p_outliers + plot_layout(ncol = 2)
    #print(combined_data)
    
    
    # (train sets will be different,
    # but test set is the same for all methods)
    trainset$y <- as.factor(trainset$y) # for some reason it needs this???
    trainset_outliers$y <- as.factor(trainset_outliers$y)
    # balance dataset with SMOTE
    #IR <- nrow(trainset[trainset$y == 1, ]) / nrow(trainset)
    smote <- SMOTE(trainset[,-3], trainset[,3], K = 5, dup_size = 0)
    smote_outliers <- SMOTE(trainset_outliers[,-3], trainset_outliers[,3], K = 5, dup_size = 0)
    
    data.smote <- smote$data
    data.smote_outliers <- smote_outliers$data
    
    syn.data.smote <- smote$syn_data
    syn.data.smote_outliers <- smote_outliers$syn_data
    
    p1 <- ggplot(data.smote, aes(x = X1, y = X2, color = factor(class))) + 
      geom_point(aes(size = factor(class)), alpha = 0.4, show.legend = c(color = TRUE, size = FALSE)) + 
      scale_color_manual(values = c("grey", "darkgreen")) + 
      scale_size_manual(values = c(1, 1)) +
      geom_point(data = trainset[trainset$y == 1,], aes(x = X1, y = X2), color = "blue", alpha = 1, size = 1.5) +
      labs(title = "SMOTE w/outliers", x = "Feature 1", y = "Feature 2", color = "Class") +
      theme_minimal()
    #print(p1)
    
    p1_outliers <- ggplot(data.smote_outliers, aes(x = X1, y = X2, color = factor(class))) + 
      geom_point(aes(size = factor(class)), alpha = 0.4, show.legend = c(color = TRUE, size = FALSE)) + 
      scale_color_manual(values = c("grey", "darkgreen")) + 
      scale_size_manual(values = c(1, 1)) +
      geom_point(data = trainset_outliers[trainset_outliers$y == 1,], aes(x = X1, y = X2), color = "blue", alpha = 1, size = 1.5) +
      labs(title = "SMOTE w/outliers", x = "Feature 1", y = "Feature 2", color = "Class") +
      theme_minimal()
    #print(p1_outliers)
    
    combined_smote <- p1 + p1_outliers + plot_layout(ncol = 2)
    #print(combined_smote)
    
    data.smote$class <- factor(data.smote$class) 
    data.smote_outliers$class <- factor(data.smote_outliers$class)
    #smote_IR <- nrow(data.smote[data.smote$class == 1, ]) / nrow(data.smote)
    
    # balance dataset with SMOTE variant
    
    smote.dirichlet <- SMOTE.DIRICHLET(trainset[,1:2], trainset$y, K = 5, dup_size = 0)
    smote.dirichlet_outliers <- SMOTE.DIRICHLET(trainset_outliers[,1:2], trainset_outliers$y, K = 5, dup_size = 0)
    
    data.smote.dirichlet <- smote.dirichlet$data
    data.smote.dirichlet_outliers <- smote.dirichlet_outliers$data
    
    syn.data.smote.dirichlet <- smote.dirichlet$syn_data
    syn.data.smote.dirichlet_outliers <- smote.dirichlet_outliers$syn_data
    
    
    p2 <- ggplot(data.smote.dirichlet, aes(x = X1, y = X2, color = factor(class))) + 
      geom_point(aes(size = factor(class)), alpha = 0.4, show.legend = c(color = TRUE, size = FALSE)) + 
      scale_color_manual(values = c("grey", "darkgreen")) + 
      scale_size_manual(values = c(1, 1)) +
      geom_point(data = trainset[trainset$y == 1,], aes(x = X1, y = X2), color = "blue", alpha = 1, size = 1.5) +
      labs(title = "SMOTE w/outliers", x = "Feature 1", y = "Feature 2", color = "Class") +
      theme_minimal()
    #print(p2)
    
    p2_outliers <- ggplot(data.smote.dirichlet_outliers, aes(x = X1, y = X2, color = factor(class))) + 
      geom_point(aes(size = factor(class)), alpha = 0.4, show.legend = c(color = TRUE, size = FALSE)) + 
      scale_color_manual(values = c("grey", "darkgreen")) + 
      scale_size_manual(values = c(1, 1)) +
      geom_point(data = trainset_outliers[trainset_outliers$y == 1,], aes(x = X1, y = X2), color = "blue", alpha = 1, size = 1.5) +
      labs(title = "SMOTE w/outliers", x = "Feature 1", y = "Feature 2", color = "Class") +
      theme_minimal()
    #print(p2_outliers)
    
    combined_dirichlet <- p2 + p2_outliers + plot_layout(ncol = 2)
    print(combined_dirichlet)
    
    
    data.smote.dirichlet$class <- factor(data.smote.dirichlet$class)
    data.smote.dirichlet_outliers$class <- factor(data.smote.dirichlet_outliers$class)
    #dirichlet_IR <- nrow(data.smote.dirichlet[data.smote.dirichlet$class == 1, ]) / nrow(data.smote.dirichlet)
    
    
    
    # apply models (tree and logistic regression) to all train sets -----------
    
    
    # Classification trees
    
    # model trained on original unbalanced data
    tree <- rpart(y ~ ., data = trainset)
    tree_outliers <- rpart(y ~ ., data = trainset_outliers)
    # model trained on data balanced using smote
    tree.smote <- rpart(class ~ ., data = data.smote)
    tree.smote_outliers <- rpart(class ~ ., data = data.smote_outliers)
    # model trained on data balanced using smote dirichlet
    tree.smote.dirichlet <- rpart(class ~ ., data = data.smote.dirichlet)
    tree.smote.dirichlet_outliers <- rpart(class ~ ., data = data.smote.dirichlet_outliers)
    
    # Logistic regression
    
    fit <- glm(y ~ . , data = trainset, family = binomial(link = "logit"))
    fit_outliers <- glm(y ~ . , data = trainset_outliers, family = binomial(link = "logit"))
    fit.smote <- glm(class ~ . , data = data.smote, family = binomial(link = "logit"))
    fit.smote_outliers <- glm(class ~ . , data = data.smote_outliers, family = binomial(link = "logit"))
    fit.smote.dirichlet <- glm(class ~ . , data = data.smote.dirichlet, family = binomial(link = "logit"))
    fit.smote.dirichlet_outliers <- glm(class ~ . , data = data.smote.dirichlet_outliers, family = binomial(link = "logit"))
    
    test_index <- ceiling((i/3))
    testset <- testsets[[test_index]]
    testset_outliers <- testsets_outliers[[test_index]]
    
    # predict all models
    pred.tree <- predict(tree, newdata = testset,type = "prob")
    pred.tree_outliers <- predict(tree_outliers, newdata = testset_outliers,type = "prob")
    pred.tree.smote <- predict(tree.smote, newdata = testset,type = "prob")
    pred.tree.smote_outliers <- predict(tree.smote_outliers, newdata = testset_outliers,type = "prob")
    pred.tree.smote.dirichlet <- predict(tree.smote.dirichlet, newdata = testset)
    pred.tree.smote.dirichlet_outliers <- predict(tree.smote.dirichlet_outliers, newdata = testset_outliers)
    
    prob_class_1 <- predict(fit, newdata = testset, type= "response")   
    prob_class_1_outliers <- predict(fit_outliers, newdata = testset_outliers, type = "response")
    prob_class_0 <- 1 - prob_class_1   
    prob_class_0_outliers <- 1 - prob_class_1_outliers 
    pred.fit <- cbind(prob_class_0, prob_class_1)
    pred.fit_outliers <- cbind(prob_class_0_outliers, prob_class_1_outliers)
    
    prob_class_1 <- predict(fit.smote, newdata = testset, type="response")    
    prob_class_1_outliers <- predict(fit.smote_outliers, newdata = testset_outliers, type="response") 
    prob_class_0 <- 1 - prob_class_1 
    prob_class_0_outliers <- 1 - prob_class_1_outliers
    pred.fit.smote <- cbind(prob_class_0, prob_class_1)
    pred.fit.smote_outliers <- cbind(prob_class_0_outliers, prob_class_1_outliers)
    
    prob_class_1 <- predict(fit.smote.dirichlet, newdata = testset, type="response") 
    prob_class_1_outliers <- predict(fit.smote.dirichlet_outliers, newdata = testset_outliers, type="response")
    prob_class_0 <- 1 - prob_class_1  
    prob_class_0_outliers <- 1 - prob_class_1_outliers  
    pred.fit.smote.dirichlet <- cbind(prob_class_0, prob_class_1)
    pred.fit.smote.dirichlet_outliers <- cbind(prob_class_0_outliers, prob_class_1_outliers)
    
    
    # compute metrics to compare our variant to the original technique --------
    
    
    # Logistic regression
    metrics.fit <- accuracy.meas(response = testset$y, predicted = pred.fit[,2])
    metrics.fit.smote <- accuracy.meas(response = testset$y, predicted = pred.fit.smote[,2])
    metrics.fit.smote.dirichlet <- accuracy.meas(response = testset$y, predicted = pred.fit.smote.dirichlet[,2])
    
    auc.fit <- roc.curve(testset$y, pred.fit[,2], plotit=FALSE, main = paste("ROC Curve - Dataset:", trainset_name, "\nLog regression"))$auc
    auc.fit.smote <- roc.curve(testset$y, pred.fit.smote[,2], plotit=FALSE,add.roc = TRUE, col = 2)$auc
    auc.fit.smote.dirichlet <- roc.curve(testset$y, pred.fit.smote.dirichlet[,2], add.roc = TRUE,plotit=FALSE, col = 3)$auc
    
    
    results[[i]][["logistic_regressor"]][[1]]$auc[k] <- auc.fit
    if (is.nan(metrics.fit$precision)| is.na(metrics.fit$precision)){
      results[[i]][["logistic_regressor"]][[1]]$precision[k] <- 0
    }else{
      results[[i]][["logistic_regressor"]][[1]]$precision[k] <- metrics.fit$precision
    }
    results[[i]][["logistic_regressor"]][[1]]$recall[k] <- metrics.fit$recall
    if (is.nan(metrics.fit$F)| is.na(metrics.fit$F)){
      results[[i]][["logistic_regressor"]][[1]]$f1[k] <- 0
    }else{
      results[[i]][["logistic_regressor"]][[1]]$f1[k] <- metrics.fit$F
    }
    
    
    results[[i]][["logistic_regressor"]][[2]]$auc[k] <- auc.fit.smote
    results[[i]][["logistic_regressor"]][[2]]$precision[k] <- metrics.fit.smote$precision
    results[[i]][["logistic_regressor"]][[2]]$recall[k] <- metrics.fit.smote$recall
    if (is.nan(metrics.fit.smote$F)| is.na(metrics.fit.smote$F)){
      results[[i]][["logistic_regressor"]][[2]]$f1[k] <- 0
    }else{
      results[[i]][["logistic_regressor"]][[2]]$f1[k] <- metrics.fit.smote$F
    }
    
    
    results[[i]][["logistic_regressor"]][[3]]$auc[k] <- auc.fit.smote.dirichlet
    results[[i]][["logistic_regressor"]][[3]]$precision[k] <- metrics.fit.smote.dirichlet$precision
    results[[i]][["logistic_regressor"]][[3]]$recall[k] <- metrics.fit.smote.dirichlet$recall
    if (is.nan(metrics.fit.smote.dirichlet$F)| is.na(metrics.fit.smote.dirichlet$F)){
      results[[i]][["logistic_regressor"]][[3]]$f1[k] <- 0
    }else{
      results[[i]][["logistic_regressor"]][[3]]$f1[k] <- metrics.fit.smote.dirichlet$F
    }
    
    
    
    # Classification trees
    metrics.tree <- accuracy.meas(response = testset$y, predicted = pred.tree[,2])
    metrics.tree.smote <- accuracy.meas(response = testset$y, predicted = pred.tree.smote[,2])
    metrics.tree.smote.dirichlet <- accuracy.meas(response = testset$y, predicted = pred.tree.smote.dirichlet[,2])
    
    auc.tree <- roc.curve(testset$y, pred.tree[,2], main = paste("ROC Curve - Dataset:", trainset_name, "\nTree"), plotit=FALSE)$auc
    auc.tree.smote <- roc.curve(testset$y, pred.tree.smote[,2],add.roc = TRUE, plotit=FALSE, col = 2)$auc
    auc.tree.smote.dirichlet <- roc.curve(testset$y, pred.tree.smote.dirichlet[,2], add.roc = TRUE, plotit=FALSE, col = 3)$auc
    
    
    results[[i]][["decision_tree"]][[1]]$auc[k] <- auc.tree
    if (is.nan(metrics.tree$precision) | is.na(metrics.tree$precision)){
      results[[i]][["decision_tree"]][[1]]$precision[k] <- 0
    }else{
      results[[i]][["decision_tree"]][[1]]$precision[k] <- metrics.tree$precision
    }
    results[[i]][["decision_tree"]][[1]]$recall[k] <- metrics.tree$recall
    if (is.nan(metrics.tree$F)| is.na(metrics.tree$F)){
      results[[i]][["decision_tree"]][[1]]$f1[k] <- 0
    }else{
      results[[i]][["decision_tree"]][[1]]$f1[k] <- metrics.tree$F
    }
    
    results[[i]][["decision_tree"]][[2]]$auc[k] <- auc.tree.smote
    results[[i]][["decision_tree"]][[2]]$precision[k] <- metrics.tree.smote$precision
    results[[i]][["decision_tree"]][[2]]$recall[k] <- metrics.tree.smote$recall
    if (is.nan(metrics.tree.smote$F)| is.na(metrics.tree.smote$F)){
      results[[i]][["decision_tree"]][[2]]$f1[k] <- 0
    }else{
      results[[i]][["decision_tree"]][[2]]$f1[k] <- metrics.tree.smote$F
    }
    
    
    results[[i]][["decision_tree"]][[3]]$auc[k] <- auc.tree.smote.dirichlet
    results[[i]][["decision_tree"]][[3]]$precision[k] <- metrics.tree.smote.dirichlet$precision
    results[[i]][["decision_tree"]][[3]]$recall[k] <- metrics.tree.smote.dirichlet$recall
    if (is.nan(metrics.tree.smote.dirichlet$F)| is.na(metrics.tree.smote.dirichlet$F)){
      results[[i]][["decision_tree"]][[3]]$f1[k] <- 0
    }else{
      results[[i]][["decision_tree"]][[3]]$f1[k] <- metrics.tree.smote.dirichlet$F
    }
    
    
    
    # outliers metrics----------------------------------------------------------------
    
    # Logistic regression
    metrics.fit_outliers <- accuracy.meas(response = testset_outliers$y, predicted = pred.fit_outliers[,2])
    metrics.fit.smote_outliers <- accuracy.meas(response = testset_outliers$y, predicted = pred.fit.smote_outliers[,2])
    metrics.fit.smote.dirichlet_outliers <- accuracy.meas(response = testset_outliers$y, predicted = pred.fit.smote.dirichlet_outliers[,2])
    
    auc.fit_outliers <- roc.curve(testset_outliers$y, pred.fit_outliers[,2], plotit=FALSE, main = paste("ROC Curve - Dataset:", trainset_outliers_name, "\nLog regression"))$auc
    auc.fit.smote_outliers <- roc.curve(testset_outliers$y, pred.fit.smote_outliers[,2], plotit=FALSE,add.roc = TRUE, col = 2)$auc
    auc.fit.smote.dirichlet_outliers <- roc.curve(testset_outliers$y, pred.fit.smote.dirichlet_outliers[,2], add.roc = TRUE,plotit=FALSE, col = 3)$auc
    
    
    results_outliers[[i]][["logistic_regressor"]][[1]]$auc[k] <- auc.fit_outliers
    if (is.nan(metrics.fit_outliers$precision)| is.na(metrics.fit_outliers$precision)){
      results_outliers[[i]][["logistic_regressor"]][[1]]$precision[k] <- 0
    }else{
      results_outliers[[i]][["logistic_regressor"]][[1]]$precision[k] <- metrics.fit_outliers$precision
    }
    results_outliers[[i]][["logistic_regressor"]][[1]]$recall[k] <- metrics.fit_outliers$recall
    if (is.nan(metrics.fit_outliers$F)| is.na(metrics.fit_outliers$F)){
      results_outliers[[i]][["logistic_regressor"]][[1]]$f1[k] <- 0
    }else{
      results_outliers[[i]][["logistic_regressor"]][[1]]$f1[k] <- metrics.fit_outliers$F
    }
    
    
    results_outliers[[i]][["logistic_regressor"]][[2]]$auc[k] <- auc.fit.smote_outliers
    results_outliers[[i]][["logistic_regressor"]][[2]]$precision[k] <- metrics.fit.smote_outliers$precision
    results_outliers[[i]][["logistic_regressor"]][[2]]$recall[k] <- metrics.fit.smote_outliers$recall
    if (is.nan(metrics.fit.smote_outliers$F)| is.na(metrics.fit.smote_outliers$F)){
      results_outliers[[i]][["logistic_regressor"]][[2]]$f1[k] <- 0
    }else{
      results_outliers[[i]][["logistic_regressor"]][[2]]$f1[k] <- metrics.fit.smote_outliers$F
    }
    
    
    # Classification trees
    metrics.tree_outliers <- accuracy.meas(response = testset_outliers$y, predicted = pred.tree_outliers[,2])
    metrics.tree.smote_outliers <- accuracy.meas(response = testset_outliers$y, predicted = pred.tree.smote_outliers[,2])
    metrics.tree.smote.dirichlet_outliers <- accuracy.meas(response = testset_outliers$y, predicted = pred.tree.smote.dirichlet_outliers[,2])
    
    auc.tree_outliers <- roc.curve(testset_outliers$y, pred.tree_outliers[,2], main = paste("ROC Curve - Dataset:", trainset_outliers_name, "\nTree"), plotit=FALSE)$auc
    auc.tree.smote_outliers <- roc.curve(testset_outliers$y, pred.tree.smote_outliers[,2],add.roc = TRUE, plotit=FALSE, col = 2)$auc
    auc.tree.smote.dirichlet_outliers <- roc.curve(testset_outliers$y, pred.tree.smote.dirichlet_outliers[,2], add.roc = TRUE, plotit=FALSE, col = 3)$auc
    
    
    results_outliers[[i]][["decision_tree"]][[1]]$auc[k] <- auc.tree_outliers
    if (is.nan(metrics.tree_outliers$precision) | is.na(metrics.tree_outliers$precision)){
      results_outliers[[i]][["decision_tree"]][[1]]$precision[k] <- 0
    }else{
      results_outliers[[i]][["decision_tree"]][[1]]$precision[k] <- metrics.tree_outliers$precision
    }
    results_outliers[[i]][["decision_tree"]][[1]]$recall[k] <- metrics.tree_outliers$recall
    if (is.nan(metrics.tree_outliers$F)| is.na(metrics.tree_outliers$F)){
      results_outliers[[i]][["decision_tree"]][[1]]$f1[k] <- 0
    }else{
      results_outliers[[i]][["decision_tree"]][[1]]$f1[k] <- metrics.tree_outliers$F
    }
    
    results_outliers[[i]][["decision_tree"]][[2]]$auc[k] <- auc.tree.smote_outliers
    results_outliers[[i]][["decision_tree"]][[2]]$precision[k] <- metrics.tree.smote_outliers$precision
    results_outliers[[i]][["decision_tree"]][[2]]$recall[k] <- metrics.tree.smote_outliers$recall
    if (is.nan(metrics.tree.smote_outliers$F)| is.na(metrics.tree.smote_outliers$F)){
      results_outliers[[i]][["decision_tree"]][[2]]$f1[k] <- 0
    }else{
      results_outliers[[i]][["decision_tree"]][[2]]$f1[k] <- metrics.tree.smote_outliers$F
    }
    
    
    results_outliers[[i]][["decision_tree"]][[3]]$auc[k] <- auc.tree.smote.dirichlet_outliers
    results_outliers[[i]][["decision_tree"]][[3]]$precision[k] <- metrics.tree.smote.dirichlet_outliers$precision
    results_outliers[[i]][["decision_tree"]][[3]]$recall[k] <- metrics.tree.smote.dirichlet_outliers$recall
    if (is.nan(metrics.tree.smote.dirichlet_outliers$F)| is.na(metrics.tree.smote.dirichlet_outliers$F)){
      results_outliers[[i]][["decision_tree"]][[3]]$f1[k] <- 0
    }else{
      results_outliers[[i]][["decision_tree"]][[3]]$f1[k] <- metrics.tree.smote.dirichlet_outliers$F
    }
    
    
    

  }
  
}


# Boxplot of AUC under Decision Tree --------------------------------------

# Specify the desired order of levels for trainset
levels <- c(
  "train_600_001", "train_600_0025", "train_600_005", "train_600_01",
  "train_1000_001", "train_1000_0025", "train_1000_005", "train_1000_01",
  "train_5000_001", "train_5000_0025", "train_5000_005", "train_5000_01"
)


# non outliers ------------------------------------------------------------



# Prepare data for plotting
plot_data <- data.frame(
  trainset = integer(),
  version = integer(),
  auc = numeric()
)

for (l in 1:9) {
  for (version in 1:3) {
    auc_values <- results[[l]]$decision_tree[[version]]$auc
    temp_df <- data.frame(
      trainset =  names(trainsets)[l],
      version = version,
      auc = auc_values
    )
    plot_data <- rbind(plot_data, temp_df)
  }
}

# Convert trainset to a factor with the specified levels
plot_data$trainset <- factor(plot_data$trainset, levels = levels)


# Create the plot
auc_dt <- ggplot(plot_data, aes(x = factor(version), y = auc, fill = factor(version))) +
  geom_boxplot() +
  facet_wrap(~ trainset, ncol = 3) +
  scale_fill_manual(
    values = c("#1b9e77", "#d95f02", "#7570b3"), 
    labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE")  
  ) +
  labs(
    title = "Boxplots of AUC values for Decision Tree Model by Version",
    x = "Version",
    y = "AUC",
    fill = "Model Version"  # Legend title
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )



# Boxplot of AUC under Logistic Regression --------------------------------

# Prepare data for plotting
plot_data <- data.frame(
  trainset = integer(),
  version = integer(),
  auc = numeric()
)

for (l in 1:9) {
  for (version in 1:3) {
    auc_values <- results[[l]]$logistic_regressor[[version]]$auc
    temp_df <- data.frame(
      trainset =  names(trainsets)[l],
      version = version,
      auc = auc_values
    )
    plot_data <- rbind(plot_data, temp_df)
  }
}

# Convert trainset to a factor with the specified levels
plot_data$trainset <- factor(plot_data$trainset, levels = levels)

# Create the plot
auc_logistic <- ggplot(plot_data, aes(x = factor(version), y = auc, fill = factor(version))) +
  geom_boxplot() +
  facet_wrap(~ trainset, ncol = 3) +
  scale_fill_manual(
    values = c("#1b9e77", "#d95f02", "#7570b3"),  
    labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE") 
  ) +
  labs(
    title = "Boxplots of AUC values for Logistic Regression Model by Version",
    x = "Version",
    y = "AUC",
    fill = "Model Version"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# -------------------------------------------------------------------------


plot_data <- data.frame(
  trainset = integer(),
  version = integer(),
  f1 = numeric()
)

for (l in 1:9) {
  for (version in 1:3) {
    f1_values <- results[[l]]$decision_tree[[version]]$f1
    temp_df <- data.frame(
      trainset =  names(trainsets)[l],
      version = version,
      f1 = f1_values
    )
    plot_data <- rbind(plot_data, temp_df)
  }
}

# Convert trainset to a factor with the specified levels
plot_data$trainset <- factor(plot_data$trainset, levels = levels)


# Create the plot
f1_dt <- ggplot(plot_data, aes(x = factor(version), y = f1, fill = factor(version))) +
  geom_boxplot() +
  facet_wrap(~ trainset, ncol = 3) +
  scale_fill_manual(
    values = c("#1b9e77", "#d95f02", "#7570b3"), 
    labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE")  
  ) +
  labs(
    title = "Boxplots of F1 values for Decision Tree Model by Version",
    x = "Version",
    y = "F1",
    fill = "Model Version"  
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )






plot_data <- data.frame(
  trainset = integer(),
  version = integer(),
  f1 = numeric()
)

for (l in 1:9) {
  for (version in 1:3) {
    f1_values <- results[[l]]$logistic_regressor[[version]]$f1
    temp_df <- data.frame(
      trainset =  names(trainsets)[l],
      version = version,
      f1 = f1_values
    )
    plot_data <- rbind(plot_data, temp_df)
  }
}

# Convert trainset to a factor with the specified levels
plot_data$trainset <- factor(plot_data$trainset, levels = levels)

# Create the plot
f1_logistic <- ggplot(plot_data, aes(x = factor(version), y = f1, fill = factor(version))) +
  geom_boxplot() +
  facet_wrap(~ trainset, ncol = 3) +
  scale_fill_manual(
    values = c("#1b9e77", "#d95f02", "#7570b3"),  
    labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE") 
  ) +
  labs(
    title = "Boxplots of F1 values for Logistic Regression Model by Version",
    x = "Version",
    y = "F1",
    fill = "Model Version" 
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )





# plotting metrics----------------------------------------------------------------

combined_metrics <- auc_dt + auc_logistic + f1_dt + f1_logistic + plot_layout(ncol = 2)
plot(combined_metrics)













# outliers ----------------------------------------------------------------



# Prepare data for plotting
plot_data <- data.frame(
  trainset = integer(),
  version = integer(),
  auc = numeric()
)

for (l in 1:9) {
  for (version in 1:3) {
    auc_values <- results_outliers[[l]]$decision_tree[[version]]$auc
    temp_df <- data.frame(
      trainset =  names(trainsets_outliers)[l],
      version = version,
      auc = auc_values
    )
    plot_data <- rbind(plot_data, temp_df)
  }
}

# Convert trainset to a factor with the specified levels
plot_data$trainset <- factor(plot_data$trainset, levels = levels)


# Create the plot
auc_dt_outliers <- ggplot(plot_data, aes(x = factor(version), y = auc, fill = factor(version))) +
  geom_boxplot() +
  facet_wrap(~ trainset, ncol = 3) +
  scale_fill_manual(
    values = c("#1b9e77", "#d95f02", "#7570b3"), 
    labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE")  
  ) +
  labs(
    title = "Boxplots of AUC values for Decision Tree Model by Version",
    x = "Version",
    y = "AUC",
    fill = "Model Version"  # Legend title
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )



# Boxplot of AUC under Logistic Regression --------------------------------

# Prepare data for plotting
plot_data <- data.frame(
  trainset = integer(),
  version = integer(),
  auc = numeric()
)

for (l in 1:9) {
  for (version in 1:3) {
    auc_values <- results_outliers[[l]]$logistic_regressor[[version]]$auc
    temp_df <- data.frame(
      trainset =  names(trainsets_outliers)[l],
      version = version,
      auc = auc_values
    )
    plot_data <- rbind(plot_data, temp_df)
  }
}

# Convert trainset to a factor with the specified levels
plot_data$trainset <- factor(plot_data$trainset, levels = levels)

# Create the plot
auc_logistic_outliers <- ggplot(plot_data, aes(x = factor(version), y = auc, fill = factor(version))) +
  geom_boxplot() +
  facet_wrap(~ trainset, ncol = 3) +
  scale_fill_manual(
    values = c("#1b9e77", "#d95f02", "#7570b3"),  
    labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE") 
  ) +
  labs(
    title = "Boxplots of AUC values for Logistic Regression Model by Version",
    x = "Version",
    y = "AUC",
    fill = "Model Version"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# -------------------------------------------------------------------------


plot_data <- data.frame(
  trainset = integer(),
  version = integer(),
  f1 = numeric()
)

for (l in 1:9) {
  for (version in 1:3) {
    f1_values <- results_outliers[[l]]$decision_tree[[version]]$f1
    temp_df <- data.frame(
      trainset =  names(trainsets_outliers)[l],
      version = version,
      f1 = f1_values
    )
    plot_data <- rbind(plot_data, temp_df)
  }
}

# Convert trainset to a factor with the specified levels
plot_data$trainset <- factor(plot_data$trainset, levels = levels)


# Create the plot
f1_dt_outliers <- ggplot(plot_data, aes(x = factor(version), y = f1, fill = factor(version))) +
  geom_boxplot() +
  facet_wrap(~ trainset, ncol = 3) +
  scale_fill_manual(
    values = c("#1b9e77", "#d95f02", "#7570b3"), 
    labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE")  
  ) +
  labs(
    title = "Boxplots of F1 values for Decision Tree Model by Version",
    x = "Version",
    y = "F1",
    fill = "Model Version"  
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )






plot_data <- data.frame(
  trainset = integer(),
  version = integer(),
  f1 = numeric()
)

for (l in 1:9) {
  for (version in 1:3) {
    f1_values <- results_outliers[[l]]$logistic_regressor[[version]]$f1
    temp_df <- data.frame(
      trainset =  names(trainsets_outliers)[l],
      version = version,
      f1 = f1_values
    )
    plot_data <- rbind(plot_data, temp_df)
  }
}

# Convert trainset to a factor with the specified levels
plot_data$trainset <- factor(plot_data$trainset, levels = levels)

# Create the plot
f1_logistic_outliers <- ggplot(plot_data, aes(x = factor(version), y = f1, fill = factor(version))) +
  geom_boxplot() +
  facet_wrap(~ trainset, ncol = 3) +
  scale_fill_manual(
    values = c("#1b9e77", "#d95f02", "#7570b3"),  
    labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE") 
  ) +
  labs(
    title = "Boxplots of F1 values for Logistic Regression Model by Version",
    x = "Version",
    y = "F1",
    fill = "Model Version" 
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )





# plotting metrics----------------------------------------------------------------

combined_metrics_outliers <- auc_dt_outliers + auc_logistic_outliers + f1_dt_outliers + f1_logistic_outliers + plot_layout(ncol = 2)
plot(combined_metrics_outliers)


print(f1_dt_outliers)

