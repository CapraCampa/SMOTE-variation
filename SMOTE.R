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
library(kknn)
library(class) # Ensure the class library is loaded for kNN


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
      # Why use 4?
      g[j, ] <- MCMCpack::rdirichlet(1, rep(1, K))
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

n_trainsets = 9

# Parameters of distribution of the two features
mu_0 <- c(0, 0)
cov_matrix_0 <- matrix(c(1, 0, 0, 1), nrow = 2)

mu_1 <- c(1, 1)
cov_matrix_1 <- matrix(c(1, -0.5, -0.5, 1), nrow = 2)



# Create a list of 12 lists to store results for each trainset
results <- vector("list", n_trainsets)  # One entry for each trainset
names(results) <- paste0("Trainset_", 1:n_trainsets)

n_simulations = 100

# Loop through each trainset and initialize model results
for (l in 1:n_trainsets) {
  results[[l]] <- list(
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
}



# 100 simulations !!!!!!!!!!!!!!!!!!!
for (k in 1:n_simulations){
  trainsets <- list()
  testsets <- list()
  
  # Simulate data for all train datasets (and corresponding test sets)
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
    
    
    # With the same method and same parameter pi, compute test set
    n_0 <- round(600 * (1 - prob))  
    n_1 <- round(600 * prob)       
    x_0 <- mvrnorm(n_0, mu_0, cov_matrix_0)
    x_0 <- data.frame(x_0)
    x_0$y <- rep(0, n_0)
    x_1 <- mvrnorm(n_1, mu_1, cov_matrix_1)
    x_1 <- data.frame(x_1)
    x_1$y <- rep(1, n_1)
    combined_data <- rbind(x_0, x_1)
    
    # Save test dataset in the list
    dataset_name <- paste0("test_", gsub("\\.", "", as.character(prob)))
    testsets[[dataset_name]] <- combined_data
  }
  
  ###############################################################################
  
  for (i in 1:n_trainsets){
    trainset <- trainsets[[i]]
    trainset_name <- names(trainsets)[i]
    
    p <- ggplot(trainset, aes(x = X1, y = X2, color = factor(y))) + 
      geom_point(aes(size = factor(y)), alpha = 0.8) + 
      scale_color_manual(values = c("grey", "blue")) + 
      scale_size_manual(values = c(1, 2)) +
      labs(title = "Train Dataset", x = "Feature 1", y = "Feature 2", color = "Class") +
      theme_minimal()
    #print(p)
    
    
    
    # (train sets will be different,
    # but test set is the same for all methods)
    trainset$y <- as.factor(trainset$y) # for some reason it needs this???
    # balance dataset with SMOTE
    IR <- nrow(trainset[trainset$y == 1, ]) / nrow(trainset)
    smote <- SMOTE(trainset[,-3], trainset[,3], K = 3, dup_size = 0)
    data.smote <- smote$data
    syn.data.smote <- smote$syn_data
    p1 <- ggplot(trainset, aes(x = X1, y = X2, color = factor(y))) + 
      geom_point(aes(size = factor(y)), alpha = 0.8, show.legend = c(color = TRUE, size = FALSE)) + 
      scale_color_manual(values = c("grey", "blue")) + 
      scale_size_manual(values = c(1, 2)) +
      geom_point(data = syn.data.smote, aes(x = X1, y = X2, color = factor(class)), shape = 17, alpha = 0.4) +
      labs(title = "SMOTE", x = "Feature 1", y = "Feature 2", color = "Class") +
      theme_minimal()
    
    #print(p1)
    
    data.smote$class <- factor(data.smote$class) #?????
    smote_IR <- nrow(data.smote[data.smote$class == 1, ]) / nrow(data.smote)
    
    # balance dataset with SMOTE variant
    
    smote.dirichlet <- SMOTE.DIRICHLET(trainset[,1:2], trainset$y, K = 5, dup_size = 0)
    data.smote.dirichlet <- smote.dirichlet$data
    syn.data.smote.dirichlet <- smote.dirichlet$syn_data
    x <- data.smote.dirichlet[,1:2]
    y <- data.smote.dirichlet$class
    
    
    p2 <- ggplot(x, aes(x = X1, y = X2, color = factor(y))) + 
      geom_point(aes(size = factor(y)), alpha = 0.8, show.legend = c(color = TRUE, size = FALSE)) + 
      scale_color_manual(values = c("grey", "blue")) + 
      scale_size_manual(values = c(1, 2)) +
      geom_point(data = syn.data.smote.dirichlet, aes(x = X1, y = X2, color = factor(class)), shape = 17, alpha = 0.4) +
      labs(title = "SMOTE.DIRICHLET", x = "Feature 1", y = "Feature 2", color = "Class") +
      theme_minimal()
    
    #print(p2)
    
    # Combine the two plots side-by-side
    combined_plot <- p1 + p2 + 
      plot_layout(ncol = 2) # Arrange plots in a single row
    
    #print(combined_plot)
    
    data.smote.dirichlet$class <- factor(data.smote.dirichlet$class) #?????
    dirichlet_IR <- nrow(data.smote.dirichlet[data.smote.dirichlet$class == 1, ]) / nrow(data.smote.dirichlet)
    
    
    ###############################################################################
    #apply models (tree and logistic regression) to all train sets
    
    # Classification trees
    
    # model trained on original unbalanced data
    tree <- rpart(y ~ ., data = trainset)
    # model trained on data balanced using smote
    tree.smote <- rpart(class ~ ., data = data.smote)
    # model trained on data balanced using smote dirichlet
    tree.smote.dirichlet <- rpart(class ~ ., data = data.smote.dirichlet)
    
    # Logistic regression
    
    fit <- glm(y ~ . , data = trainset, family = binomial(link = "logit"))
    fit.smote <- glm(class ~ . , data = data.smote, family = binomial(link = "logit"))
    fit.smote.dirichlet <- glm(class ~ . , data = data.smote.dirichlet, family = binomial(link = "logit"))
    
    test_index <- ceiling((i/3))
    testset <- testsets[[test_index]]
    
    # predict all models
    pred.tree <- predict(tree, newdata = testset,type = "prob")
    pred.tree.smote <- predict(tree.smote, newdata = testset,type = "prob")
    pred.tree.smote.dirichlet <- predict(tree.smote.dirichlet, newdata = testset)
    
    prob_class_1 <- predict(fit, newdata = testset, type= "response")      
    prob_class_0 <- 1 - prob_class_1   
    pred.fit <- cbind(prob_class_0, prob_class_1)
    
    prob_class_1 <- predict(fit.smote, newdata = testset, type="response")      
    prob_class_0 <- 1 - prob_class_1   
    pred.fit.smote <- cbind(prob_class_0, prob_class_1)
    
    prob_class_1 <- predict(fit.smote.dirichlet, newdata = testset, type="response")      
    prob_class_0 <- 1 - prob_class_1   
    pred.fit.smote.dirichlet <- cbind(prob_class_0, prob_class_1)
    
    
    ###############################################################################
    #compute metrics to compare our variant to the original technique
    
    # Logistic regression
    metrics.fit <- accuracy.meas(response = testset$y, predicted = pred.fit[,2])
    metrics.fit.smote <- accuracy.meas(response = testset$y, predicted = pred.fit.smote[,2])
    metrics.fit.smote.dirichlet <- accuracy.meas(response = testset$y, predicted = pred.fit.smote.dirichlet[,2])
    
    auc.fit <- roc.curve(testset$y, pred.fit[,2], plotit=FALSE, main = paste("ROC Curve - Dataset:", trainset_name, "\nLog regression"))$auc
    auc.fit.smote <- roc.curve(testset$y, pred.fit.smote[,2], plotit=FALSE,add.roc = TRUE, col = 2)$auc
    auc.fit.smote.dirichlet <- roc.curve(testset$y, pred.fit.smote.dirichlet[,2], add.roc = TRUE,plotit=FALSE, col = 3)$auc
    
    # Add a legend to identify the models
    #legend("bottomright", legend = c("Original", "SMOTE", "SMOTE with Dirichlet"), col = c(1, 2, 3), lwd = 2, title = "Model Type", cex = 0.8)
    
    
    results[[i]][["logistic_regressor"]][[1]]$auc[k] <- auc.fit
    if (is.nan(metrics.fit$precision)){
      results[[i]][["logistic_regressor"]][[1]]$precision[k] <- 0
    }else{
      results[[i]][["logistic_regressor"]][[1]]$precision[k] <- metrics.fit$precision
    }
    results[[i]][["logistic_regressor"]][[1]]$recall[k] <- metrics.fit$recall
    if (is.nan(metrics.fit$F)){
      results[[i]][["logistic_regressor"]][[1]]$f1[k] <- 0
    }else{
      results[[i]][["logistic_regressor"]][[1]]$f1[k] <- metrics.fit$F
    }
    
    
    results[[i]][["logistic_regressor"]][[2]]$auc[k] <- auc.fit.smote
    results[[i]][["logistic_regressor"]][[2]]$precision[k] <- metrics.fit.smote$precision
    results[[i]][["logistic_regressor"]][[2]]$recall[k] <- metrics.fit.smote$recall
    results[[i]][["logistic_regressor"]][[2]]$f1[k] <- metrics.fit.smote$F
    
    
    results[[i]][["logistic_regressor"]][[3]]$auc[k] <- auc.fit.smote.dirichlet
    results[[i]][["logistic_regressor"]][[3]]$precision[k] <- metrics.fit.smote.dirichlet$precision
    results[[i]][["logistic_regressor"]][[3]]$recall[k] <- metrics.fit.smote.dirichlet$recall
    results[[i]][["logistic_regressor"]][[3]]$f1[k] <- metrics.fit.smote.dirichlet$F
    
    
    
    # Classification trees
    metrics.tree <- accuracy.meas(response = testset$y, predicted = pred.tree[,2])
    metrics.tree.smote <- accuracy.meas(response = testset$y, predicted = pred.tree.smote[,2])
    metrics.tree.smote.dirichlet <- accuracy.meas(response = testset$y, predicted = pred.tree.smote.dirichlet[,2])
    
    auc.tree <- roc.curve(testset$y, pred.tree[,2], main = paste("ROC Curve - Dataset:", trainset_name, "\nTree"), plotit=FALSE)$auc
    auc.tree.smote <- roc.curve(testset$y, pred.tree.smote[,2],add.roc = TRUE, plotit=FALSE, col = 2)$auc
    auc.tree.smote.dirichlet <- roc.curve(testset$y, pred.tree.smote.dirichlet[,2], add.roc = TRUE, plotit=FALSE, col = 3)$auc
    # Add a legend to identify the models
    #legend("bottomright", legend = c("Original", "SMOTE", "SMOTE with Dirichlet"), col = c(1, 2, 3), lwd = 2, title = "Model Type", cex = 0.8)
    
    
    results[[i]][["decision_tree"]][[1]]$auc[k] <- auc.tree
    if (is.nan(metrics.tree$precision)){
      results[[i]][["decision_tree"]][[1]]$precision[k] <- 0
    }else{
      results[[i]][["decision_tree"]][[1]]$precision[k] <- metrics.tree$precision
    }
    results[[i]][["decision_tree"]][[1]]$recall[k] <- metrics.tree$recall
    if (is.nan(metrics.tree$precision)){
      results[[i]][["decision_tree"]][[1]]$f1[k] <- 0
    }else{
      results[[i]][["decision_tree"]][[1]]$f1[k] <- metrics.tree$F
    }
    
    results[[i]][["decision_tree"]][[2]]$auc[k] <- auc.tree.smote
    results[[i]][["decision_tree"]][[2]]$precision[k] <- metrics.tree.smote$precision
    results[[i]][["decision_tree"]][[2]]$recall[k] <- metrics.tree.smote$recall
    results[[i]][["decision_tree"]][[2]]$f1[k] <- metrics.tree.smote$F
    
    
    results[[i]][["decision_tree"]][[3]]$auc[k] <- auc.tree.smote.dirichlet
    results[[i]][["decision_tree"]][[3]]$precision[k] <- metrics.tree.smote.dirichlet$precision
    results[[i]][["decision_tree"]][[3]]$recall[k] <- metrics.tree.smote.dirichlet$recall
    results[[i]][["decision_tree"]][[3]]$f1[k] <- metrics.tree.smote.dirichlet$F
    
  }
  
}
# Boxplot of AUC under Decision Tree --------------------------------------

# Specify the desired order of levels for trainset
levels <- c(
  "train_600_001", "train_600_0025", "train_600_005", "train_600_01",
  "train_1000_001", "train_1000_0025", "train_1000_005", "train_1000_01",
  "train_5000_001", "train_5000_0025", "train_5000_005", "train_5000_01"
)

# Prepare data for plotting
plot_data <- data.frame(
  trainset = integer(),
  version = integer(),
  auc = numeric()
)

for (l in 1:12) {
  for (version in 1:3) {
    # Extract AUC values from the decision tree for the current version
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
ggplot(plot_data, aes(x = factor(version), y = auc, fill = factor(version))) +
  geom_boxplot() +
  facet_wrap(~ trainset, ncol = 4) +
  scale_fill_manual(
    values = c("#1b9e77", "#d95f02", "#7570b3"),  # Custom colors (optional)
    labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE")  # Custom labels
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

for (l in 1:12) {
  for (version in 1:3) {
    # Extract AUC values from the decision tree for the current version
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
ggplot(plot_data, aes(x = factor(version), y = auc, fill = factor(version))) +
  geom_boxplot() +
  facet_wrap(~ trainset, ncol = 4) +
  scale_fill_manual(
    values = c("#1b9e77", "#d95f02", "#7570b3"),  # Custom colors (optional)
    labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE")  # Custom labels
  ) +
  labs(
    title = "Boxplots of AUC values for Logistic Regression Model by Version",
    x = "Version",
    y = "AUC",
    fill = "Model Version"  # Legend title
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

for (l in 1:12) {
  for (version in 1:3) {
    # Extract AUC values from the decision tree for the current version
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
ggplot(plot_data, aes(x = factor(version), y = f1, fill = factor(version))) +
  geom_boxplot() +
  facet_wrap(~ trainset, ncol = 4) +
  scale_fill_manual(
    values = c("#1b9e77", "#d95f02", "#7570b3"),  # Custom colors (optional)
    labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE")  # Custom labels
  ) +
  labs(
    title = "Boxplots of F1 values for Decision Tree Model by Version",
    x = "Version",
    y = "F1",
    fill = "Model Version"  # Legend title
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

for (l in 1:12) {
  for (version in 1:3) {
    # Extract AUC values from the decision tree for the current version
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
ggplot(plot_data, aes(x = factor(version), y = f1, fill = factor(version))) +
  geom_boxplot() +
  facet_wrap(~ trainset, ncol = 4) +
  scale_fill_manual(
    values = c("#1b9e77", "#d95f02", "#7570b3"),  # Custom colors (optional)
    labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE")  # Custom labels
  ) +
  labs(
    title = "Boxplots of F1 values for Logistic Regression Model by Version",
    x = "Version",
    y = "F1",
    fill = "Model Version"  # Legend title
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# -------------------------------------------------------------------------

# pred <- cbind(prob_class_0, prob_class_1)
# 
# pred <- as.data.frame(pred)
# 
# pred$pred <- ifelse(pred$prob_class_0>pred$prob_class_1, 0, 1)
# 
# predicted <- pred$pred
# actual <- testset$y
# 
# confusion_matrix <- table(predicted, actual)
# 
# library(pheatmap)
# 
# pheatmap(confusion_matrix, display_numbers = TRUE, main = "Confusion Matrix")

# problem to address: if I have less than 6 observations in the rare class
# the SMOTE classic technique doesn't work

# problem to address: the smote.dirichlet doesn't seem to generate the right points