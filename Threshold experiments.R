#install.packages("smotefamily")
#install.packages("caret")
#install.packages("ROSE")
#install.packages("MCMCpack")
#install.packages("rpart.plot")
#install.packages("tidyverse")
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
#library(pROC)
library(patchwork)
library(tidyverse)

# Functions --------------------------------------------------------------------

SMOTE.DIRICHLET <- function (X, target, K = 5, dup_size = 0){
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
      g[j, ] <- MCMCpack::rdirichlet(1, rep(1, K))
    }
    
    # multiplies the sum_dup weights for the k neighbors
    # in this way I obtain sum_dup new points
    syn_i = g %*% matrix(as.matrix(P_set[knear[i, ],]), ncol = ncD, byrow = FALSE)
    
    
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

# Function to compute TPR and FPR at given threshold
compute_rates <- function(threshold, predictions, true_labels) {
  predicted_labels <- ifelse(predictions >= threshold, 1, 0)
  TP <- sum(predicted_labels == 1 & true_labels == 1)
  FP <- sum(predicted_labels == 1 & true_labels == 0)
  FN <- sum(predicted_labels == 0 & true_labels == 1)
  TN <- sum(predicted_labels == 0 & true_labels == 0)
  
  # Calculate TPR and FPR
  TPR <- TP / (TP + FN)  # True Positive Rate
  FPR <- FP / (FP + TN)  # False Positive Rate
  return(c(FPR, TPR))
}

# Main experiment --------------------------------------------------------------

# Parameters
y <- c(0, 1) # 0 most frequent class, 1 less frequent class
train_size <- c(600, 1000, 5000) # size of train set
pi <- c(0.1, 0.05, 0.025) # proportion of rare class in train set
k_values <- c(3,5) # number of nearest neighbour in smote/smote dirichlet functions 

# Parameters of distribution of the two features
mu_0 <- c(0, 0)
cov_matrix_0 <- matrix(c(1, 0, 0, 1), nrow = 2)

mu_1 <- c(1, 1)
cov_matrix_1 <- matrix(c(1, -0.5, -0.5, 1), nrow = 2)

# Create a list of 9 lists to store results for each trainset
results <- vector("list", 9)  # One entry for each trainset
names(results) <- paste0("Trainset_", 1:9)

### NUMBER OF SIMULATIONS
n_simulations = 1

for (K in k_values){
  
  # Loop through each trainset and initialize model results ------------------
  for (l in 1:9) {
    results[[l]] <- list(
      logistic_regressor = vector("list", 3),         # 3 versions for KNN
      decision_tree = vector("list", 3)        # 3 versions for decision tree
    )
    
    # Loop through each model type and each version to initialize the sublists
    for (model_type in c("logistic_regressor", "decision_tree")) {
      for (version in 1:3) {
        results[[l]][[model_type]][[version]] <- list(
          auc = numeric(n_simulations),          # AUC values
          precision = numeric(n_simulations),    # Precision values
          recall = numeric(n_simulations),       # Recall values
          f1 = numeric(n_simulations),           # F1 values
          balanced_acc = numeric(n_simulations)  # Balanced Acc values
        )
      }
    }
  }
  
  # Simulate the experiment for n_simulations times ---------------------------
  for (k in 1:n_simulations){
    trainsets <- list()
    testsets <- list()
    
    # Simulate data for all train datasets (and corresponding test sets) ------
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
        
        # Combine
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
      
      # print("IR:")
      # print((1-prob)/(prob))
      
    }
    
    # Train and test models on all trainsets/testsets ------------
    
    for (i in 1:length(trainsets)){
      trainset <- trainsets[[i]]
      trainset_name <- names(trainsets)[i]
      
      p <- ggplot(trainset, aes(x = X1, y = X2, color = factor(y))) + 
        geom_point(aes(size = factor(y)), alpha = 1) + 
        scale_color_manual(values = c("grey", "blue")) + 
        scale_size_manual(values = c(1, 1.5)) +
        labs(title = "Data", x = "X1", y = "X2", color = "Class") +
        guides(size = "none") + # Remove legend for size
        theme_minimal()
      
      # pdf("data.pdf")
      # print(p)
      # dev.off()
      
      
      # (train sets will be different,
      # but test set is the same for all methods)
      trainset$y <- as.factor(trainset$y) # for some reason it needs this???
      # balance dataset with SMOTE
      smote <- SMOTE(trainset[,-3], trainset[,3], K = K, dup_size = 0)
      data.smote <- smote$data
      syn.data.smote <- smote$syn_data
      p1 <- ggplot(data.smote, aes(x = X1, y = X2, color = factor(class))) + 
        geom_point(aes(size = factor(class)), alpha = 0.4, show.legend = c(color = TRUE, size = FALSE)) + 
        geom_point(
          data = trainset[trainset$y == 1,], 
          aes(x = X1, y = X2, color = "trainset"),  # Add aesthetic mapping for color
          alpha = 1, 
          size = 1.5, 
          show.legend = TRUE  # Ensure this layer appears in the legend
        ) +
        scale_color_manual(
          values = c("grey", "darkgreen", "blue"),  # Define all colors here
          labels = c("0", "SMOTE", "trainset")  # Define all labels here
        ) +
        scale_size_manual(values = c(1, 1), guide = "none") +
        labs(
          title = "SMOTE", 
          x = "X1", 
          y = "X2", 
          color = "Class"
        ) +
        theme_minimal()
      
      # pdf("smote.pdf")
      # print(p1)
      # dev.off()
      
      
      data.smote$class <- factor(data.smote$class) 
      
      # balance dataset with SMOTE variant
      
      smote.dirichlet <- SMOTE.DIRICHLET(trainset[,1:2], trainset$y, K = K, dup_size = 0)
      data.smote.dirichlet <- smote.dirichlet$data
      syn.data.smote.dirichlet <- smote.dirichlet$syn_data
      x <- data.smote.dirichlet[,1:2]
      y <- data.smote.dirichlet$class
      
      
      p2 <- ggplot(data.smote.dirichlet, aes(x = X1, y = X2, color = factor(class))) + 
        geom_point(aes(size = factor(class)), alpha = 0.4, show.legend = c(color = TRUE, size = FALSE)) + 
        geom_point(
          data = trainset[trainset$y == 1,], 
          aes(x = X1, y = X2, color = "trainset"),  # Add aesthetic mapping for color
          alpha = 1, 
          size = 1.5, 
          show.legend = TRUE  # Ensure this layer appears in the legend
        ) +
        scale_color_manual(
          values = c("grey", "darkgreen", "blue"),  # Define all colors here
          labels = c("0", "SMOTE.DIRICHLET", "trainset")  # Define all labels here
        ) +
        scale_size_manual(values = c(1, 1), guide = "none") +
        labs(
          title = "SMOTE.DIRICHLET", 
          x = "X1", 
          y = "X2", 
          color = "Class"
        ) +
        theme_minimal()
      
      # pdf("dirichlet.pdf")
      # print(p2)
      # dev.off()
      
      
      # Combine the two plots side-by-side
      combined_plot <- p1 + p2 + plot_layout(ncol = 2) 
      
      # pdf("combined.pdf")
      # print(combined_plot)
      # dev.off()
      
      data.smote.dirichlet$class <- factor(data.smote.dirichlet$class)
      
      
      
      # Apply models (tree and logistic regression) to the 3 different train sets -----------
      
      # Classification trees
      
      # model trained on original unbalanced data
      tree <- rpart(y ~ ., data = trainset)
      best.cp <- tree$cptable[which.min(tree$cptable[,"xerror"]),]
      tree <- prune(tree, cp = best.cp[1])
      # model trained on data balanced using smote
      tree.smote <- rpart(class ~ ., data = data.smote)
      best.cp <- tree.smote$cptable[which.min(tree.smote$cptable[,"xerror"]),]
      tree.smote <- prune(tree.smote, cp = best.cp[1])
      # model trained on data balanced using smote dirichlet
      tree.smote.dirichlet <- rpart(class ~ ., data = data.smote.dirichlet)
      best.cp <- tree.smote.dirichlet$cptable[which.min(tree.smote.dirichlet$cptable[,"xerror"]),]
      tree.smote.dirichlet <- prune(tree.smote.dirichlet, cp = best.cp[1])
      
      # Logistic regression
      
      # model trained on original unbalanced data  
      fit <- glm(y ~ . , data = trainset, family = binomial(link = "logit"))
      # model trained on data balanced using smote
      fit.smote <- glm(class ~ . , data = data.smote, family = binomial(link = "logit"))
      # model trained on data balanced using smote dirichlet
      fit.smote.dirichlet <- glm(class ~ . , data = data.smote.dirichlet, family = binomial(link = "logit"))
      
      
      # Find and plot test set ---------------------------------------------
      test_index <- ceiling((i/3))
      testset <- testsets[[test_index]]
      testset$y <- factor(testset$y, levels = c(0, 1))
      
      p_test <- ggplot(testset, aes(x = X1, y = X2, color = factor(y))) + 
        geom_point(aes(size = factor(y)), alpha = 0.8, show.legend = c(color = TRUE, size = FALSE)) + 
        scale_color_manual(values = c("grey", "blue")) + 
        scale_size_manual(values = c(1, 2)) +
        labs(title = "Test data", x = "Feature 1", y = "Feature 2", color = "Class") +
        theme_minimal()
      
      #print(p_test)
      
      
      # Use optimal threshold to obtain predictions from probabilities ---------
      
      # Tree model
      pred.tree <- predict(tree, newdata = testset,type = "prob")
      pred.tree.smote <- predict(tree.smote, newdata = testset,type = "prob")
      pred.tree.smote.dirichlet <- predict(tree.smote.dirichlet, newdata = testset)
      
      threshold <- pi[(ceiling(i/3))]
      #threshold <- 0.5
      threshold.tree.smote <- 0.5
      threshold.tree.smote.dirichlet <- 0.5
      
      pred.tree.class <- ifelse(pred.tree[,2] > threshold, 1, 0)
      pred.tree.smote.class <- ifelse(pred.tree.smote[,2] > threshold.tree.smote, 1, 0)
      pred.tree.smote.dirichlet.class <- ifelse(pred.tree.smote.dirichlet[,2] > threshold.tree.smote.dirichlet, 1, 0)
      
      # Logistic regressor model
      prob_class_1 <- predict(fit, newdata = testset, type= "response")
      prob_class_0 <- 1 - prob_class_1
      pred.fit <- cbind(prob_class_0, prob_class_1)
      
      prob_class_1 <- predict(fit.smote, newdata = testset, type="response")
      prob_class_0 <- 1 - prob_class_1
      pred.fit.smote <- cbind(prob_class_0, prob_class_1)
      
      prob_class_1 <- predict(fit.smote.dirichlet, newdata = testset, type="response")
      prob_class_0 <- 1 - prob_class_1
      pred.fit.smote.dirichlet <- cbind(prob_class_0, prob_class_1)
      
      threshold <- pi[(ceiling(i/3))]
      #threshold <- 0.5
      threshold.fit.smote <- 0.5
      threshold.fit.smote.dirichlet <- 0.5
      
      pred.fit.class <- ifelse(pred.fit[,2] > threshold, 1, 0)
      pred.fit.smote.class <- ifelse(pred.fit.smote[,2] > threshold.fit.smote, 1, 0)
      pred.fit.smote.dirichlet.class <- ifelse(pred.fit.smote.dirichlet[,2] > threshold.fit.smote.dirichlet, 1, 0)
      
      pred.fit.class <- factor(pred.fit.class, levels = c(0,1))
      pred.fit.smote.class <- factor(pred.fit.smote.class, levels = c(0,1))
      pred.fit.smote.dirichlet.class <- factor(pred.fit.smote.dirichlet.class,levels = c(0,1))
      
      # Plot prediction boundaries ---------------------------------------------
      
      pred_df <- data.frame(
        X1 = testset$X1,
        X2 = testset$X2,
        Original = pred.tree.class,
        SMOTE = pred.tree.smote.class,
        Dirichlet = pred.tree.smote.dirichlet.class
      ) %>%
        pivot_longer(cols = -c(X1, X2), 
                     names_to = "Modello", 
                     values_to = "Predizione")
      
      # Plot 2: Previsioni Modello Originale
      p_original <- ggplot(pred_df %>% filter(Modello == "Original"), 
                           aes(x = X1, y = X2, color = factor(Predizione))) +
        geom_point(alpha = 0.6) +
        scale_color_manual(values = c("grey", "blue"), 
                           labels = c("Classe 0", "Classe 1")) +
        labs(title = "Previsioni Modello Originale", 
             color = "Classe Predetta") +
        theme_minimal()
      
      # Plot 3: Previsioni Modello SMOTE
      p_smote <- ggplot(pred_df %>% filter(Modello == "SMOTE"), 
                        aes(x = X1, y = X2, color = factor(Predizione))) +
        geom_point(alpha = 0.6) +
        scale_color_manual(values = c("grey", "blue"), 
                           labels = c("Classe 0", "Classe 1")) +
        labs(title = "Previsioni Modello SMOTE", 
             color = "Classe Predetta") +
        theme_minimal()
      
      # Plot 4: Previsioni Modello Dirichlet
      p_dirichlet <- ggplot(pred_df %>% filter(Modello == "Dirichlet"), 
                            aes(x = X1, y = X2, color = factor(Predizione))) +
        geom_point(alpha = 0.6) +
        scale_color_manual(values = c("grey", "blue"), 
                           labels = c("Classe 0", "Classe 1")) +
        labs(title = "Previsioni Modello Dirichlet", 
             color = "Classe Predetta") +
        theme_minimal()
      
      
      
      #print((p_test + p_original) / (p_smote + p_dirichlet))
      
      
      pred_df <- data.frame(
        X1 = testset$X1,
        X2 = testset$X2,
        Original = pred.fit.class,
        SMOTE = pred.fit.smote.class,
        Dirichlet = pred.fit.smote.dirichlet.class
      ) %>%
        pivot_longer(cols = -c(X1, X2), 
                     names_to = "Modello", 
                     values_to = "Predizione")
      
      # Plot 2: Previsioni Modello Originale
      p_original <- ggplot(pred_df %>% filter(Modello == "Original"), 
                           aes(x = X1, y = X2, color = factor(Predizione))) +
        geom_point(alpha = 0.6) +
        scale_color_manual(values = c("grey", "blue"), 
                           labels = c("Class 0", "Class 1")) +
        labs(title = "Model trained on unbalanced data", 
             color = "Predicted class") +
        theme_minimal()
      
      # Plot 3: Previsioni Modello SMOTE
      p_smote <- ggplot(pred_df %>% filter(Modello == "SMOTE"), 
                        aes(x = X1, y = X2, color = factor(Predizione))) +
        geom_point(alpha = 0.6) +
        scale_color_manual(values = c("grey", "blue"), 
                           labels = c("Class 0", "Class 1")) +
        labs(title = "Model trained on SMOTE data", 
             color = "Predicted class") +
        theme_minimal()
      
      # Plot 4: Previsioni Modello Dirichlet
      p_dirichlet <- ggplot(pred_df %>% filter(Modello == "Dirichlet"), 
                            aes(x = X1, y = X2, color = factor(Predizione))) +
        geom_point(alpha = 0.6) +
        scale_color_manual(values = c("grey", "blue"), 
                           labels = c("Class 0", "Class 1")) +
        labs(title = "Model trained on SMOTE DIRICHLET data", 
             color = "Predicted class") +
        theme_minimal()
      
      
      # pdf("compare_predictions_1.pdf")
      # print((p_test) / (p_smote))
      # dev.off()
      
      # pdf("compare_predictions_2.pdf")
      # print((p_original) / (p_dirichlet))
      # dev.off()
      
      
      # Compute and save metrics for all models trained on all datasets --------
      
      # Logistic regression
      metrics.fit <- accuracy.meas(response = testset$y, predicted = pred.fit[,2], threshold = threshold)
      metrics.fit.smote <- accuracy.meas(response = testset$y, predicted = pred.fit.smote[,2], threshold = threshold.fit.smote)
      metrics.fit.smote.dirichlet <- accuracy.meas(response = testset$y, predicted = pred.fit.smote.dirichlet[,2], threshold = threshold.fit.smote.dirichlet)
      
      auc.fit <- roc.curve(testset$y, pred.fit[,2], plotit=FALSE,col = "darkgreen", main = paste("ROC Curve - Dataset:", trainset_name, "\nLog regression"))$auc
      auc.fit.smote <- roc.curve(testset$y, pred.fit.smote[,2], plotit=FALSE,add.roc = TRUE, col = "orange")$auc
      auc.fit.smote.dirichlet <- roc.curve(testset$y, pred.fit.smote.dirichlet[,2], add.roc = TRUE,plotit=FALSE, col = "purple")$auc
      
      cm <- confusionMatrix(pred.fit.class, testset$y, positive = "1")
      acc.fit <- cm$byClass["Balanced Accuracy"]
      
      cm.smote <- confusionMatrix(pred.fit.smote.class, testset$y, positive = "1")
      acc.fit.smote <- cm.smote$byClass["Balanced Accuracy"]
      
      cm.smote.dirichlet <- confusionMatrix(pred.fit.smote.dirichlet.class, testset$y, positive = "1")
      acc.fit.smote.dirichlet <- cm.smote.dirichlet$byClass["Balanced Accuracy"]
      
      
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
      results[[i]][["logistic_regressor"]][[1]]$balanced_acc[k] <- acc.fit
      
      
      results[[i]][["logistic_regressor"]][[2]]$auc[k] <- auc.fit.smote
      results[[i]][["logistic_regressor"]][[2]]$precision[k] <- metrics.fit.smote$precision
      results[[i]][["logistic_regressor"]][[2]]$recall[k] <- metrics.fit.smote$recall
      if (is.nan(metrics.fit.smote$F)| is.na(metrics.fit.smote$F)){
        results[[i]][["logistic_regressor"]][[2]]$f1[k] <- 0
      }else{
        results[[i]][["logistic_regressor"]][[2]]$f1[k] <- metrics.fit.smote$F
      }
      results[[i]][["logistic_regressor"]][[2]]$balanced_acc[k] <- acc.fit.smote
      
      
      results[[i]][["logistic_regressor"]][[3]]$auc[k] <- auc.fit.smote.dirichlet
      results[[i]][["logistic_regressor"]][[3]]$precision[k] <- metrics.fit.smote.dirichlet$precision
      results[[i]][["logistic_regressor"]][[3]]$recall[k] <- metrics.fit.smote.dirichlet$recall
      if (is.nan(metrics.fit.smote.dirichlet$F)| is.na(metrics.fit.smote.dirichlet$F)){
        results[[i]][["logistic_regressor"]][[3]]$f1[k] <- 0
      }else{
        results[[i]][["logistic_regressor"]][[3]]$f1[k] <- metrics.fit.smote.dirichlet$F
      }
      results[[i]][["logistic_regressor"]][[3]]$balanced_acc[k] <- acc.fit.smote.dirichlet
      
      
      # Classification trees
      metrics.tree <- accuracy.meas(response = testset$y, predicted = pred.tree[,2], threshold = threshold)
      metrics.tree.smote <- accuracy.meas(response = testset$y, predicted = pred.tree.smote[,2], threshold = threshold.fit.smote)
      metrics.tree.smote.dirichlet <- accuracy.meas(response = testset$y, predicted = pred.tree.smote.dirichlet[,2], threshold = threshold.fit.smote.dirichlet)
      
      auc.tree <- roc.curve(testset$y, pred.tree[,2], main = paste("ROC Curve - Dataset:", trainset_name, "\nTree"), plotit=FALSE)$auc
      auc.tree.smote <- roc.curve(testset$y, pred.tree.smote[,2],add.roc = TRUE, plotit=FALSE, col = 2)$auc
      auc.tree.smote.dirichlet <- roc.curve(testset$y, pred.tree.smote.dirichlet[,2], add.roc = TRUE, plotit=FALSE, col = 3)$auc
      
      cm <- confusionMatrix(factor(pred.tree.class, levels=c(0,1)), testset$y, positive = "1")
      acc.tree <- cm$byClass["Balanced Accuracy"]
      
      cm <- confusionMatrix(factor(pred.tree.smote.class, levels = c(0,1)), testset$y, positive = "1")
      acc.tree.smote <- cm$byClass["Balanced Accuracy"]
      
      cm <- confusionMatrix(factor(pred.tree.smote.dirichlet.class, levels=c(0,1)), testset$y, positive = "1")
      acc.tree.smote.dirichlet <- cm$byClass["Balanced Accuracy"]
      
      
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
      results[[i]][["decision_tree"]][[1]]$balanced_acc[k] <- acc.tree
      
      
      results[[i]][["decision_tree"]][[2]]$auc[k] <- auc.tree.smote
      results[[i]][["decision_tree"]][[2]]$precision[k] <- metrics.tree.smote$precision
      results[[i]][["decision_tree"]][[2]]$recall[k] <- metrics.tree.smote$recall
      if (is.nan(metrics.tree.smote$F)| is.na(metrics.tree.smote$F)){
        results[[i]][["decision_tree"]][[2]]$f1[k] <- 0
      }else{
        results[[i]][["decision_tree"]][[2]]$f1[k] <- metrics.tree.smote$F
      }
      results[[i]][["decision_tree"]][[2]]$balanced_acc[k] <- acc.tree.smote
      
      
      results[[i]][["decision_tree"]][[3]]$auc[k] <- auc.tree.smote.dirichlet
      results[[i]][["decision_tree"]][[3]]$precision[k] <- metrics.tree.smote.dirichlet$precision
      results[[i]][["decision_tree"]][[3]]$recall[k] <- metrics.tree.smote.dirichlet$recall
      if (is.nan(metrics.tree.smote.dirichlet$F)| is.na(metrics.tree.smote.dirichlet$F)){
        results[[i]][["decision_tree"]][[3]]$f1[k] <- 0
      }else{
        results[[i]][["decision_tree"]][[3]]$f1[k] <- metrics.tree.smote.dirichlet$F
      }
      results[[i]][["decision_tree"]][[3]]$balanced_acc[k] <- acc.tree.smote.dirichlet 
      
       
      # Plot all roc curves from 0 to 1 with step 0.01 ---------------
      # Define thresholds to evaluate
      thresholds <- seq(0,1, by = 0.01)
      
      # Compute FPR and TPR for each threshold
      roc_points.fit <- t(sapply(thresholds, compute_rates, predictions = pred.fit[,2], true_labels = testset$y))
      fpr_values.fit <- roc_points.fit[, 1]
      tpr_values.fit <- roc_points.fit[, 2]
      
      roc_points.fit.smote <- t(sapply(thresholds, compute_rates, predictions = pred.fit.smote[,2], true_labels = testset$y))
      fpr_values.fit.smote <- roc_points.fit.smote[, 1]
      tpr_values.fit.smote <- roc_points.fit.smote[, 2]
      
      roc_points.fit.smote.dirichlet <- t(sapply(thresholds, compute_rates, predictions = pred.fit.smote.dirichlet[,2], true_labels = testset$y))
      fpr_values.fit.smote.dirichlet <- roc_points.fit.smote.dirichlet[, 1]
      tpr_values.fit.smote.dirichlet <- roc_points.fit.smote.dirichlet[, 2]
      
      # Plot the ROC curve
      plot(fpr_values.fit, tpr_values.fit, type = "b", col = "darkgreen", pch = 16, lwd = 2,
           xlab = "False Positive Rate", ylab = "True Positive Rate", 
           main = "ROC Curve with Explicit Thresholds",xlim = c(0,1), ylim=c(0,1))
      abline(a = 0, b = 1, lty = 2, col = "gray")  # Add diagonal reference line
      # The most interesting part of the first roc curve is obtained 
      # with threshold between 0 and 0.1 and step of 0.01
      
      
      lines(fpr_values.fit.smote, tpr_values.fit.smote, type = "b", col = "orange", pch = 16, lwd = 2)
      # We can obtain the same values as the previous roc curve with threshold
      # between 0.1 and 0.8 and step of 0.05
      
      lines(fpr_values.fit.smote.dirichlet, tpr_values.fit.smote.dirichlet, type = "b", col = "purple", pch = 16, lwd = 2)
      
      # Plot all roc curves from 0 to 1 with step 0.1 ---------------
      # Define thresholds to evaluate
      thresholds <- seq(0,1, by = 0.1)
      
      # Compute FPR and TPR for each threshold
      roc_points.fit <- t(sapply(thresholds, compute_rates, predictions = pred.fit[,2], true_labels = testset$y))
      fpr_values.fit <- roc_points.fit[, 1]
      tpr_values.fit <- roc_points.fit[, 2]
      
      roc_points.fit.smote <- t(sapply(thresholds, compute_rates, predictions = pred.fit.smote[,2], true_labels = testset$y))
      fpr_values.fit.smote <- roc_points.fit.smote[, 1]
      tpr_values.fit.smote <- roc_points.fit.smote[, 2]
      
      roc_points.fit.smote.dirichlet <- t(sapply(thresholds, compute_rates, predictions = pred.fit.smote.dirichlet[,2], true_labels = testset$y))
      fpr_values.fit.smote.dirichlet <- roc_points.fit.smote.dirichlet[, 1]
      tpr_values.fit.smote.dirichlet <- roc_points.fit.smote.dirichlet[, 2]
       
      # Plot the ROC curve
      plot(fpr_values.fit, tpr_values.fit, type = "b", col = "darkgreen", pch = 16, lwd = 2,
          xlab = "False Positive Rate", ylab = "True Positive Rate", 
           main = "ROC Curve with Explicit Thresholds",xlim = c(0,1), ylim=c(0,1))
      abline(a = 0, b = 1, lty = 2, col = "gray")  # Add diagonal reference line
      # The most interesting part of the first roc curve is obtained 
      # with threshold between 0 and 0.1 and step of 0.01
      
      
      lines(fpr_values.fit.smote, tpr_values.fit.smote, type = "b", col = "orange", pch = 16, lwd = 2)
      text(fpr_values.fit.smote, tpr_values.fit.smote, labels = round(thresholds, 2), pos = 4, cex = 0.8)
      # We can obtain the same values as the previous roc curve with threshold
      # between 0.1 and 0.8 and step of 0.05
      
      lines(fpr_values.fit.smote.dirichlet, tpr_values.fit.smote.dirichlet, type = "b", col = "purple", pch = 16, lwd = 2)
      
      # Plot all roc curves from 0 to 0.1 with step 0.01 -----------------------
      
      thresholds <- seq(0,0.1, by = 0.01)
      
      # Compute FPR and TPR for each threshold
      roc_points.fit <- t(sapply(thresholds, compute_rates, predictions = pred.fit[,2], true_labels = testset$y))
      fpr_values.fit <- roc_points.fit[, 1]
      tpr_values.fit <- roc_points.fit[, 2]
      
      roc_points.fit.smote <- t(sapply(thresholds, compute_rates, predictions = pred.fit.smote[,2], true_labels = testset$y))
      fpr_values.fit.smote <- roc_points.fit.smote[, 1]
      tpr_values.fit.smote <- roc_points.fit.smote[, 2]
      
      roc_points.fit.smote.dirichlet <- t(sapply(thresholds, compute_rates, predictions = pred.fit.smote.dirichlet[,2], true_labels = testset$y))
      fpr_values.fit.smote.dirichlet <- roc_points.fit.smote.dirichlet[, 1]
      tpr_values.fit.smote.dirichlet <- roc_points.fit.smote.dirichlet[, 2]
      
      # Plot the ROC curve
      plot(fpr_values.fit, tpr_values.fit, type = "b", col = "darkgreen", pch = 16, lwd = 2,
           xlab = "False Positive Rate", ylab = "True Positive Rate", 
           main = "ROC Curve with Explicit Thresholds",xlim = c(0,1), ylim=c(0,1))
      abline(a = 0, b = 1, lty = 2, col = "gray")  # Add diagonal reference line
      text(fpr_values.fit, tpr_values.fit, labels = round(thresholds, 2), pos = 4, cex = 0.8)
      # The most interesting part of the first roc curve is obtained 
      # with threshold between 0 and 0.1 and step of 0.01
      
      
      # plot(fpr_values.fit.smote, tpr_values.fit.smote, type = "b", col = "orange", pch = 16, lwd = 2,
      #      xlab = "False Positive Rate", ylab = "True Positive Rate", 
      #      main = "ROC Curve with Explicit Thresholds", xlim = c(0,1), ylim=c(0,1))
      lines(fpr_values.fit.smote, tpr_values.fit.smote, type = "b", col = "orange", pch = 16, lwd = 2)
      #text(fpr_values.fit.smote, tpr_values.fit.smote, labels = round(thresholds, 2), pos = 4, cex = 0.8)
      # We can obtain the same values as the previous roc curve with threshold
      # between 0.1 and 0.8 and step of 0.05
      
      lines(fpr_values.fit.smote.dirichlet, tpr_values.fit.smote.dirichlet, type = "b", col = "purple", pch = 16, lwd = 2)
      #text(fpr_values.fit.smote.dirichlet, tpr_values.fit.smote.dirichlet, labels = round(thresholds, 2), pos = 4, cex = 0.8)
      
      # Plot the first roc curve from 0 to 0.1 with step 0.01 and the other 2 from 0.1 to 1 with step 0.1 -------------------------
      
      # Define thresholds to evaluate
      thresholds_1 <- seq(0,0.1, by = 0.01)
      thresholds_2 <- seq(0.1,1, by = 0.1)
      
      # Compute FPR and TPR for each threshold
      roc_points.fit <- t(sapply(thresholds_1, compute_rates, predictions = pred.fit[,2], true_labels = testset$y))
      fpr_values.fit <- roc_points.fit[, 1]
      tpr_values.fit <- roc_points.fit[, 2]
      
      roc_points.fit.smote <- t(sapply(thresholds_2, compute_rates, predictions = pred.fit.smote[,2], true_labels = testset$y))
      fpr_values.fit.smote <- roc_points.fit.smote[, 1]
      tpr_values.fit.smote <- roc_points.fit.smote[, 2]
      
      roc_points.fit.smote.dirichlet <- t(sapply(thresholds_2, compute_rates, predictions = pred.fit.smote.dirichlet[,2], true_labels = testset$y))
      fpr_values.fit.smote.dirichlet <- roc_points.fit.smote.dirichlet[, 1]
      tpr_values.fit.smote.dirichlet <- roc_points.fit.smote.dirichlet[, 2]
      
      # Plot the ROC curve
      plot(fpr_values.fit, tpr_values.fit, type = "b", col = "darkgreen", pch = 16, lwd = 2,
           xlab = "False Positive Rate", ylab = "True Positive Rate", 
           main = "ROC Curve with Explicit Thresholds",xlim = c(0,1), ylim=c(0,1))
      abline(a = 0, b = 1, lty = 2, col = "gray")  # Add diagonal reference line
      # The most interesting part of the first roc curve is obtained 
      # with threshold between 0 and 0.1 and step of 0.01
      
      
      lines(fpr_values.fit.smote, tpr_values.fit.smote, type = "b", col = "orange", pch = 16, lwd = 2)
      # We can obtain the same values as the previous roc curve with threshold
      # between 0.1 and 0.8 and step of 0.05
      
      lines(fpr_values.fit.smote.dirichlet, tpr_values.fit.smote.dirichlet, type = "b", col = "purple", pch = 16, lwd = 2)
      
       
      
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
  
  
  
  # Boxplot of F1 under Decision Tree -------------------------
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
      title = "F1 of tree model",
      x = "",
      y = "F1",
      fill = "Model Version"  
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 10, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  
  
  # Boxplot of F1 under Logistic Regression -------------------------
  
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
      title = "F1 of logit model",
      x = "",
      y = "F1",
      fill = "Model Version" 
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 10, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  
  
  # Boxplot of Balanced Accuracy under Decision Tree -------------------------
  plot_data <- data.frame(
    trainset = integer(),
    version = integer(),
    acc = numeric()
  )
  
  for (l in 1:9) {
    for (version in 1:3) {
      acc_values <- results[[l]]$decision_tree[[version]]$balanced_acc
      temp_df <- data.frame(
        trainset =  names(trainsets)[l],
        version = version,
        acc = acc_values
      )
      plot_data <- rbind(plot_data, temp_df)
    }
  }
  
  # Convert trainset to a factor with the specified levels
  plot_data$trainset <- factor(plot_data$trainset, levels = levels)
  
  
  # Create the plot
  acc_dt <- ggplot(plot_data, aes(x = factor(version), y = acc, fill = factor(version))) +
    geom_boxplot() +
    facet_wrap(~ trainset, ncol = 3) +
    scale_fill_manual(
      values = c("#1b9e77", "#d95f02", "#7570b3"), 
      labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE")  
    ) +
    labs(
      title = "Balanced accuracy of tree model",
      x = "",
      y = "Balanced accuracy",
      fill = "Model Version"  
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 10, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  
  
  # Boxplot of Balanced Accuracy under Logistic Regression -------------------------
  
  
  plot_data <- data.frame(
    trainset = integer(),
    version = integer(),
    acc = numeric()
  )
  
  for (l in 1:9) {
    for (version in 1:3) {
      acc_values <- results[[l]]$logistic_regressor[[version]]$balanced_acc
      temp_df <- data.frame(
        trainset =  names(trainsets)[l],
        version = version,
        acc = acc_values
      )
      plot_data <- rbind(plot_data, temp_df)
    }
  }
  
  # Convert trainset to a factor with the specified levels
  plot_data$trainset <- factor(plot_data$trainset, levels = levels)
  
  # Create the plot
  acc_logistic <- ggplot(plot_data, aes(x = factor(version), y = acc, fill = factor(version))) +
    geom_boxplot() +
    facet_wrap(~ trainset, ncol = 3) +
    scale_fill_manual(
      values = c("#1b9e77", "#d95f02", "#7570b3"),  
      labels = c("Unbalanced data", "SMOTE", "Dirichlet SMOTE") 
    ) +
    labs(
      title = "Balanced accuracy of logit model",
      x = "",
      y = "Balanced accuracy",
      fill = "Model Version" 
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 10, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  
  
  
  # Plotting boxplots of metrics----------------------------------------------------------------
  
  #combined_metrics <- acc_dt + acc_logistic + f1_dt + f1_logistic + plot_layout(ncol = 2)
  #print(combined_metrics)
  # pdf("Tree_f1_optimized_threshold.pdf")
  # plot((f1_dt))
  # dev.off()
  # 
  # pdf("Tree_balanced_accuracy_optimized_threshold.pdf")
  # plot((acc_dt))
  # dev.off()
  # 
  # pdf("Logit_f1_optimized_threshold.pdf")
  # plot((f1_logistic))
  # dev.off()
  # 
  # pdf("Logit_balanced_accuracy_optimized_threshold.pdf")
  # plot((acc_logistic))
  # dev.off()
  
}

