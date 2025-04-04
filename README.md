<div align="center">
    <h1>SMOTE variation</h1>
    <h3>Authors: Valeria De Stasio, Christian Faccio, Andrea Suklan, Agnese Valentini</h3>
    <h6>Final project of Statistical Methods course - UniTs</h6>
</div>

---

## Introduction

This project focuses on addressing the challenge of imbalanced datasets in binary classification tasks[cite: 513]. Standard classification algorithms often perform suboptimally when one class (the minority class) is significantly less frequent than the other (the majority class), leading to poor prediction accuracy for the rare, often more interesting, class[cite: 511, 514].

This project implements and evaluates a variant of the Synthetic Minority Over-sampling Technique (SMOTE), known as SMOTE Dirichlet, comparing its effectiveness against standard SMOTE and models trained on the original unbalanced data[cite: 415, 519, 535].

## Techniques Used

* **SMOTE (Synthetic Minority Over-sampling Technique):** A widely used data-level approach to address class imbalance. It generates synthetic minority class instances by interpolating between existing minority instances and their k-nearest minority class neighbors [cite: 518, 541-543].
* **SMOTE Dirichlet:** A variant of SMOTE where synthetic instances for a minority point are generated within the convex hull formed by its k-nearest neighbors (excluding the point itself)[cite: 545]. The combination weights are generated using a Dirichlet distribution[cite: 421, 546]. This project uses the Uniform Vector (UV) approach where the Dirichlet parameter α is a vector of ones[cite: 548, 551].
* **Baseline:** Models trained directly on the original, unbalanced dataset.

## Methodology

1.  **Data Generation:**
    * Simulated datasets were created with varying training set sizes (n\_train = 600, 1000, 5000) and different proportions (π = 0.10, 0.05, 0.025) of the minority class[cite: 416, 520]. The test set size was fixed at 600[cite: 416].
    * Data points were drawn from two bivariate normal distributions [cite: 417-419, 532-534]:
        * Class 0: N₂( (0, 0)ᵀ, I₂ ) [cite: 417, 533]
        * Class 1: N₂( (1, 1)ᵀ, [[1, -0.5], [-0.5, 1]] ) [cite: 418, 534]
    * For SMOTE and SMOTE Dirichlet, datasets were balanced by generating minority instances until the proportion reached 0.5[cite: 536].

2.  **Model Training:**
    * Two classification models were trained:
        * Decision Tree [cite: 420, 534]
        * Logistic Regression [cite: 420, 534]
    * Models were trained on three types of datasets: original unbalanced, SMOTE-balanced, and SMOTE Dirichlet-balanced[cite: 415, 535].

3.  **Performance Evaluation:**
    * Models were evaluated on a separate test set over 100 simulations[cite: 420, 534].
    * Metrics used:
        * **Balanced Accuracy:** The average of sensitivity (TPR) and specificity (TNR)[cite: 420].
        * **F1 Score:** The harmonic mean of precision and recall[cite: 420, 523].
    * Evaluation was performed using both a default classification threshold of 0.5 [cite: 424, 426] and an optimized threshold (threshold = π for the model trained on unbalanced data)[cite: 428, 430, 452, 455].

4.  **Parameters:**
    * For both SMOTE and SMOTE Dirichlet, the number of neighbors `k` was set to 5[cite: 549].
    * For SMOTE Dirichlet, the Dirichlet distribution parameter `α` was set to **1**ᵏ (a vector of K ones)[cite: 421, 551].

## Results Summary

* **Effectiveness of Balancing:** Both SMOTE and SMOTE Dirichlet generally led to improved performance (particularly in F1 score) compared to models trained on the original unbalanced data. This highlights the benefit of addressing class imbalance at the data level.
* **SMOTE vs. SMOTE Dirichlet:** The SMOTE Dirichlet variant performed quite similarly to the standard SMOTE across different dataset configurations and models[cite: 563]. There is some indication that SMOTE Dirichlet might result in less variance in AUC values, though this requires further confirmation[cite: 564].
* **Model Comparison:** Logistic Regression models generally achieved higher and more stable performance (especially in terms of Balanced Accuracy) compared to Decision Trees across the tested scenarios [cite: 559, Figure: Distribution of the balanced accuracy and F1 for the logit model, Figure: Distribution of the balanced accuracy and F1 for the classification tree].
* **AUC Anomaly:** AUC values for the logistic regression models showed unexpected uniformity across the different balancing methods, warranting further investigation.

## Conclusions and Future Work

The SMOTE Dirichlet variant presents a viable alternative for balancing datasets, showing comparable performance to standard SMOTE in these simulations[cite: 563].

Future work includes:
* Investigating the uniform AUC results observed for the logistic regression models[cite: 565].
* Experimenting with different parameter values for SMOTE Dirichlet (e.g., different `k` or `α` settings)[cite: 565].
* Evaluating the performance of k-Nearest Neighbors (k-NN) on the same datasets[cite: 565].
* Testing the effectiveness of SMOTE Dirichlet in scenarios with outliers in the minority class, as suggested in related literature[cite: 547, 566].

## References

* Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of artificial intelligence research*, 16, 321-357 [cite: 457, bibliography.bib].
* Matharaarachchi, S., Domaratzki, M., & Muthukumarana, S. (2024). Enhancing SMOTE for imbalanced data with abnormal minority instances. *Machine Learning with Applications*, 18, 100597 [cite: 421, 460, bibliography.bib].
* Menardi, G., & Torelli, N. (2014). Training and assessing classification rules with imbalanced data. *Data mining and knowledge discovery*, 28, 92-122 [cite: 461, bibliography.bib].
* Additional context derived from the research paper "Training and assessing classification rules with unbalanced data" by Giovanna Menardi and Nicola Torelli (Working Paper Series, N. 2, 2010) included in the project resources.