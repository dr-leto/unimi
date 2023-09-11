# Plotting tools
library(ggplot2)
library(plotly)

# Preprocessing Tools
library(caret)

# Metrics
library(caret)

# Models
library(randomForest)

# Warnings
options(warn=-1)

cat_bar_plot <- function(data, cat_col, target_col, agg_func_name="sum", max_n_cat=NULL, create_dummy_cat=TRUE) {
  groups <- data %>%
    group_by({{ cat_col }}) %>%
    summarize({{ agg_func_name }} := sum({{ target_col }}, na.rm=TRUE), count = n()) %>%
    arrange(desc({{ agg_func_name }}))

  if (!is.null(max_n_cat) && max_n_cat < nrow(groups)) {
    if (create_dummy_cat) {
      extra_agg <- sum(groups[(max_n_cat + 1):nrow(groups), ][[agg_func_name]], na.rm=TRUE)
      extra_count <- sum(groups[(max_n_cat + 1):nrow(groups), ]$count)
      groups <- groups[1:max_n_cat, ]
      groups[nrow(groups) + 1, ] <- c("other", extra_agg, extra_count)
    } else {
      groups <- groups[1:max_n_cat, ]
    }
  }

  p <- ggplot(groups, aes(x = {{ agg_func_name }}, y = {{ cat_col }})) +
    geom_bar(stat = "identity", fill = "blue", color = "black") +
    labs(x = paste0(agg_func_name, " ", target_col), title = paste0(agg_func_name, " ", target_col, " across ", cat_col)) +
    theme_minimal() +
    theme(axis.text.y = element_text(hjust = 0.5))

  if (max_n_cat < nrow(groups)) {
    p <- p + scale_x_continuous(labels = scales::comma)
  }

  return(p)
}

num_bar_plot <- function(data, target_col, num_col, group_num = 5, hue_col = NULL) {
  if (!is.null(hue_col)) {
    group_target_mean <- data %>%
      group_by({{ num_col }}_cut = cut({{ num_col }}, breaks = quantile({{ num_col }}, probs = seq(0, 1, length.out = group_num), na.rm = TRUE), include.lowest = TRUE), {{ hue_col }}) %>%
      summarize(mean_{{ target_col }} = mean({{ target_col }}, na.rm = TRUE)) %>%
      ungroup()
  } else {
    group_target_mean <- data %>%
      group_by({{ num_col }}_cut = cut({{ num_col }}, breaks = quantile({{ num_col }}, probs = seq(0, 1, length.out = group_num), na.rm = TRUE), include.lowest = TRUE)) %>%
      summarize(mean_{{ target_col }} = mean({{ target_col }}, na.rm = TRUE)) %>%
      ungroup()
  }

  p <- ggplot(group_target_mean, aes(x = mean_{{ target_col }}, y = {{ num_col }}_cut, fill = {{ hue_col }})) +
    geom_bar(stat = "identity", position = "dodge", color = "black") +
    labs(x = paste0("Mean ", target_col), title = paste0("Mean ", target_col, " across ", num_col)) +
    theme_minimal() +
    theme(legend.position = "top")

  return(p)
}

# Read CSV file
raw_data <- read.csv("pokemons.csv")

cat("TRAIN number of samples:", nrow(raw_data), "\n")
cat("TRAIN number of columns:", ncol(raw_data), "\n\n")
head(raw_data)

# Identify constant features
const_features <- names(raw_data)[sapply(raw_data, function(x) length(unique(x)) == 1)]
if (length(const_features) > 0) {
  cat("WARNING! Detected", length(const_features), "columns in training with constant values.\n")
  print(const_features)
}

# Manual definition of categorical features
cat_features <- c("Type_1", "Type_2", "Generation", "Color", "hasGender", 
                  "Egg_Group_1", "Egg_Group_2", "Body_Style", "isLegendary")
num_features <- c("HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed", 
                  "Pr_Male", "Height_m", "Weight_kg", "Catch_Rate")
target_col <- "hasMegaEvolution"

# Convert boolean columns to integers
for (c in c(cat_features, target_col)) {
  if (is.logical(raw_data[[c]])) {
    raw_data[[c]] <- as.integer(raw_data[[c]])
  }
}

features <- c(cat_features, num_features)

cat("Numerical features:", paste(num_features, collapse = ", "), "\n")
cat("Categorical features:", paste(cat_features, collapse = ", "), "\n")
cat("Target:", target_col, "\n")

miss <- data.frame(miss_perc = (colSums(is.na(raw_data)) / nrow(raw_data)) * 100)
miss <- miss[miss$miss_perc > 0, ]
miss <- miss[order(miss$miss_perc, decreasing = TRUE), ]
colnames(miss) <- "miss_perc"
print(miss)

# Cast categorical features to string to avoid future inconsistencies
for (c in cat_features) {
  raw_data[[c]] <- as.character(raw_data[[c]])
}

vis_num_features <- num_features
vis_cat_features <- cat_features

# Fill in NaN values
raw_data$Pr_Male[is.na(raw_data$Pr_Male)] <- 0.5

# Pairplot of numerical features
library(ggplot2)
pairplot <- ggplot(raw_data, aes(x = ., color = as.factor(hasMegaEvolution))) +
  geom_density() +
  facet_wrap(~names(raw_data[, vis_num_features]), scales = "free", nrow = 3) +
  labs(title = "Pairplot of numerical features", y = "Density") +
  theme_minimal()
pairplot + theme(legend.position = "top") +
  theme(legend.title = element_blank()) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Correlation heatmap
library(reshape2)
library(corrplot)
cor_matrix <- cor(raw_data[, num_features])
cor_matrix_melted <- melt(cor_matrix)
cor_matrix_melted$value <- round(cor_matrix_melted$value, 2)
heatmap <- cor_matrix %>%
  corrplot::corrplot(type = "upper", order = "hclust", tl.col = "black", 
                     tl.srt = 45, method = "color", addCoef.col = "black") +
  labs(title = "Correlation Heatmap") +
  theme_minimal()
heatmap

# Bar plots for numerical features
library(ggplot2)
for (feat in vis_num_features) {
  p <- num_bar_plot(raw_data, target_col, feat)
  print(p)
}

# Bar plots for categorical features
for (feat in vis_cat_features) {
  p <- cat_bar_plot(raw_data, feat, target_col, agg_func_name = "mean")
  print(p)
}

# Data preprocessing
library(caret)
library(Matrix)
library(glmnet)

# Column transformation
feat_trans <- preProcess(raw_data[, features], method = c("center", "scale"))
full_feat_trans <- feat_trans$means

# Create a data matrix for modeling
data <- raw_data


# Data transformation for clustering
X_train_test <- predict(feat_trans, raw_data[, features])

# Data transformation for test based on the fitted pipeline
set.seed(28)
split_data <- splitIndices(nrow(raw_data), 0.7, method = "stratified")
X_train_raw <- raw_data[split_data$Resample1, features]
X_test_raw <- raw_data[-split_data$Resample1, features]
y_train <- raw_data[split_data$Resample1, target_col]
y_test <- raw_data[-split_data$Resample1, target_col]

X_train <- predict(feat_trans, X_train_raw)
X_test <- predict(feat_trans, X_test_raw)

# Model training and hyperparameter tuning
library(glmnet)

metric <- list(name = "F1-Score", func = function(y_true, y_pred) {
  f1 <- 2 * sum(y_true * y_pred) / (sum(y_true) + sum(y_pred))
  return(f1)
})

library(caret)
library(Matrix)

lr <- cv.glmnet(x = Matrix(X_train), y = y_train, alpha = 0.5, family = "binomial", type.measure = metric$name)

# Plotting hyperparameter tuning results
plot(lr)

# Best model selection
best_lambda <- lr$lambda.min
best_lr <- glmnet(x = Matrix(X_train), y = y_train, alpha = 0.5, family = "binomial", lambda = best_lambda)
y_test_probs <- predict(best_lr, s = best_lambda, newx = Matrix(X_test))
y_test_preds <- ifelse(y_test_probs > 0.5, 1, 0)

# Model evaluation
library(caret)

confusion_matrix <- confusionMatrix(data = y_test_preds, reference = y_test)
print("Confusion Matrix:")
print(confusion_matrix)

f1_score <- metric$func(y_test, y_test_preds)
print("F1-Score:", f1_score)

roc_curve <- roc(y_test, y_test_probs)
roc_auc <- auc(roc_curve)
print("ROC AUC:", roc_auc)

# Feature importance analysis
top_n <- 15

lr_feat_imp <- data.frame(weight = coef(best_lr), feature = colnames(X_train))
lr_feat_imp$abs_weight <- abs(lr_feat_imp$weight)
lr_feat_imp <- lr_feat_imp[order(lr_feat_imp$abs_weight, decreasing = TRUE), ]

# Plot top N important features
library(ggplot2)

ggplot(lr_feat_imp[1:top_n, ], aes(x = weight, y = reorder(factor(feature, levels = lr_feat_imp$feature)), fill = factor(weight > 0))) +
  geom_bar(stat = "identity", color = "black") +
  labs(title = "Logistic Regression feature importance", x = "Feature weight", y = "Feature") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("TRUE" = "blue", "FALSE" = "red"))


# Model training and hyperparameter tuning for Random Forest
library(randomForest)

rf <- randomForest(x = Matrix(X_train), y = y_train, ntree = 50, classwt = c(0.5, 0.5))

# Hyperparameter tuning with RandomizedSearchCV
library(caret)
library(randomForest)

param_distr <- list(
  max_depth = sample(1:4, 100, replace = TRUE),
  min_samples_split = sample(2:19, 100, replace = TRUE),
  min_samples_leaf = sample(1:9, 100, replace = TRUE)
)

rf_grid <- randomForest(x = Matrix(X_train), y = y_train, ntree = 50, mtry = 2, importance = TRUE)
best_rf_params <- list(
  max_depth = rf_grid$best$bestDepth,
  min_samples_split = rf_grid$best$bestSplit,
  min_samples_leaf = rf_grid$best$bestCut
)

best_rf <- randomForest(x = Matrix(X_train), y = y_train, ntree = 50, mtry = 2, importance = TRUE, 
                        maxdepth = best_rf_params$max_depth, 
                        minsplit = best_rf_params$min_samples_split, 
                        minbucket = best_rf_params$min_samples_leaf)

# Model evaluation for Random Forest
y_test_probs <- predict(best_rf, newdata = Matrix(X_test), type = "response")
y_test_preds <- ifelse(y_test_probs > 0.5, 1, 0)

library(caret)

confusion_matrix <- confusionMatrix(data = y_test_preds, reference = y_test)
print("Confusion Matrix:")
print(confusion_matrix)

f1_score <- metric$func(y_test, y_test_preds)
print("F1-Score:", f1_score)

roc_curve <- roc(y_test, y_test_probs)
roc_auc <- auc(roc_curve)
print("ROC AUC:", roc_auc)

# Feature importance analysis for Random Forest
library(ggplot2)

forest_importances <- as.data.frame(importance(best_rf))
names(forest_importances) <- colnames(X_train)
forest_importances <- t(forest_importances)
forest_importances <- data.frame(Feature = rownames(forest_importances), Importance = forest_importances[, 1])

forest_importances <- forest_importances[order(forest_importances$Importance, decreasing = TRUE), ]
top_k <- 10

fig <- ggplot(forest_importances[1:top_k, ], aes(x = Importance, y = reorder(Feature, Importance))) +
  geom_bar(stat = "identity", color = "black", fill = "blue") +
  labs(title = "Feature importances using MDI", x = "Mean decrease in impurity", y = "Feature") +
  theme_minimal() +
  theme(legend.position = "none") +
  coord_flip()

print(fig)

# Visualize one decision tree from the Random Forest
library(randomForest)

decision_tree <- best_rf$forest[[1]]
var_names <- colnames(X_train)
fig <- randomForest:::plot.randomForest(decision_tree, vars = var_names, main = "Decision Tree")

# PCA feature reduction
library(FactoMineR)

pca <- PCA(scale(X_train_test))
reduced_X <- as.data.frame(pca$ind$coord[, 1:2])

library(ggplot2)
ggplot(reduced_X, aes(x = Dim.1, y = Dim.2, color = as.factor(data[, target_col]))) +
  geom_point() +
  labs(title = "PCA feature reduction", x = "First principal component", y = "Second principal component") +
  theme_minimal()

# Selecting number of components
n_comps <- 20
list_comps <- 1:n_comps
pca <- PCA(X_train_test, scale.unit = TRUE, graph = FALSE)
explained_variance <- pca$eig[1:n_comps, 2]
cumulative_variance <- cumsum(explained_variance) / sum(explained_variance)

plot(list_comps, explained_variance, type = "b", pch = 19, xlab = "Number of components",
     ylab = "Explained variance ratio", main = "Selecting number of components")

# Choose the best number of components
best_n <- 10
pca <- PCA(X_train_test, scale.unit = TRUE, ncp = best_n)

print(paste("Explained variance by", best_n, "components:", round(sum(pca$eig[1:best_n, 2]) * 100, 2), "%"))

# Clustering using AgglomerativeClustering
library(stats)

n_clust <- 2:10
scores <- numeric(length(n_clust))

for (n in n_clust) {
  hier <- hclust(dist(X_train_test), method = "average")
  labels <- cutree(hier, k = n)
  scores[n - 1] <- silhouette(X_train_test, labels)
}

plot(n_clust, scores, type = "b", pch = 19, xlab = "Number of clusters",
     ylab = "Silhouette score", main = "Selecting number of clusters")

# Visualizing the best cluster using SilhouetteVisualizer
best_k <- 3
hier <- hclust(dist(X_train_test), method = "average")
labels <- cutree(hier, k = best_k)

library(cluster)
visualizer <- silhouette(X_train_test, labels)
plot(visualizer)

# Assigning cluster labels to the data
data$label <- as.factor(labels)

# Summary statistics by cluster
library(dplyr)

data %>%
  group_by(label) %>%
  summarize_at(vars(num_features), mean) %>%
  ungroup()

data$pokemon_type <- factor(data$label, labels = c("beginner", "intermediate", "pro"))
table(data$pokemon_type)

# PCA visualization of clusters
library(ggplot2)

pca <- PCA(X_train_test, scale.unit = TRUE, graph = FALSE)
reduced_X <- as.data.frame(pca$ind$coord[, 1:2])

ggplot(reduced_X, aes(x = Dim.1, y = Dim.2, color = data$pokemon_type)) +
  geom_point() +
  labs(title = "Pokemon type clusters visualized PCA k=2",
       x = "First principal component", y = "Second principal component")
