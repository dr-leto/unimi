import re
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from typing import List
from tqdm.notebook import tqdm

pd.set_option('display.max_columns', None)

# Plotting tools
from matplotlib import pyplot as plt
from matplotlib.markers import CARETDOWN
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sns.set(style='ticks')
sns.color_palette("muted", 10)

# Preprocessing Tools
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, OrdinalEncoder
from sklearn.decomposition import PCA

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def cat_bar_plot(data, cat_col, target_col, agg_func_name="sum", max_n_cat=None, create_dummy_cat=True):
    groups = data.groupby(cat_col)[target_col].agg([agg_func_name, 'count']) \
        .sort_values(by=agg_func_name, ascending=False).reset_index()

    if max_n_cat and max_n_cat < groups.shape[0]:
        if create_dummy_cat:
            extra_agg = groups[max_n_cat:][agg_func_name].sum()
            extra_count = groups[max_n_cat:]["count"].sum()
            groups = groups.iloc[:max_n_cat, :]
            groups.loc[max_n_cat] = ["other", extra_agg, extra_count]
        else:
            groups = groups.iloc[:max_n_cat, :]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    fig.tight_layout()
    plt.xlabel(f'{agg_func_name} {target_col}')
    plt.title(f'{agg_func_name} {target_col} across {cat_col}')
    sns.barplot(x=agg_func_name, y=cat_col, data=groups, edgecolor="black")

    ax.bar_label(labels=groups['count'], container=ax.containers[0])

    return fig


def num_bar_plot(data, target_col, num_col, group_num=5, hue_col=None):
    group_target_mean = None
    if hue_col:
        group_target_mean = data \
            .groupby([pd.qcut(data[num_col], group_num, duplicates='drop'), hue_col])[target_col].mean().reset_index()
    else:
        group_target_mean = data \
            .groupby(pd.qcut(data[num_col], group_num, duplicates='drop'))[target_col].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    fig.tight_layout()
    plt.title(f'Mean {target_col} across {num_col}')
    plt.xlabel(f'Mean {target_col}')
    sns.barplot(x=target_col, y=num_col, hue=hue_col, data=group_target_mean, edgecolor="black")
    
    return fig 

raw_data = pd.read_csv("pokemons.csv")

print("%d - TRAIN number of samples" % raw_data.shape[0])
print("%d - TRAIN number of columns \n" % raw_data.shape[1])
raw_data.head()

const_features = [f for f in raw_data.columns if raw_data[f].nunique() == 1]
if const_features:
    print("WARNING! Detected %d columns in training with constant values." % len(const_features))
print(const_features)

# Manual definition of categorical features
cat_features = ["Type_1", "Type_2", "Generation", "Color", "hasGender", 
                "Egg_Group_1", "Egg_Group_2", "Body_Style", "isLegendary"]
num_features = ["HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed", 
                "Pr_Male", "Height_m", "Weight_kg", "Catch_Rate"]
target_col = "hasMegaEvolution"

for c in cat_features + [target_col]:
    if raw_data[c].dtype == bool:
        raw_data[c] = raw_data[c].astype(int)

features = cat_features + num_features

print(f"Numerical features: {num_features}")
print(f"Categorical features: {cat_features}")
print(f"Target: {target_col}")

miss = (raw_data.isnull().sum() / raw_data.shape[0] * 100).to_frame()
miss.columns = ["miss_perc"]
miss[miss["miss_perc"] > 0].sort_values("miss_perc", ascending=False)

# Cast categorical features to string to avoid future inconsistencies
for c in cat_features:
    raw_data[c] = raw_data[c].astype(str)

vis_num_features = num_features

vis_cat_features = cat_features

# Fill in NaN values
raw_data["Pr_Male"] = raw_data["Pr_Male"].fillna(0.5)

pairplot = sns.pairplot(raw_data, hue=target_col, vars=vis_num_features)
pairplot.fig.set_size_inches(15,10)
pairplot.fig.suptitle("Pairplot of numerical features", y=1.05);

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(raw_data[num_features].corr(), cmap="YlGnBu", annot=True)
plt.xticks(rotation=45);

for feat in vis_num_features:
    num_bar_plot(raw_data, target_col, feat);

for feat in vis_cat_features:
    fig = cat_bar_plot(raw_data, feat, target_col, agg_func_name="mean")

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

feat_trans = ColumnTransformer([("num_scaler", StandardScaler(with_mean=False), num_features),
                                ("cat_ohe", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features)])
full_feat_trans = ColumnTransformer(feat_trans.transformers)

data = raw_data

# Make full transformation for all dataset - required for clustering
X_train_test = full_feat_trans.fit_transform(data[features])

# Make transformation for test based on fitted on train pipeline - required for prediction
X_train_raw, X_test_raw, y_train, y_test = train_test_split(data[features], data[target_col], random_state=28)
X_train = feat_trans.fit_transform(X_train_raw)
X_test = feat_trans.transform(X_test_raw)

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

metric = {}
metric["name"] = "F1-Score" 
metric["func"] = f1_score
metric["scorer"] = make_scorer(f1_score, greater_is_better=True,
                             needs_threshold=False)

lr = LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=3000)

param_grid = {'C': [10 ** k for k in range(-8, 6)]}
lr_grid = GridSearchCV(lr, param_grid, scoring=metric["scorer"], return_train_score=True)
lr_grid.fit(X_train, y_train);

param = 'param_C'
param_values = lr_grid.cv_results_[param].data
train_score = lr_grid.cv_results_['mean_train_score']
test_score = lr_grid.cv_results_['mean_test_score']

fig, ax = plt.subplots(figsize=(10, 5))
x_line = list(range(len(param_values)))
sns.lineplot(x=x_line, y=train_score, label = 'Train', ax=ax)
sns.lineplot(x=x_line, y=test_score, label = 'Test', ax=ax)

ax.xaxis.set_ticks(x_line)
ax.xaxis.set_ticklabels(param_values)

plt.xlabel('Inversed regularization strength')
plt.ylabel(metric['name'])
plt.title('Learning curve of Logistic Regression')
ax.legend();

best_lr_params = lr_grid.best_estimator_.get_params()
best_lr = LogisticRegression(**best_lr_params)
best_lr.fit(X_train, y_train)
y_test_probs = lr_grid.best_estimator_.predict_proba(X_test)[:, 1]
y_test_preds = lr_grid.best_estimator_.predict(X_test)

print("Test report")
print(classification_report(y_test, y_test_preds))
RocCurveDisplay.from_predictions(y_test, y_test_probs)
plt.title("Test ROC-curve");

top_n = 15

lr_feat_imp = pd.DataFrame({"weight": best_lr.coef_[0], "feature": feat_trans.get_feature_names_out()})
lr_feat_imp["abs_weight"] = abs(lr_feat_imp["weight"])
lr_feat_imp = lr_feat_imp.sort_values(by=["abs_weight"])

fig, ax = plt.subplots(figsize=(5, 5))
plt.title("Logistic Regression feature importance")
plt.xlabel("feature weight")
plt.barh(y="feature", width="weight", data=lr_feat_imp[-top_n:]);

from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier(class_weight='balanced', n_estimators=50)
    
param_distr = {'max_depth': list(range(1, 5)),
              'min_samples_split': list(range(2, 20)),
              'min_samples_leaf': list(range(1, 10))}
rf_grid = RandomizedSearchCV(rf, param_distr, n_iter=100, scoring=metric["scorer"], return_train_score=True, cv=3, n_jobs=-1)
rf_grid.fit(X_train, y_train);

best_rf_params = rf_grid.best_estimator_.get_params()
best_rf = RandomForestClassifier(**best_rf_params)
best_rf.fit(X_train, y_train)
y_test_probs = best_rf.predict_proba(X_test)[:, 1]
y_test_preds = best_rf.predict(X_test)

print("Test report")
print(classification_report(y_test, y_test_preds))
RocCurveDisplay.from_predictions(y_test, y_test_probs)
plt.title("Test ROC-curve");

forest_importances = pd.Series(rf_grid.best_estimator_.feature_importances_, index=feat_trans.get_feature_names_out())
index_order = np.argsort(forest_importances)[::-1]
std = np.std([tree.feature_importances_ for tree in rf_grid.best_estimator_.estimators_], axis=0)

top_k = 10
fig, ax = plt.subplots(figsize=(10, 5))
forest_importances[index_order][:top_k].plot.bar(yerr=std[index_order][:top_k], ax=ax)
plt.xticks(rotation=45)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity");

from sklearn.tree import plot_tree

decision_tree = rf_grid.best_estimator_.estimators_[0]
fig, ax = plt.subplots(figsize=(12, 9))
plot_tree(decision_tree, feature_names=feat_trans.get_feature_names_out(), fontsize=9);

pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X_train_test.toarray())

plt.figure(figsize=(10, 5))
plt.title('PCA feature reduction')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
sns.scatterplot(x=reduced_X[:, 0], y=reduced_X[:, 1], hue=data[target_col]);

n_comps = 20
list_comps = list(range(1, n_comps + 1))
pca = PCA(n_components=20)
pca.fit(X_train_test.toarray())

fig, ax = plt.subplots(figsize=(10, 5))
ax.xaxis.set_ticks(list_comps)
ax.xaxis.set_ticklabels(list_comps)
sns.lineplot(x=list_comps, y=pca.explained_variance_ratio_, ax=ax)

plt.xlabel('Number of components')
plt.ylabel('Explained variance ratio')
plt.title('Selecting number of components');

best_n = 10

pca = PCA(n_components=best_n)
X_train_red = pca.fit_transform(X_train_test.toarray())

print(f"%.2f%% - Explained variance by {best_n} components" % (pca.explained_variance_ratio_.sum() * 100))

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

n_clust = list(range(2, 11))
scores = []

for n in tqdm(n_clust):
    hier = AgglomerativeClustering(n_clusters=n)
    labels = hier.fit_predict(X_train_test.toarray())
    scores.append(silhouette_score(X_train_test, labels))
    
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=n_clust, y=scores, ax=ax)

plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Selecting number of clusters');

from yellowbrick.cluster import SilhouetteVisualizer

best_k = 3
best_hier = AgglomerativeClustering(n_clusters=best_k)

visualizer = SilhouetteVisualizer(best_hier, colors='yellowbrick')
visualizer.fit(X_train_test.toarray())
visualizer.show();


data["label"] = best_hier.fit_predict(X_train_test.toarray())

data.groupby("label").mean()

data["pokemon_type"] = data["label"].map({1: "beginner", 0: "intermediate", 2: "pro"})
data.groupby("pokemon_type").size().to_frame().rename(columns={0: "pokemon_count"}).plot(kind="barh");

pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X_train_test.toarray())

plt.figure(figsize=(10, 5))
plt.title('Pokemon type clusters visualized PCA k=2')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
sns.scatterplot(x=reduced_X[:, 0], y=reduced_X[:, 1], hue=data["pokemon_type"]);