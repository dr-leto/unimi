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


class DistributionComparator:
    def __init__(self, significance_level=0.05):
        self.sign_lvl = significance_level

    def _create_base_df_report(self, feats, p_vals):
        results = []
        p_vals = [round(p, 5) for p in p_vals]
        for p in p_vals:
            if p < self.sign_lvl:
                results.append("REJECT")
            else:
                results.append("DO NOT REJECT")
        return pd.DataFrame({"feature": feats, "p_value": p_vals, "H0_test_result": results})

    def compare_num_feats(self, df_a, df_b, num_feats):
        p_vals = []
        for num_feat in tqdm(num_feats):
            p_vals.append(ks_2samp(df_a[num_feat], df_b[num_feat])[1])
        return self._create_base_df_report(num_feats, p_vals)

    def _create_contingency_table(self, a, b):
        a_cnt = a.value_counts().to_frame()
        a_cnt.columns = ["a"]
        b_cnt = b.value_counts().to_frame()
        b_cnt.columns = ["b"]

        ab_cnt = a_cnt.join(b_cnt, how="outer")
        ab_cnt = ab_cnt.fillna(0)
        ab_cnt["total"] = ab_cnt["a"] + ab_cnt["b"]
        return ab_cnt.transpose()

    def compare_cat_feats(self, df_a, df_b, cat_feats, df_a_name="TRAIN", df_b_name="TEST"):
        p_vals = []
        a_uniqs, b_uniqs, ab_inters = [], [], []
        for cat_feat in tqdm(cat_feats):
            cont_table = self._create_contingency_table(df_a[cat_feat], df_b[cat_feat]).values
            p_vals.append(chi2_contingency(cont_table)[1])
            a_uniq, b_uniq = df_a[cat_feat].unique(), df_b[cat_feat].unique()
            a_uniqs.append(len(a_uniq))
            b_uniqs.append(len(b_uniq))
            ab_inters.append(len(set(a_uniq) & set(b_uniq)))

        base_cat_report = self._create_base_df_report(cat_feats, p_vals)
        base_cat_report[f"{df_a_name} nunique categories"] = a_uniqs
        base_cat_report[f"{df_b_name} nunique categories"] = b_uniqs
        base_cat_report[f"{df_a_name} and {df_b_name} intersection"] = ab_inters
        return base_cat_report


class FrequencyImputer(TransformerMixin):
    def __init__(self, max_cat_num=10, dummy_value="other"):
        self.max_cat_num = max_cat_num
        self.dummy_value = dummy_value
        self.freq_values = []

    def fit(self, X, y=None):
        unique, counts = np.unique(X, return_counts=True)
        self.freq_values = unique[np.argsort(counts)[::-1][:self.max_cat_num]]
        return self

    def transform(self, X, y=None):
        X_trans = X.copy()
        for i in range(len(X_trans)):
            if X_trans[i] not in self.freq_values:
                X_trans[i] = self.dummy_value
        return X_trans