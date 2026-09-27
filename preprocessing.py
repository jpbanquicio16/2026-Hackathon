"""Explicit feature types and preprocessing fitted inside each training fold."""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import analysis

MAX_CATEGORIES = 50


@dataclass(frozen=True)
class TrainingFeatures:
    usable: tuple[str, ...]
    excluded: dict[str, str]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


def feature_choices(raw, target, split_column=None):
    reserved = {target: "target", **{c: "identifier" for c in analysis.suggest_id_columns(raw, exclude=(target,))}}
    if split_column:
        reserved[split_column] = "split column"
    numeric, categorical, excluded = [], [], {}
    for column in raw:
        if column in reserved:
            excluded[column] = reserved[column]
            continue
        labels = analysis.clean_labels(raw[column]).dropna()
        if labels.nunique() < 2:
            excluded[column] = "no values or only one distinct value"
            continue
        parsed = analysis.parse_numeric(raw[column])
        if parsed.valid.sum() >= analysis.MIN_NUMERIC_SHARE * len(labels):
            numeric.append(column)
        elif labels.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() >= .9 or re.search(r"(?:^|[_\s])(date|timestamp)(?:$|[_\s])", column, re.I):
            excluded[column] = "date/time: choose a split column or engineer features explicitly"
        elif labels.nunique() > MAX_CATEGORIES or labels.nunique() == len(labels):
            excluded[column] = f"free text or high cardinality: categorical features need repeated values and at most {MAX_CATEGORIES} categories"
        else:
            categorical.append(column)
    usable = tuple(c for c in raw if c in numeric or c in categorical)
    return TrainingFeatures(usable, excluded, tuple(numeric), tuple(categorical))


def values(raw, features, categorical):
    return pd.DataFrame({
        c: analysis.clean_labels(raw[c]).astype(object).where(lambda s: s.notna(), np.nan)
        if c in categorical else analysis.parse_numeric(raw[c]).values
        for c in features
    })


def steps(features=(), categorical=()):
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]
    if not categorical:
        return numeric_steps  # preserve the simple numeric pipeline and its inspectable statistics
    numeric = [c for c in features if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    transformers.append(("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=MAX_CATEGORIES)),
    ]), list(categorical)))
    return [("preprocessor", ColumnTransformer(transformers, remainder="drop"))]


def feature_names(model, originals):
    if isinstance(model, Pipeline) and "preprocessor" in model.named_steps:
        return tuple(model["preprocessor"].get_feature_names_out())
    return tuple(originals)
