"""Training controls and bounded session history for the Streamlit app."""

import hashlib
import json

import pandas as pd
import streamlit as st

import analysis as A
import diagnostics as D
import exports as X
import training as T


@st.cache_data(show_spinner=False, max_entries=4, ttl="1h")
def quality_table(raw):
    return D.dataset_summary(raw), D.duplicate_count(raw)


def dataset_quality(raw):
    with st.expander("Dataset quality summary"):
        table, duplicates = quality_table(raw)
        st.dataframe(table, hide_index=True)
        st.caption(f"{duplicates:,} exact duplicate rows beyond the first occurrence. Rows are retained; review whether repeated observations should stay together in a grouped split.")


def training_controls(loaded, name, dk, task):
    raw = loaded.frame
    st.header("1 · Train a classifier" if task == "classification" else "1 · Train a regressor", divider="gray")
    st.caption(f"{name} · {len(raw):,} rows × {raw.shape[1]} columns")
    for note in loaded.notes:
        st.warning(note)
    with st.expander("Preview the first rows"):
        st.dataframe(raw.head(20))
    dataset_quality(raw)
    guess = D.target_suggestion(raw)
    target = st.selectbox(
        "Target column", list(raw.columns), index=list(raw.columns).index(guess) if guess else None,
        placeholder="Choose a target", key=f"target::{dk}",
        help="Classification predicts classes; Regression predicts a numeric measurement. A suggestion is not proof of the column's meaning.",
    )
    if target is None:
        st.session_state.pop("training_result", None)
        st.info("Choose a target column to see the available features.")
        return None
    for warning in D.target_warnings(raw, target, task):
        st.warning(warning)
    if task == "classification":
        with st.expander("Target class balance"):
            counts = D.class_counts(raw[target])
            st.dataframe(counts.assign(share=counts["share"] * 100), hide_index=True, column_config={"share": st.column_config.NumberColumn("share (%)", format="%.1f")})

    split_method = st.selectbox("Split method", T.SPLITS, key=f"split::{dk}::{task}")
    split_column = None
    if split_method != "Random":
        split_column = st.selectbox("Group column" if split_method == "Grouped" else "Time column", [c for c in raw if c != target], index=None, key=f"split_column::{dk}::{split_method}")
    st.caption({
        "Random": "Random held-out split; classification is stratified. Use independently sampled examples.",
        "Grouped": "All rows from a group stay together. The test proportion applies to groups, so the row proportion can differ. Validation also holds out complete groups.",
        "Time ordered": "Train on earlier rows and test on later rows. Equal timestamps stay together. Validation uses expanding time windows. Use a parseable date/time column; invalid times are excluded.",
    }[split_method])
    options = T.feature_choices(raw, target, split_column)
    features = st.multiselect("Training features", options.usable, default=list(options.usable), key=f"training_features::{dk}::{target}::{split_column}")
    st.caption(
        "Numeric features only (at least 90% finite numbers among non-missing values). "
        "The target, detected IDs, split column and constant columns are excluded. Text, categorical "
        "and date strings are not encoded. Invalid or missing feature values use training-only median "
        "imputation; scaling is also fitted inside each training/validation split. Exported values stay original."
    )
    with st.expander("Columns excluded from training features"):
        st.dataframe(pd.DataFrame(options.excluded.items(), columns=["column", "reason"]), hide_index=True)
    clues = D.leakage_warnings(raw, target, tuple(features))
    if not clues.empty:
        st.warning("Possible target leakage: review these selected features before training.")
        st.dataframe(clues, hide_index=True)
    st.caption("Leakage checks are clues, not proof. Confirm every feature would be available when making a real prediction.")
    left, middle, right = st.columns(3)
    with left:
        test_percent = st.slider("Test-set proportion (%)", 10, 50, 20, 5, key=f"test_percent::{dk}")
    with middle:
        seed = st.number_input("Random seed", 0, 2**32 - 1, 42, 1, key=f"seed::{dk}")
    models = T.CLASSIFIERS if task == "classification" else T.REGRESSORS
    with right:
        model = st.selectbox("Classifier" if task == "classification" else "Regressor", models, key=f"classifier::{dk}::{task}")
    with st.expander("Validation, balancing and model comparison", expanded=True):
        cv_enabled = st.checkbox("Cross-validation on training data", value=True, key=f"cv::{dk}::{task}")
        folds, repeats, tune = 0, 1, False
        if cv_enabled:
            folds = int(st.number_input("Validation folds", 2, 10, 3, key=f"folds::{dk}::{task}"))
            if split_method == "Random":
                repeats = int(st.number_input("Validation repeats", 1, 5, 1, key=f"repeats::{dk}::{task}"))
            tune = st.checkbox("Select hyperparameters using cross-validation", key=f"tune::{dk}::{task}")
            st.caption("Selection uses balanced accuracy for classification and MAE for regression. The final test set is never used to fit preprocessing, choose parameters or rank validation candidates.")
        balance = st.checkbox("Balance training classes by random oversampling", key=f"balance::{dk}") if task == "classification" else False
        if balance:
            st.caption("Minority classes are resampled inside each training fit, including validation and calibration folds. Test rows are never resampled. Probabilities may need recalibration under the real class distribution.")
        extra = st.multiselect("Also compare these models", [m for m in models if m != model], key=f"compare::{dk}::{task}::{model}")
        st.caption("Compared models use identical train/test rows. Prefer validation scores for model selection; repeatedly choosing a model from test results makes those results exploratory.")

    settings = dict(test_size=test_percent / 100, seed=int(seed), task=task, split_method=split_method,
                    split_column=split_column, cv_folds=folds, cv_repeats=repeats, tune=tune, balance=balance)
    selected_models = [model, *extra]
    signature = hashlib.sha256(json.dumps([T.fingerprint(raw), target, features, selected_models, settings], sort_keys=True).encode()).hexdigest()
    if st.session_state.get("training_signature") != signature:
        st.session_state["training_signature"] = signature
        st.session_state.pop("training_result", None)
        st.session_state.pop("training_batch", None)
    if st.button("Train and evaluate", type="primary", key="train_button"):
        st.session_state.pop("training_result", None)
        st.session_state.pop("training_batch", None)
        st.session_state[f"history_view::{dk}"] = None
        if split_method != "Random" and split_column is None:
            st.error("Choose the group or time column before training.")
            return None
        cache = st.session_state.setdefault("training_cache", {})
        try:
            if signature in cache:
                results = cache[signature]
                st.caption("Reused the fitted models for these exact data and settings.")
            else:
                progress = st.progress(0, text="Preparing training and held-out rows…")
                def update(i, total, label):
                    progress.progress(i / total, text=f"{label}: {i} of {total} models complete")
                with st.spinner("Fitting models and validating on training folds…"):
                    results = T.compare_models(raw, target, tuple(features), selected_models, progress=update, **settings)
                progress.empty()
                for result in results:
                    result.metadata["source_file"] = name
                cache[signature] = results
                while len(cache) > 3:
                    cache.pop(next(iter(cache)))
            st.session_state["training_batch"] = results
            history = st.session_state.setdefault("experiment_history", {})
            for result in results:
                run_key = hashlib.sha256((signature + result.classifier).encode()).hexdigest()[:16]
                result.metadata["run_id"] = run_key
                history[run_key] = result
            while len(history) > 10:
                history.pop(next(iter(history)))
        except A.DataError as exc:
            st.error(str(exc))
            return None

    history = st.session_state.get("experiment_history", {})
    saved_choice = None
    if history:
        with st.expander("Session experiment history"):
            keys = list(history)
            table = T.comparison_table(list(history.values()))
            table.insert(0, "run", keys)
            st.dataframe(table, hide_index=True)
            st.download_button("Download experiment history", table.to_csv(index=False).encode(), "experiment_history.csv", on_click="ignore")
            st.caption("The latest 10 fitted results stay in this session. Rows with the same split ID are comparable on identical data partitions; different seeds or splits are separate experiments.")
            compare_keys = st.multiselect("Compare saved runs", keys, key=f"history_compare::{dk}")
            if compare_keys:
                chosen = [history[k] for k in compare_keys if k in history]
                if len({r.metadata["split_id"] for r in chosen}) > 1:
                    st.warning("These runs use different test sets or targets. Their scores are not a controlled model comparison.")
                st.dataframe(T.comparison_table(chosen), hide_index=True)
            saved_choice = st.selectbox("View a saved run", keys, index=None, key=f"history_view::{dk}")
    batch = st.session_state.get("training_batch", [])
    if saved_choice:
        result = history[saved_choice]
        st.info(f"Viewing saved run {saved_choice} from {result.metadata.get('source_file', 'uploaded data')}. Its task, features and split are recorded below; controls above configure the next run.")
    elif batch:
        if len(batch) > 1:
            st.subheader("Model comparison on the same held-out test set")
            st.dataframe(T.comparison_table(batch), hide_index=True)
            selection = st.selectbox("Model to inspect", [r.classifier for r in batch], key=f"inspect_model::{signature}")
            result = next(r for r in batch if r.classifier == selection)
        else:
            result = batch[0]
    else:
        st.info("Choose settings, then select Train and evaluate to generate held-out predictions.")
        return None
    st.session_state["training_result"] = result
    run_id = result.metadata["run_id"]
    st.success(f"{len(result.train_rows):,} training rows · {len(result.test_rows):,} held-out test rows · {result.classifier} · random seed {result.seed} · {result.metadata['split_method'].lower()} split")
    st.caption(f"Task: {result.task}. Target: {result.metadata['target']}. Features: {', '.join(result.features)}.")
    if result.missing_targets:
        st.warning(f"{result.missing_targets:,} missing or unusable targets excluded before splitting.")
    if result.metadata["missing_split_values"]:
        st.warning(f"{result.metadata['missing_split_values']:,} rows with missing/invalid split values excluded.")
    for note in result.notes:
        st.warning(note)
    if not result.cv_results.empty:
        st.subheader("Cross-validation performance")
        meta = result.metadata
        st.caption(f"Training data only · {meta['cv_splits']} validation splits · {meta['cv_metric']}: {meta['cv_mean']:.4f} ± {meta['cv_std']:.4f} (standard deviation across folds, not a confidence interval).")
        st.dataframe(result.cv_results, hide_index=True)
    if st.checkbox("Show training performance", key=f"show_training::{run_id}"):
        st.subheader("Training performance")
        st.caption("Measured on the fitting rows. This is not a generalization estimate.")
        st.dataframe(pd.DataFrame([result.training_metrics]), hide_index=True)
    st.info("Held-out test performance. All metrics, map cells and row details below use only the held-out test rows. source_row_id points to the original input data row (1 = first row after the header).")
    st.download_button("Download held-out prediction CSV", result.csv_bytes(), file_name="heldout_predictions.csv", mime="text/csv", on_click="ignore", key="heldout_download")
    st.download_button("Download training metadata", result.metadata_bytes(), file_name="training_metadata.json", mime="application/json", on_click="ignore")
    with st.expander("Held-out predictions"):
        st.dataframe(result.frame)
    return result, f"heldout:{run_id}"


def model_explanations(result, key):
    with st.expander("Feature importance and row explanations"):
        st.caption("Permutation importance measures the drop in test performance after shuffling a feature (balanced accuracy for classification, negative MAE for regression). It is exploratory, can be negative, and does not establish causation; correlated features can mask one another.")
        cache = st.session_state.setdefault("importance_cache", {})
        if st.button("Calculate permutation importance", key=f"importance::{key}"):
            with st.spinner("Shuffling held-out features (3 repeats, up to 500 rows)…"):
                cache[key] = T.permutation_scores(result)
            while len(cache) > 10:
                cache.pop(next(iter(cache)))
        if key in cache:
            st.dataframe(cache[key], hide_index=True)
            st.bar_chart(cache[key], x="feature", y="importance", horizontal=True)
        if st.checkbox("Explain an individual test row", key=f"explain_row::{key}"):
            row = int(st.number_input("Row in held-out CSV", 1, len(result.frame), 1, key=f"explain_id::{key}"))
            st.dataframe(result.frame.loc[[row]])
            note, table = T.row_explanation(result, row)
            st.caption(note + " This explanation refers to the original fitted model, before any threshold exploration.")
            if not table.empty:
                st.dataframe(table, hide_index=True)
