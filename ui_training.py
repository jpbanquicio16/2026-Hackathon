"""Training controls and bounded session history for the Streamlit app."""

import hashlib
import json
import re

import pandas as pd
import streamlit as st

import analysis
import diagnostics
import experiments
import exports
import training

STATUS_ORDER = {"Problem": 0, "Warning": 1, "Review": 2, "Not applicable": 3, "Info": 4, "OK": 5}


# Keyed by the dataset fingerprint; the leading underscore stops Streamlit re-hashing the frame.
@st.cache_data(show_spinner=False, max_entries=8, ttl="1h")
def quality_table(fp, _raw):
    return diagnostics.dataset_summary(_raw), diagnostics.duplicate_count(_raw)


@st.cache_data(show_spinner=False, max_entries=16, ttl="1h")
def leakage_review(fp, target, features, _raw):
    return diagnostics.leakage_findings(_raw, target, None if features is None else list(features))


def show_leakage(findings, caption=None):
    for finding in findings:
        heading = "Probable target leakage" if finding.risk == "high" else "Review the column name"
        st.markdown(f"**{md(finding.feature)}** · {heading}\n" + "\n".join(f"* {md(reason)}" for reason in finding.reasons))
    if caption:
        st.caption(caption)


def md(text) -> str:
    """Escape text from the CSV so Markdown never reformats it."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|$<>~])", r"\\\1", str(text))


def dataset_quality(raw, fp=None, target=None, task="classification", treat_as_classes=False):
    fp = fp or training.fingerprint(raw)
    table, duplicates = quality_table(fp, raw)
    findings = leakage_review(fp, target, None, raw) if target is not None else None
    overview = diagnostics.quality_overview(raw, target, task, treat_as_classes, summary=table, findings=findings)
    attention = int(overview["status"].isin(["Problem", "Warning"]).sum())
    label = "Dataset quality summary" + (f" · {attention} check{'s' if attention != 1 else ''} need attention" if attention else "")
    with st.expander(label):
        st.caption("Is this dataset safe and sensible to train on? Problems and warnings are listed first.")
        st.dataframe(overview.sort_values("status", key=lambda s: s.map(STATUS_ORDER), kind="stable"), hide_index=True)
        st.markdown("**Columns**")
        st.dataframe(table, hide_index=True)
        st.caption(f"{duplicates:,} exact duplicate rows beyond the first occurrence. Rows are retained; review whether repeated observations should stay together in a grouped split.")
        if target is not None and task == "classification":
            guidance = diagnostics.target_guidance(raw, target, task, treat_as_classes)
            if guidance.show_class_diagnostics:
                st.markdown("**Target class balance**")
                counts = diagnostics.class_counts(raw[target])
                st.dataframe(counts.assign(share=counts["share"] * 100), hide_index=True, column_config={"share": st.column_config.NumberColumn("share (%)", format="%.1f")})
        if findings:
            st.markdown("**Potential leakage across all columns**")
            show_leakage(findings, "Checked against every column, including ones not selected for training.")
    return findings


def training_controls(loaded, name, dk, task, fp=None):
    raw = loaded.frame
    fp = fp or training.fingerprint(raw)
    st.header("1 · Train a classifier" if task == "classification" else "1 · Train a regressor", divider="gray")
    st.caption(f"{name} · {len(raw):,} rows × {raw.shape[1]} columns")
    for note in loaded.notes:
        st.warning(note)
    with st.expander("Preview the first rows"):
        st.dataframe(raw.head(20))
    guess = diagnostics.target_suggestion(raw)
    target = st.selectbox(
        "Target column", list(raw.columns), index=list(raw.columns).index(guess) if guess else None,
        placeholder="Choose a target", key=f"target::{dk}",
        help="Classification predicts classes; Regression predicts a numeric measurement. A suggestion is not proof of the column's meaning.",
    )
    if target is None:
        dataset_quality(raw, fp)
        st.session_state.pop("training_result", None)
        st.info("Choose a target column to see the available features.")
        return None
    profile = diagnostics.target_profile(raw, target)
    treat_as_classes = False
    if task == "classification" and profile.kind in ("continuous", "ambiguous"):
        treat_as_classes = st.checkbox(
            f"Treat the values of “{md(target)}” as classes anyway", key=f"treat_classes::{dk}::{target}",
            help="Shows class counts and imbalance diagnostics for these values. Training still needs at least two rows per value.",
        )
    guidance = diagnostics.target_guidance(raw, target, task, treat_as_classes)
    st.caption(f"Target type: {profile.description} ({md(profile.reason)}).")
    for warning in guidance.warnings:
        st.warning(warning)
    for note in guidance.notes:
        st.caption(note)
    dataset_quality(raw, fp, target, task, treat_as_classes)

    split_method = st.selectbox("Split method", training.SPLITS, key=f"split::{dk}::{task}")
    split_column = None
    if split_method != "Random":
        split_column = st.selectbox("Group column" if split_method == "Grouped" else "Time column", [c for c in raw if c != target], index=None, key=f"split_column::{dk}::{split_method}")
    st.caption({
        "Random": "Random held-out split; classification is stratified. Use independently sampled examples.",
        "Grouped": "All rows from a group stay together. The test proportion applies to groups, so the row proportion can differ. For classification, the app looks for a group partition that puts every class in both sets and warns when that is impossible. Validation also holds out complete groups, balancing classes across folds.",
        "Time ordered": "Train on earlier rows and test on later rows. Equal timestamps stay together. Validation uses expanding time windows. Use a parseable date/time column; invalid times are excluded.",
    }[split_method])
    options = training.feature_choices(raw, target, split_column)
    features = st.multiselect("Training features", options.usable, default=list(options.usable), key=f"training_features::{dk}::{target}::{split_column}")
    categorical = st.multiselect("Treat these features as categories", features,
                                 default=[c for c in features if c in options.categorical],
                                 key=f"categorical::{dk}::{target}::{split_column}",
                                 help="Text features must be categorical. Select numeric codes here if their values represent categories rather than amounts.")
    st.caption(
        "Numeric features use median imputation and scaling. Categories use the most frequent training value for missing entries, "
        "then one-hot encoding; unseen test categories produce all-zero indicators. All preprocessing is fitted inside each training/validation fold. "
        "The target, detected IDs, split column, constants, date strings and free text/high-cardinality columns are excluded. "
        "Categorical columns are limited to 50 encoded levels. Map axes remain numeric; exported values stay original."
    )
    with st.expander("Columns excluded from training features"):
        st.dataframe(pd.DataFrame(options.excluded.items(), columns=["column", "reason"]), hide_index=True)
    selected_leakage = leakage_review(fp, target, tuple(features), raw) if features else []
    if selected_leakage:
        st.warning("Possible target leakage: review these selected features before training.")
        show_leakage(selected_leakage)
    st.caption("Leakage checks are clues, not proof. Confirm every feature would be available when making a real prediction.")
    left, middle, right = st.columns(3)
    with left:
        test_percent = st.slider("Test-set proportion (%)", 10, 50, 20, 5, key=f"test_percent::{dk}")
    with middle:
        seed = st.number_input("Random seed", 0, 2**32 - 1, 42, 1, key=f"seed::{dk}")
    models = training.CLASSIFIERS if task == "classification" else training.REGRESSORS
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
        imbalance = "None"
        if task == "classification":
            choices = [s for s in training.IMBALANCE_STRATEGIES if s != "Class weights" or training.supports_class_weights(model)]
            imbalance = st.selectbox("Class imbalance handling", choices, key=f"imbalance::{dk}::{model}",
                                     help="Class weights reweight the loss; oversampling repeats minority rows. Only one strategy is applied.")
            if not training.supports_class_weights(model):
                st.caption(f"{model} does not accept class weights, so only oversampling is offered.")
            if imbalance == "Oversampling":
                st.caption("Minority classes are resampled inside each training fit, including validation and calibration folds. Test rows are never resampled. Probabilities may need recalibration under the real class distribution.")
            elif imbalance == "Class weights":
                st.caption("class_weight='balanced' weights each class inversely to its training frequency inside every fit. Test rows are never reweighted.")
        compatible = [m for m in models if m != model and (imbalance != "Class weights" or training.supports_class_weights(m))]
        extra = st.multiselect("Also compare these models", compatible, key=f"compare::{dk}::{task}::{model}::{imbalance}")
        if imbalance == "Class weights" and len(compatible) < len(models) - 1:
            st.caption("Models without class-weight support are not offered for comparison under this strategy.")
        st.caption("Compared models use identical folds and train/test rows. Only cross-validation scores are shown until you lock one model and reveal its test result. CV scores used for tuning are selection estimates, not an independent final score.")

    settings = dict(test_size=test_percent / 100, seed=int(seed), task=task, split_method=split_method,
                    split_column=split_column, cv_folds=folds, cv_repeats=repeats, tune=tune, imbalance=imbalance,
                    dataset_fingerprint=fp, categorical_features=tuple(categorical), evaluate_test=False)
    selected_models = [model, *extra]
    signature = hashlib.sha256(json.dumps([fp, target, features, selected_models, settings], sort_keys=True).encode()).hexdigest()
    if st.session_state.get("training_signature") != signature:
        st.session_state["training_signature"] = signature
        st.session_state.pop("training_result", None)
        st.session_state.pop("training_batch", None)
        st.session_state.pop("locked_training_result", None)
    if st.button("Fit and compare on training data", type="primary", key="train_button"):
        st.session_state.pop("training_result", None)
        st.session_state.pop("training_batch", None)
        st.session_state.pop("locked_training_result", None)
        st.session_state[f"history_view::{dk}"] = None
        if split_method != "Random" and split_column is None:
            st.error("Choose the group or time column before training.")
            return None
        if len(selected_models) > 1 and not cv_enabled:
            st.error("Enable cross-validation to compare models without using test results.")
            return None
        cache = st.session_state.setdefault("training_cache", {})
        try:
            if signature in cache:
                results = cache[signature]
                st.caption("Reused the fitted models for these exact data and settings.")
            else:
                progress = st.progress(0, text="Preparing training and held-out rows…")

                def update(position, total, label):
                    current = min(int(position) + 1, total)
                    progress.progress(min(position / total, 1.0), text=f"Model {current} of {total}: {label}")

                with st.spinner("Fitting models and validating on training folds…"):
                    results = training.compare_models(raw, target, tuple(features), selected_models, progress=update, **settings)
                progress.empty()
                for result in results:
                    result.metadata["source_file"] = name
                cache[signature] = results
                while len(cache) > 3:
                    cache.pop(next(iter(cache)))
            st.session_state["training_batch"] = results
        except analysis.DataError as exc:
            st.error(str(exc))
            return None

    batch = st.session_state.get("training_batch", [])
    if batch:
        st.subheader("Model selection on training-fold validation")
        ranking = training.validation_ranking(batch)
        st.dataframe(ranking, hide_index=True)
        locked = st.session_state.get("locked_training_result")
        selection = st.selectbox("Model to lock for test evaluation", ranking["model"].tolist(),
                                 key=f"lock_model::{signature}", disabled=locked is not None)
        if not cv_enabled:
            st.warning("Cross-validation is disabled. This is a preselected model; there is no validation evidence for choosing it.")
        st.caption("The top model is selected by CV balanced accuracy (higher is better) or CV MAE (lower is better). Ties keep your configured model order. Locking evaluates only the selected model; other candidates keep their test results hidden.")
        if locked is None and st.button("Lock model and reveal held-out test performance", key="reveal_test", type="primary"):
            chosen = next(r for r in batch if r.classifier == selection)
            result = training.reveal_test(chosen, raw)
            prior = register_test_exposure(result)
            result.metadata["selection"] = {"method": "training-fold CV" if cv_enabled else "preselected without CV",
                                            "candidates": ranking.to_dict(orient="records"),
                                            "previous_test_reveals_for_dataset_target": prior,
                                            "exploratory_test_reuse": prior > 0}
            result.metadata["leakage_review"] = [{"feature": f.feature, "risk": f.risk, "reasons": list(f.reasons)} for f in selected_leakage if f.feature in result.features]
            result.metadata["run_id"] = hashlib.sha256((signature + result.classifier).encode()).hexdigest()[:16]
            st.session_state["locked_training_result"] = result
            save_history(result)
            st.rerun()

    history = st.session_state.get("experiment_history", {})
    saved_choice = None
    if history:
        with st.expander("Session experiment history"):
            table = training.history_table(history)
            st.dataframe(table, hide_index=True)
            full = {key: exports.json_safe(run.metadata) for key, run in history.items()}
            st.download_button("Download experiment history", table.to_csv(index=False).encode(), "experiment_history.csv", on_click="ignore")
            st.download_button("Download full run metadata (JSON)", exports.json_bytes(full), "experiment_history.json", mime="application/json", on_click="ignore")
            st.caption("The latest 10 fitted results stay in this session. Rows with the same split ID are comparable on identical data partitions; different seeds or splits are separate experiments.")
            keys = list(history)
            details = st.selectbox("Show full metadata for run", keys, index=None, key=f"history_details::{dk}")
            if details:
                st.json(full[details], expanded=False)
            compare_keys = st.multiselect("Compare saved runs", keys, key=f"history_compare::{dk}")
            if compare_keys:
                chosen = [history[k] for k in compare_keys if k in history]
                if training.comparable(chosen):
                    st.dataframe(training.comparison_table(chosen), hide_index=True)
                else:
                    st.warning("These runs use different test sets or targets. Their scores are not a controlled model comparison, so no side-by-side table is shown; compare their split details in the history above.")
            saved_choice = st.selectbox("View a saved run", keys, index=None, key=f"history_view::{dk}")
    if saved_choice:
        result = history[saved_choice]
        st.info(f"Viewing saved run {saved_choice} from {result.metadata.get('source_file', 'uploaded data')}. Its task, features and split are recorded below; controls above configure the next run.")
    elif batch and st.session_state.get("locked_training_result") is not None:
        result = st.session_state["locked_training_result"]
    else:
        st.info("Fit candidates using training data, then lock a model to reveal held-out predictions.")
        return None
    return show_result(result)


def save_history(result):
    history = st.session_state.setdefault("experiment_history", {})
    history[result.metadata["run_id"]] = result
    while len(history) > 10:
        history.pop(next(iter(history)))


def register_test_exposure(result):
    """Replaying a saved test result also informs later model choices in this session."""
    exposures = st.session_state.setdefault("test_exposures", {})
    metadata = result.metadata
    key = f"{metadata['dataset_sha256']}:{metadata['target']}:{result.task}"
    recorded = metadata.get("selection", {}).get("previous_test_reveals_for_dataset_target", 0)
    prior = max(exposures.get(key, 0), recorded)
    exposures[key] = prior + 1
    return prior


def restore_controls(data):
    digest = hashlib.sha256(data).hexdigest()
    try:
        manifest, raw, files = experiments.read_archive(data)
        st.header("Reproduce a saved experiment")
        st.caption("This archive contains source data, settings and saved outputs. Reproduction refits the model and verifies the split and predictions; no model code is loaded from the archive.")
        st.json(manifest["recipe"], expanded=True)
        allow = st.checkbox("Allow changed code or package versions (exploratory replay)", key=f"replay_allow::{digest}")
        if st.button("Reproduce saved run", type="primary", key="reproduce_run"):
            with st.spinner("Refitting the saved configuration and comparing predictions…"):
                result = experiments.reproduce(data, allow_environment_change=allow)
            register_test_exposure(result)
            st.session_state["restored_result"] = (digest, result)
            save_history(result)
        saved = st.session_state.get("restored_result")
        if saved and saved[0] == digest:
            result = saved[1]
            check = result.metadata["reproduction"]
            if check["exact_predictions"]:
                st.success("Reproduction verified: split row IDs and prediction CSV match the saved run exactly.")
            else:
                st.warning("Exploratory replay: split IDs match, but predictions differ from the saved run.")
            return show_result(result)
    except analysis.DataError as exc:
        st.error(str(exc))
    return None


def show_result(result):
    st.session_state["training_result"] = result
    run_id = result.metadata["run_id"]
    st.success(f"{len(result.train_rows):,} training rows · {len(result.test_rows):,} held-out test rows · {result.classifier} · random seed {result.seed} · {result.metadata['split_method'].lower()} split")
    st.caption(f"Task: {result.task}. Target: {result.metadata['target']}. Features: {', '.join(result.features)}. Class imbalance handling: {result.metadata.get('imbalance_strategy', 'None').lower()}.")
    if result.missing_targets:
        st.warning(f"{result.missing_targets:,} missing or unusable targets excluded before splitting.")
    if result.metadata["missing_split_values"]:
        st.warning(f"{result.metadata['missing_split_values']:,} rows with missing/invalid split values excluded.")
    for note in result.notes:
        st.warning(note)
    if not result.cv_results.empty:
        st.subheader("Cross-validation performance")
        meta = result.metadata
        st.caption(f"Training data only · {meta.get('cv_method', 'folds')} · {meta['cv_splits']} validation splits · {meta['cv_metric']}: {meta['cv_mean']:.4f} ± {meta['cv_std']:.4f} (standard deviation across folds, not a confidence interval).")
        st.dataframe(result.cv_results, hide_index=True)
    if st.checkbox("Show training performance", key=f"show_training::{run_id}"):
        st.subheader("Training performance")
        st.caption("Measured on the fitting rows. This is not a generalization estimate.")
        st.dataframe(pd.DataFrame([result.training_metrics]), hide_index=True)
    st.info("Held-out test performance. All metrics, map cells and row details below use only the held-out test rows. source_row_id points to the original input data row (1 = first row after the header).")
    if result.metadata.get("selection", {}).get("exploratory_test_reuse"):
        st.warning("Exploratory test reuse: test results for this dataset and target were already revealed in this session. Further model choices are influenced by those results; use a new independent dataset for confirmation. This record survives in downloaded metadata; this app cannot track reveals in other sessions.")
    st.download_button("Download held-out prediction CSV", result.csv_bytes(), file_name="heldout_predictions.csv", mime="text/csv", on_click="ignore", key="heldout_download")
    st.download_button("Download training metadata", result.metadata_bytes(), file_name="training_metadata.json", mime="application/json", on_click="ignore")
    with st.expander("Held-out predictions"):
        st.dataframe(result.frame)
    return result, f"heldout:{run_id}"


def model_explanations(result, key):
    with st.expander("Feature importance and row explanations"):
        note, built_in = training.model_importance(result)
        st.markdown("**Model-based importance**")
        st.caption(note)
        if not built_in.empty:
            st.dataframe(built_in, hide_index=True)
        st.markdown("**Permutation importance**")
        st.caption("Permutation importance measures the drop in test performance after shuffling a feature (balanced accuracy for classification, negative MAE for regression). It is exploratory, can be negative, and does not establish causation; correlated features can mask one another.")
        cache = st.session_state.setdefault("importance_cache", {})

        def permutation():
            with st.spinner("Shuffling held-out features (3 repeats, up to 500 rows)…"):
                cache[key] = training.permutation_scores(result)
            while len(cache) > 10:
                cache.pop(next(iter(cache)))

        if st.button("Calculate permutation importance", key=f"importance::{key}"):
            permutation()
        if st.checkbox("Explain an individual test row", key=f"explain_row::{key}"):
            row = int(st.number_input("Row in held-out CSV", 1, len(result.frame), 1, key=f"explain_id::{key}"))
            st.dataframe(result.frame.loc[[row]])
            note, table = training.row_explanation(result, row)
            if table.empty:
                st.info(note)
                if built_in.empty and key not in cache:
                    permutation()
            else:
                st.caption(note + " This explanation refers to the original fitted model, before any threshold exploration.")
                st.dataframe(table, hide_index=True)
        if key in cache:
            st.dataframe(cache[key], hide_index=True)
            st.bar_chart(cache[key], x="feature", y="importance", horizontal=True)
