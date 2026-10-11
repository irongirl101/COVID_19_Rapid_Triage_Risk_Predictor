import warnings
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import joblib

warnings.simplefilter("ignore", UserWarning)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


def resolve_model_path(name_or_path):
    """Resolve relative model paths against the project directory."""
    path = Path(name_or_path)
    return path if path.is_absolute() else MODEL_DIR / path


def _restore_legacy_lightgbm_state(model):
    """Support older LightGBM pickles loaded under newer lightgbm/scikit-learn."""
    if hasattr(model, "_Booster"):
        model.__dict__.setdefault("booster_", model.__dict__["_Booster"])
        model.__dict__.setdefault("fitted_", True)

    if hasattr(model, "_n_features") and not hasattr(model, "n_features_in_"):
        model.__dict__["n_features_in_"] = model._n_features

    if hasattr(model, "_classes") and not hasattr(model, "classes_"):
        model.__dict__["classes_"] = model._classes

    if hasattr(model, "_n_classes") and not hasattr(model, "n_classes_"):
        model.__dict__["n_classes_"] = model._n_classes

    return model


def _restore_legacy_svc_state(model):
    """Compat aliases needed by older SVC pickles under newer scikit-learn."""
    for legacy_name, modern_name in (("probA_", "_probA"), ("probB_", "_probB")):
        if legacy_name in model.__dict__ and modern_name not in model.__dict__:
            model.__dict__[modern_name] = model.__dict__[legacy_name]

    if hasattr(model, "_n_support") and not hasattr(model, "n_support_"):
        model.__dict__["n_support_"] = model._n_support

    return model


def load_legacy_selector(path, *, raise_on_error=True):
    """Load SelectFromModel models saved with older scikit-learn versions."""
    try:
        model = joblib.load(path)
    except Exception as exc:  # pragma: no cover - exercised at runtime
        if raise_on_error:
            raise
        return None

    # Compatibility with older scikit-learn SelectFromModel objects
    if not hasattr(model, "importance_getter"):
        model.importance_getter = "auto"
    estimator = getattr(model, "estimator", None)
    if (
        model.threshold is None
        and getattr(estimator, "penalty", None) == "l1"
        and getattr(estimator, "l1_ratio", "legacy") is None
    ):
        # Older sklearn pickles relied on the default 1e-5 threshold.
        model.threshold = 1e-5

    return model


# Columns that are not radiomic features
DEL_COL_NAME = [
    "ct_path",
    "study_id",
    "series_id",
    "diagnostics_Versions_PyRadiomics",
    "diagnostics_Versions_Numpy",
    "diagnostics_Versions_SimpleITK",
    "diagnostics_Versions_PyWavelet",
    "diagnostics_Versions_Python",
    "diagnostics_Configuration_Settings",
    "diagnostics_Configuration_EnabledImageTypes",
    "diagnostics_Image-original_Hash",
    "diagnostics_Image-original_Dimensionality",
    "diagnostics_Image-original_Spacing",
    "diagnostics_Image-original_Size",
    "diagnostics_Mask-original_Hash",
    "diagnostics_Mask-original_Spacing",
    "diagnostics_Mask-original_Size",
    "diagnostics_Mask-original_BoundingBox",
    "diagnostics_Mask-original_CenterOfMassIndex",
    "diagnostics_Mask-original_CenterOfMass",
]


def lesion_to_patient(data):
    """Convert lesion-level radiomics data into patient-level features."""

    data = data.copy()

    # Make sure patient_id is excluded from the feature columns
    del_columns = DEL_COL_NAME + ["patient_id"]

    Xcols = [col for col in data.columns if col not in del_columns]

    # Convert numeric feature columns
    data[Xcols] = data[Xcols].apply(pd.to_numeric, errors="coerce")

    feature_name_list = [
        "mean",
        "std",
        "medium",
        "skew",
        "75",
        "25",
    ]

    X_mean = data.groupby("patient_id")[Xcols].mean()
    X_std = data.groupby("patient_id")[Xcols].std()
    X_medium = data.groupby("patient_id")[Xcols].median()
    X_skew = data.groupby("patient_id")[Xcols].skew()
    X_75 = data.groupby("patient_id")[Xcols].quantile(q=0.75)
    X_25 = data.groupby("patient_id")[Xcols].quantile(q=0.25)

    feature_list = [
        X_mean,
        X_std,
        X_medium,
        X_skew,
        X_75,
        X_25,
    ]

    for i in range(len(feature_name_list)):
        name_dict = {}

        for name in Xcols:
            name_dict[name] = name + "_" + feature_name_list[i]

        feature_list[i] = feature_list[i].rename(columns=name_dict)

    X_final_no_count = pd.concat(feature_list, axis=1)

    lesion_count = data.groupby("patient_id")[
        "log-sigma-2-0-mm-3D_firstorder_Skewness"
    ].count()

    lesion_count.name = "lesion count"

    X_final = X_final_no_count.merge(
        lesion_count,
        on="patient_id"
    )

    return X_final.fillna(0)


def check_model_fitted(model, path):
    """
    Check that a serialized classifier actually contains a fitted model.

    We do NOT call fit() here because these are pretrained prediction models.
    """

    model_type = type(model).__name__

    # LightGBM can be serialized with older attribute names in legacy joblib files.
    if model_type.startswith("LGBM"):
        _restore_legacy_lightgbm_state(model)
        if not (getattr(model, "fitted_", False) or hasattr(model, "booster_") or hasattr(model, "_Booster")):
            raise RuntimeError(
                f"\nModel is not fitted:\n{path}\n"
                f"Loaded type: {type(model)}\n\n"
                "This .joblib file does not contain a fitted LightGBM model."
            )

    # Scikit-learn neural network
    elif model_type == "MLPClassifier":
        if not hasattr(model, "coefs_"):
            raise RuntimeError(
                f"\nModel is not fitted:\n{path}\n"
                f"Loaded type: {type(model)}\n\n"
                "This .joblib file contains an unfitted MLPClassifier."
            )

    elif model_type == "SVC":
        _restore_legacy_svc_state(model)
        if not (
            hasattr(model, "support_")
            or hasattr(model, "support_vectors_")
            or hasattr(model, "_probA")
            or hasattr(model, "_probB")
        ):
            raise RuntimeError(
                f"\nModel is not fitted:\n{path}\n"
                f"Loaded type: {type(model)}\n\n"
                "This .joblib file contains an unfitted SVC model."
            )

    # SelectFromModel wrappers
    elif model_type == "SelectFromModel":
        if not hasattr(model, "estimator_"):
            raise RuntimeError(
                f"\nModel is not fitted:\n{path}\n"
                f"Loaded type: {type(model)}\n\n"
                "This .joblib file contains an unfitted SelectFromModel result."
            )

    # General sklearn estimators
    elif not (
        hasattr(model, "n_features_in_")
        or hasattr(model, "coef_")
        or hasattr(model, "coefs_")
        or hasattr(model, "tree_")
        or hasattr(model, "support_")
        or hasattr(model, "estimator_")
        or hasattr(model, "classes_")
    ):
        print(
            f"Warning: could not automatically verify whether "
            f"{path} is fitted."
        )

    return model


def load_classifier(path, *, raise_on_error=True):
    """Load and validate a pretrained classifier."""
    try:
        model = joblib.load(path)
    except Exception as exc:  # pragma: no cover - exercised at runtime
        if raise_on_error:
            raise
        return None

    if type(model).__name__ == "SVC":
        _restore_legacy_svc_state(model)

    try:
        return check_model_fitted(model, path)
    except Exception as exc:  # pragma: no cover - exercised at runtime
        if raise_on_error:
            raise
        return None


def _format_prediction(value):
    """Render a prediction result for the CLI output."""
    if value is None:
        return "unavailable"
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        value = value[0]
    return value


def _transform_or_default(selector, X):
    if selector is None:
        return X
    return selector.transform(X)


def _predict_or_default(model, X):
    if model is None:
        return None
    return model.predict(X)


def _selected_feature_names(selector, feature_names):
    if selector is None:
        return list(feature_names)

    support = selector.get_support()
    if len(support) != len(feature_names):
        raise ValueError(
            "Feature selector output does not match the input feature names."
        )

    return [
        name for name, selected in zip(feature_names, support) if selected
    ]


def _print_shap_explanation(model, X, feature_names, outcome):
    """Print per-patient SHAP contributions to the model's class-1 probability."""
    if model is None:
        print(f"SHAP explanation unavailable for {outcome}: model is unavailable.")
        return

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            f"Cannot explain {outcome}: its model does not provide predict_proba."
        )

    import shap

    classes = list(model.classes_)
    if 1 not in classes:
        raise RuntimeError(
            f"Cannot explain {outcome}: the model has no positive class labeled 1."
        )
    positive_class_index = classes.index(1)

    X_values = np.asarray(X, dtype=float)
    if X_values.ndim == 1:
        X_values = X_values.reshape(1, -1)
    if X_values.shape[1] != len(feature_names):
        raise ValueError(
            f"Cannot explain {outcome}: {X_values.shape[1]} model features "
            f"but {len(feature_names)} feature names were provided."
        )

    # Inputs are standardized, so zero represents the training-feature mean.
    background = np.zeros((1, X_values.shape[1]))
    model_type = type(model).__name__
    is_tree_model = (
        hasattr(model, "tree_")
        or hasattr(model, "booster_")
        or hasattr(model, "_Booster")
        or model_type.startswith("LGBM")
    )

    if is_tree_model:
        explainer = shap.TreeExplainer(
            model,
            data=background,
            model_output="probability",
        )
        shap_values = explainer.shap_values(X_values)
    else:
        def positive_class_probability(rows):
            probabilities = np.asarray(model.predict_proba(rows))
            return probabilities[:, positive_class_index]

        explainer = shap.KernelExplainer(
            positive_class_probability,
            background,
        )
        shap_values = explainer.shap_values(X_values)

    expected_value = np.asarray(explainer.expected_value)
    if expected_value.ndim > 0 and expected_value.size > 1:
        base_value = float(expected_value.reshape(-1)[positive_class_index])
    else:
        base_value = float(expected_value.reshape(-1)[0])

    if isinstance(shap_values, list):
        shap_values = shap_values[positive_class_index]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, positive_class_index]
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    probabilities = np.asarray(model.predict_proba(X_values))[
        :, positive_class_index
    ]
    print(
        f"\nSHAP for {outcome} (class-1 probability; "
        "reference is zero-valued standardized features):"
    )
    for row_index, row_values in enumerate(shap_values):
        print(
            f"Patient {row_index + 1}: model output={probabilities[row_index]:.4f}, "
            f"baseline={base_value:.4f}"
        )
        top_indices = np.argsort(np.abs(row_values))[::-1][:10]
        for feature_index in top_indices:
            print(
                f"  {feature_names[feature_index]}: "
                f"{row_values[feature_index]:+.4f}"
            )


def radiomicsModel(radiomics_path, explain=False):
    """Run the radiomics-only prediction model."""

    print("Loading radiomics data:")
    print(radiomics_path)

    radiomics_data = pd.read_csv(radiomics_path)

    patient_input = lesion_to_patient(radiomics_data)

    # Standardization parameters
    mean_std = np.load(resolve_model_path("radiomics_ss.npy"))

    # Feature ordering
    feature_name = joblib.load(
        resolve_model_path("radiomics_feature_name.joblib")
    )

    reordered = patient_input[feature_name]

    test_X_std = (
        reordered - mean_std[0, :]
    ) / mean_std[1, :]

    # -----------------------------
    # ICU
    # -----------------------------

    ICU_clf_path = resolve_model_path("radiomics_models_ICU_clfmodel.joblib")
    ICU_fs_path = resolve_model_path("radiomics_models_ICU_fsmodel.joblib")

    ICU_clfmodel = load_classifier(ICU_clf_path, raise_on_error=False)
    ICU_fsmodel = load_legacy_selector(ICU_fs_path, raise_on_error=False)

    ICU_x = _transform_or_default(ICU_fsmodel, test_X_std)
    ICU = _predict_or_default(ICU_clfmodel, ICU_x)
    if explain:
        _print_shap_explanation(
            ICU_clfmodel,
            ICU_x,
            _selected_feature_names(ICU_fsmodel, test_X_std.columns),
            "ICU",
        )

    # -----------------------------
    # Mechanical ventilation
    # -----------------------------

    MV_clf_path = resolve_model_path("radiomics_models_MV_clfmodel.joblib")
    MV_clfmodel = load_classifier(MV_clf_path, raise_on_error=False)
    MV = _predict_or_default(MV_clfmodel, test_X_std)
    if explain:
        _print_shap_explanation(
            MV_clfmodel,
            test_X_std,
            list(test_X_std.columns),
            "mechanical ventilation",
        )

    # -----------------------------
    # Death
    # -----------------------------

    death_clf_path = resolve_model_path("radiomics_models_death_clfmodel.joblib")
    death_fs_path = resolve_model_path("radiomics_models_death_fsmodel.joblib")

    death_clfmodel = load_classifier(death_clf_path, raise_on_error=False)
    death_fsmodel = load_legacy_selector(death_fs_path, raise_on_error=False)

    death_x = _transform_or_default(death_fsmodel, test_X_std)
    death = _predict_or_default(death_clfmodel, death_x)
    if explain:
        _print_shap_explanation(
            death_clfmodel,
            death_x,
            _selected_feature_names(death_fsmodel, test_X_std.columns),
            "death",
        )

    if ICU is None:
        print("Warning: ICU model is unavailable in the current environment.")
    if MV is None:
        print("Warning: mechanical ventilation model is unavailable in the current environment.")
    if death is None:
        print("Warning: death model is unavailable in the current environment.")

    print(
        "\nRadiomics Model:"
        f"\nICU: {_format_prediction(ICU)}"
        f"\nMV: {_format_prediction(MV)}"
        f"\ndeath: {_format_prediction(death)}"
    )


def radiomicsClinLab(radiomics_path, lab_path, explain=False):
    """Run the radiomics + clinical + laboratory prediction model."""

    print("Loading radiomics data:")
    print(radiomics_path)

    print("Loading laboratory data:")
    print(lab_path)

    radiomics_data = pd.read_csv(radiomics_path)
    lab_data = pd.read_csv(lab_path)

    patient_input = lesion_to_patient(radiomics_data)

    # Standardization parameters
    mean_std = np.load(
        resolve_model_path("radioclinlab_ss.npy")
    )

    feature_name = joblib.load(
        resolve_model_path("radiomics_feature_name.joblib")
    )

    # -----------------------------
    # Lab-data imputation
    # -----------------------------

    lab_median = pd.read_csv(
        BASE_DIR / "example_median.csv"
    )

    lab_median = lab_median.drop(
        ["patient_id"],
        axis=1,
    )

    lab_data = lab_data.set_index(
        "patient_id"
    )

    columns = lab_median.columns

    for item in columns:
        patient_value = lab_data[item].iloc[0]
        if pd.isna(patient_value):
            lab_data.loc[lab_data.index[0], item] = lab_median[item].iloc[0]
            print("Impute:", item)

    # -----------------------------
    # Combine radiomics + lab
    # -----------------------------

    reordered = patient_input[feature_name]

    patient_input_radioclinlab = pd.concat(
        [
            reordered,
            lab_data,
        ],
        axis=1,
        ignore_index=True,
    )

    test_X_std = (
        patient_input_radioclinlab
        - mean_std[0, :]
    ) / mean_std[1, :]
    radioclinlab_feature_names = list(reordered.columns) + list(lab_data.columns)
    test_X_std.columns = radioclinlab_feature_names

    # -----------------------------
    # ICU
    # -----------------------------

    ICU_clf_path = resolve_model_path("radioclinlab_models_ICU_clfmodel.joblib")
    ICU_fs_path = resolve_model_path("radioclinlab_models_ICU_fsmodel.joblib")

    ICU_clfmodel = load_classifier(ICU_clf_path, raise_on_error=False)
    ICU_fsmodel = load_legacy_selector(ICU_fs_path, raise_on_error=False)

    ICU_x = _transform_or_default(ICU_fsmodel, test_X_std)
    ICU = _predict_or_default(ICU_clfmodel, ICU_x)
    if explain:
        _print_shap_explanation(
            ICU_clfmodel,
            ICU_x,
            _selected_feature_names(ICU_fsmodel, test_X_std.columns),
            "ICU",
        )

    # -----------------------------
    # Mechanical ventilation
    # -----------------------------

    MV_clf_path = resolve_model_path("radioclinlab_models_MV_clfmodel.joblib")
    MV_fs_path = resolve_model_path("radioclinlab_models_MV_fsmodel.joblib")

    MV_clfmodel = load_classifier(MV_clf_path, raise_on_error=False)
    MV_fsmodel = load_legacy_selector(MV_fs_path, raise_on_error=False)

    MV_x = _transform_or_default(MV_fsmodel, test_X_std)
    MV = _predict_or_default(MV_clfmodel, MV_x)
    if explain:
        _print_shap_explanation(
            MV_clfmodel,
            MV_x,
            _selected_feature_names(MV_fsmodel, test_X_std.columns),
            "mechanical ventilation",
        )

    # -----------------------------
    # Death
    # -----------------------------

    death_clf_path = resolve_model_path("radioclinlab_models_death_clfmodel.joblib")
    death_fs_path = resolve_model_path("radioclinlab_models_death_fsmodel.joblib")

    death_clfmodel = load_classifier(death_clf_path, raise_on_error=False)
    death_fsmodel = load_legacy_selector(death_fs_path, raise_on_error=False)

    death_x = _transform_or_default(death_fsmodel, test_X_std)
    death = _predict_or_default(death_clfmodel, death_x)
    if explain:
        _print_shap_explanation(
            death_clfmodel,
            death_x,
            _selected_feature_names(death_fsmodel, test_X_std.columns),
            "death",
        )

    if ICU is None:
        print("Warning: ICU model is unavailable in the current environment.")
    if MV is None:
        print("Warning: mechanical ventilation model is unavailable in the current environment.")
    if death is None:
        print("Warning: death model is unavailable in the current environment.")

    print(
        "\nRadioClinLab Model:"
        f"\nICU: {_format_prediction(ICU)}"
        f"\nMV: {_format_prediction(MV)}"
        f"\ndeath: {_format_prediction(death)}"
    )


def parse_opts():
    parser = argparse.ArgumentParser(
        description="COVID-19 prediction pipeline"
    )

    parser.add_argument(
        "--radiomics_data",
        required=True,
        type=str,
        help="Path to radiomics CSV file",
    )

    parser.add_argument(
        "--lab_data",
        default=None,
        type=str,
        help="Path to laboratory CSV file",
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show SHAP feature contributions for each available outcome model",
    )

    return parser.parse_args()


if __name__ == "__main__":

    opt = parse_opts()

    if opt.lab_data is None:

        # Radiomics-only
        radiomicsModel(
            opt.radiomics_data,
            explain=opt.explain,
        )

    else:

        # Radiomics + clinical/lab
        radiomicsClinLab(
            opt.radiomics_data,
            opt.lab_data,
            explain=opt.explain,
        )
