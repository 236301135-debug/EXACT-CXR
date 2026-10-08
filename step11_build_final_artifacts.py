
"""
EXACT-CXR — STEP 11
BUILD AND VERIFY FINAL MODEL ARTIFACTS

Purpose:
    Build the reproducible model artifacts selected in Step 10.

Selected experimental models:
    1. Cardiomegaly -> pretrained TorchXRayVision DenseNet
    2. Pneumonia    -> XRV frozen features + StandardScaler
                       + LogisticRegression

IMPORTANT:
    Results are based on only 24 images and are NOT clinical validation.

    This script intentionally does NOT fine-tune the XRV backbone.

    Step 11 also verifies that:
        - preprocessing works correctly
        - XRV feature extraction works
        - pneumonia artifact can be saved and reloaded
        - cardiomegaly XRV inference works
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import ast
import json
import math
import pickle

import numpy as np
import pandas as pd
import torch

from PIL import Image

import torchxrayvision as xrv

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "data" / "sample"

CSV_PATH = (
    IMAGE_DIR /
    "chest_x_ray_images_labels_sample.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

STEP10_CONFIG = (
    OUTPUT_DIR /
    "step10_final_label_config.json"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ------------------------------------------------------------
# Pneumonia feature model
# Selected in Step 10:
#
# C = 0.001
# class_weight = balanced
# threshold = 0.40
# ------------------------------------------------------------

PNEUMONIA_C = 0.001

PNEUMONIA_CLASS_WEIGHT = "balanced"

PNEUMONIA_THRESHOLD = 0.40


# ------------------------------------------------------------
# Cardiomegaly
#
# Step 7 / Step 10:
# pretrained XRV
# XRV target index = 10
# threshold selected in Step 9 = 0.45
# ------------------------------------------------------------

CARDIOMEGALY_THRESHOLD = 0.45

CARDIOMEGALY_XRV_INDEX = 10


# ------------------------------------------------------------
# XRV model
# ------------------------------------------------------------

XRV_WEIGHTS = (
    "densenet121-res224-all"
)


# ============================================================
# JSON SANITIZER
# ============================================================

def sanitize_json(obj):
    """
    Recursively convert NumPy / Torch values and non-finite
    floating-point values into JSON-safe Python values.
    """

    if isinstance(obj, dict):

        return {
            str(k): sanitize_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):

        return [
            sanitize_json(v)
            for v in obj
        ]

    if isinstance(obj, tuple):

        return [
            sanitize_json(v)
            for v in obj
        ]

    if isinstance(
        obj,
        np.ndarray
    ):

        return sanitize_json(
            obj.tolist()
        )

    if isinstance(
        obj,
        torch.Tensor
    ):

        return sanitize_json(
            obj.detach()
            .cpu()
            .numpy()
        )

    if isinstance(
        obj,
        np.integer
    ):

        return int(obj)

    if isinstance(
        obj,
        np.floating
    ):

        value = float(obj)

        if not math.isfinite(value):
            return None

        return value

    if isinstance(
        obj,
        float
    ):

        if not math.isfinite(obj):
            return None

        return obj

    return obj


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print()
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"CSV rows: {len(df)}"
    )

    print(
        f"CSV columns: {len(df.columns)}"
    )

    return df


# ============================================================
# LABEL PARSER
# ============================================================

def parse_labels(value):
    """
    Parse the CSV Labels field.

    Example:

        "['pneumonia', 'cardiomegaly']"

    becomes:

        ['pneumonia', 'cardiomegaly']
    """

    if pd.isna(value):

        return []

    if isinstance(
        value,
        list
    ):

        return [
            str(x).strip()
            for x in value
        ]

    try:

        parsed = ast.literal_eval(
            str(value)
        )

        if isinstance(
            parsed,
            list
        ):

            return [
                str(x).strip()
                for x in parsed
            ]

    except Exception:

        pass

    return []


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def get_image_path(row):

    image_id = str(
        row["ImageID"]
    ).strip()

    candidates = [

        IMAGE_DIR / image_id,

        IMAGE_DIR /
        f"{image_id}.png",

        IMAGE_DIR /
        f"{image_id}.jpg",

        IMAGE_DIR /
        f"{image_id}.jpeg",
    ]

    image_dir = row.get(
        "ImageDir",
        ""
    )

    if not pd.isna(
        image_dir
    ):

        image_dir = str(
            image_dir
        ).strip()

        if image_dir:

            candidates.extend([

                IMAGE_DIR /
                image_dir /
                image_id,

                IMAGE_DIR /
                image_dir /
                f"{image_id}.png",

                IMAGE_DIR /
                image_dir /
                f"{image_id}.jpg",

                IMAGE_DIR /
                image_dir /
                f"{image_id}.jpeg",
            ])

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        f"Could not find image "
        f"for ImageID={image_id}"
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_xrv_image(image_path):
    """
    Load and preprocess a uint16 PNG for TorchXRayVision.

    IMPORTANT:

    TorchXRayVision's XRayCenterCrop and XRayResizer operate
    on NumPy arrays.

    Therefore the correct sequence is:

        PIL
          |
          v
        NumPy float32
          |
          v
        normalize to [-1024,1024]
          |
          v
        shape [1,H,W]
          |
          v
        XRayCenterCrop
          |
          v
        XRayResizer(224)
          |
          v
        torch.from_numpy
          |
          v
        Tensor [1,224,224]

    This fixes the previous error:

        TypeError:
        Cannot interpret 'torch.float32' as a data type
    """

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("I")

    array = np.asarray(
        image,
        dtype=np.float32
    )

    if array.ndim != 2:

        raise ValueError(
            "Expected grayscale 2-D image, "
            f"got shape {array.shape}"
        )

    # --------------------------------------------------------
    # Source PNGs are uint16 representations of the original
    # 10/12-bit data.
    #
    # The audited dataset showed that the source data had been
    # linearly rescaled into approximately the uint16 range.
    #
    # Convert:
    #
    #       [0,65535]
    #
    # to:
    #
    #       [-1024,1024]
    # --------------------------------------------------------

    array = (
        (array / 65535.0)
        * 2048.0
        - 1024.0
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # XRV transforms expect NumPy [C,H,W],
    # not a PyTorch tensor.
    # --------------------------------------------------------

    array = array[
        None,
        ...
    ]

    # --------------------------------------------------------
    # Center crop
    # --------------------------------------------------------

    array = (
        xrv.datasets
        .XRayCenterCrop()
        (array)
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    array = (
        xrv.datasets
        .XRayResizer(224)
        (array)
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not isinstance(
        array,
        np.ndarray
    ):

        raise TypeError(
            "XRV preprocessing did not return "
            "a NumPy array."
        )

    if array.shape != (
        1,
        224,
        224
    ):

        raise ValueError(
            "Unexpected preprocessed shape: "
            f"{array.shape}; expected "
            "(1,224,224)"
        )

    if not np.all(
        np.isfinite(array)
    ):

        raise ValueError(
            "Preprocessed image contains "
            "NaN or infinite values."
        )

    # --------------------------------------------------------
    # Convert NumPy -> Torch only AFTER XRV preprocessing
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        array
    ).float()

    return tensor


# ============================================================
# LOAD XRV MODEL
# ============================================================

def load_xrv_model():

    print()
    print("=" * 70)
    print("LOADING TORCHXRAYVISION")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Weights: {XRV_WEIGHTS}"
    )

    model = xrv.models.DenseNet(
        weights=XRV_WEIGHTS
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "XRV model loaded successfully."
    )

    print()
    print(
        "XRV targets:"
    )

    for i, name in enumerate(
        model.targets
    ):

        print(
            f"{i:2d}: {name}"
        )

    return model


# ============================================================
# XRV FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_xrv_features(
    model,
    tensor
):
    """
    Extract the frozen DenseNet representation.

    TorchXRayVision documents model.features(...) as the
    feature-extraction interface.

    For DenseNet121 this produces the feature representation
    used by the Step 8 / Step 9B feature-model experiments.
    """

    # --------------------------------------------------------
    # [1,224,224]
    # ->
    # [1,1,224,224]
    # --------------------------------------------------------

    batch = (
        tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Frozen XRV feature extractor
    # --------------------------------------------------------

    features = model.features(
        batch
    )

    # --------------------------------------------------------
    # Some versions / configurations may return a spatial
    # feature map. Convert it to one vector using global
    # average pooling.
    # --------------------------------------------------------

    if features.ndim == 4:

        features = (
            torch.nn.functional
            .adaptive_avg_pool2d(
                features,
                (1, 1)
            )
        )

        features = features.flatten(
            1
        )

    elif features.ndim != 2:

        raise ValueError(
            "Unexpected XRV feature tensor "
            f"shape: {tuple(features.shape)}"
        )

    features = (
        features
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    if not np.all(
        np.isfinite(features)
    ):

        raise ValueError(
            "XRV feature vector contains "
            "NaN or infinite values."
        )

    return features


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

def build_feature_matrix(
    df,
    model
):

    print()
    print("=" * 70)
    print("BUILDING XRV FEATURE MATRIX")
    print("=" * 70)

    features = []

    labels = []

    paths = []

    for i, row in df.iterrows():

        image_path = get_image_path(
            row
        )

        image_tensor = load_xrv_image(
            image_path
        )

        feature_vector = (
            extract_xrv_features(
                model,
                image_tensor
            )
        )

        features.append(
            feature_vector
        )

        labels.append(
            parse_labels(
                row["Labels"]
            )
        )

        paths.append(
            str(image_path)
        )

        print(
            f"[{i + 1:02d}/{len(df):02d}] "
            f"{image_path.name} "
            f"features={feature_vector.shape}"
        )

    X = np.vstack(
        features
    ).astype(
        np.float32
    )

    print()
    print(
        f"Feature matrix shape: "
        f"{X.shape}"
    )

    print(
        f"Feature min: "
        f"{X.min():.6f}"
    )

    print(
        f"Feature max: "
        f"{X.max():.6f}"
    )

    print(
        f"Feature mean: "
        f"{X.mean():.6f}"
    )

    return (
        X,
        labels,
        paths
    )


# ============================================================
# BINARY TARGET
# ============================================================

def make_binary_target(
    label_lists,
    target
):

    target_key = (
        target
        .strip()
        .casefold()
    )

    y = np.array(

        [
            int(

                any(

                    str(label)
                    .strip()
                    .casefold()
                    == target_key

                    for label in labels

                )

            )

            for labels in label_lists

        ],

        dtype=np.int64
    )

    return y


# ============================================================
# TRAIN PNEUMONIA MODEL
# ============================================================

def train_pneumonia_model(
    X,
    label_lists
):

    print()
    print("=" * 70)
    print("TRAINING PNEUMONIA FEATURE MODEL")
    print("=" * 70)

    y = make_binary_target(
        label_lists,
        "pneumonia"
    )

    positives = int(
        y.sum()
    )

    negatives = int(
        len(y) - positives
    )

    print(
        f"Samples: {len(y)}"
    )

    print(
        f"Positive: {positives}"
    )

    print(
        f"Negative: {negatives}"
    )

    if positives < 2:

        raise RuntimeError(
            "Not enough pneumonia-positive "
            "examples."
        )

    # --------------------------------------------------------
    # StandardScaler
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    classifier = LogisticRegression(

        C=PNEUMONIA_C,

        class_weight=(
            PNEUMONIA_CLASS_WEIGHT
        ),

        max_iter=5000,

        solver="liblinear",

        random_state=42
    )

    classifier.fit(
        X_scaled,
        y
    )

    probabilities = (
        classifier
        .predict_proba(
            X_scaled
        )[:, 1]
    )

    print()
    print(
        "Pneumonia model trained."
    )

    print(
        f"C: {PNEUMONIA_C}"
    )

    print(
        f"class_weight: "
        f"{PNEUMONIA_CLASS_WEIGHT}"
    )

    print(
        f"threshold: "
        f"{PNEUMONIA_THRESHOLD}"
    )

    print()
    print(
        "Training-set probability range:"
    )

    print(
        f"min={probabilities.min():.6f}"
    )

    print(
        f"max={probabilities.max():.6f}"
    )

    return {

        "model_family":
            "XRV frozen features + LogisticRegression",

        "feature_dimension":
            int(X.shape[1]),

        "scaler":
            scaler,

        "classifier":
            classifier,

        "threshold":
            PNEUMONIA_THRESHOLD,

        "positive_count":
            positives,

        "negative_count":
            negatives
    }


# ============================================================
# SAVE PNEUMONIA MODEL
# ============================================================

def save_pneumonia_model(
    artifact
):

    path = (
        OUTPUT_DIR /
        "step11_pneumonia_xrv_feature_lr.pkl"
    )

    with open(
        path,
        "wb"
    ) as f:

        pickle.dump(

            artifact,

            f,

            protocol=(
                pickle.HIGHEST_PROTOCOL
            )
        )

    print()
    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# SAVE XRV METADATA
# ============================================================

def save_xrv_metadata(
    model
):

    path = (
        OUTPUT_DIR /
        "step11_xrv_model_metadata.json"
    )

    metadata = {

        "model_family":
            "TorchXRayVision",

        "weights":
            XRV_WEIGHTS,

        "device_used_for_build":
            str(DEVICE),

        "input_size":
            224,

        "preprocessing":
        {
            "source_dtype":
                "uint16",

            "source_range":
                "[0,65535]",

            "xrv_range":
                "[-1024,1024]",

            "conversion":
                "(pixel / 65535) * 2048 - 1024",

            "shape_before_xrv_transforms":
                "[1,H,W]",

            "center_crop":
                True,

            "resize":
                224,

            "final_tensor_shape":
                "[1,224,224]"
        },

        "targets":
        [
            str(x)
            for x in model.targets
        ],

        "cardiomegaly":
        {
            "target_name":
                "Cardiomegaly",

            "xrv_target_index":
                CARDIOMEGALY_XRV_INDEX,

            "threshold":
                CARDIOMEGALY_THRESHOLD
        }
    }

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            sanitize_json(
                metadata
            ),

            f,

            indent=2,

            allow_nan=False
        )

    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# SAVE FINAL CONFIG
# ============================================================

def save_final_config(
    pneumonia_path,
    xrv_metadata_path,
    feature_dimension
):

    print()
    print("=" * 70)
    print("CREATING FINAL EXACT-CXR MODEL CONFIG")
    print("=" * 70)

    config = {

        "project":
            "EXACT-CXR",

        "step":
            11,

        "dataset":
        {
            "images_used":
                24,

            "source_csv":
                str(CSV_PATH),

            "clinical_validation":
                False,

            "warning":
                (
                    "Experimental research results "
                    "only. Not clinical validation."
                )
        },

        "base_model":
        {
            "family":
                "TorchXRayVision",

            "weights":
                XRV_WEIGHTS,

            "input_size":
                224
        },

        "preprocessing":
        {
            "source_dtype":
                "uint16",

            "source_range":
                [0, 65535],

            "normalized_range":
                [-1024, 1024],

            "center_crop":
                True,

            "resize":
                224
        },

        "models":
        {

            "cardiomegaly":
            {
                "label":
                    "Cardiomegaly",

                "model_type":
                    "pretrained_xrv",

                "xrv_target":
                    "Cardiomegaly",

                "xrv_target_index":
                    CARDIOMEGALY_XRV_INDEX,

                "threshold":
                    CARDIOMEGALY_THRESHOLD,

                "step10_status":
                    "EXPERIMENTAL_BEST",

                "step10_auroc":
                    0.95,

                "step10_auprc":
                    0.875,

                "step10_f1":
                    0.8571428571428571
            },

            "pneumonia":
            {
                "label":
                    "Pneumonia",

                "model_type":
                    "xrv_features_logistic_regression",

                "artifact":
                    str(pneumonia_path),

                "xrv_weights":
                    XRV_WEIGHTS,

                "feature_dimension":
                    feature_dimension,

                "C":
                    PNEUMONIA_C,

                "class_weight":
                    PNEUMONIA_CLASS_WEIGHT,

                "threshold":
                    PNEUMONIA_THRESHOLD,

                "step10_status":
                    "EXPERIMENTAL_BEST",

                "step10_auroc":
                    0.8571428571428572,

                "step10_auprc":
                    0.6,

                "step10_f1":
                    0.4615384615384615
            }
        },

        "not_deployed":
        {

            "costophrenic_angle_blunting":
            {
                "status":
                    "EXPERIMENTAL_ONLY",

                "reason":
                    (
                        "Only 4 positive examples; "
                        "not deployed."
                    )
            },

            "interstitial_pattern":
            {
                "status":
                    "EXPERIMENTAL_ONLY",

                "reason":
                    (
                        "Only 2 positive examples and "
                        "current F1 is 0; not deployed."
                    )
            },

            "rib_fracture":
            {
                "status":
                    "EXPERIMENTAL_ONLY",

                "reason":
                    (
                        "AUROC 0.295 and sensitivity 0; "
                        "not deployed."
                    )
            }
        }
    }

    path = (
        OUTPUT_DIR /
        "step11_final_model_config.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            sanitize_json(
                config
            ),

            f,

            indent=2,

            allow_nan=False
        )

    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# VERIFY PNEUMONIA ARTIFACT
# ============================================================

def verify_pneumonia_artifact(
    artifact_path,
    X
):

    print()
    print("=" * 70)
    print("VERIFYING PNEUMONIA ARTIFACT")
    print("=" * 70)

    with open(
        artifact_path,
        "rb"
    ) as f:

        artifact = pickle.load(
            f
        )

    if not isinstance(
        artifact,
        dict
    ):

        raise RuntimeError(
            "Pneumonia artifact is not a dictionary."
        )

    scaler = artifact[
        "scaler"
    ]

    classifier = artifact[
        "classifier"
    ]

    threshold = float(
        artifact[
            "threshold"
        ]
    )

    # --------------------------------------------------------
    # Verify feature dimension
    # --------------------------------------------------------

    feature_dimension = int(
        X.shape[1]
    )

    saved_dimension = int(
        artifact[
            "feature_dimension"
        ]
    )

    if feature_dimension != saved_dimension:

        raise RuntimeError(
            "Feature dimension mismatch: "
            f"current={feature_dimension}, "
            f"saved={saved_dimension}"
        )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    probabilities = (
        classifier
        .predict_proba(
            X_scaled
        )[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(
        np.int64
    )

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(
            probabilities
        )
    ):

        raise RuntimeError(
            "Reloaded pneumonia model produced "
            "non-finite probabilities."
        )

    print(
        "Scaler type: "
        f"{type(scaler).__name__}"
    )

    print(
        "Classifier type: "
        f"{type(classifier).__name__}"
    )

    print(
        f"Feature dimension: "
        f"{feature_dimension}"
    )

    print(
        f"Reloaded probabilities shape: "
        f"{probabilities.shape}"
    )

    print(
        f"Probability min: "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Probability max: "
        f"{probabilities.max():.6f}"
    )

    print(
        f"Threshold: "
        f"{threshold:.2f}"
    )

    print(
        "Predicted positives: "
        f"{predictions.sum()}"
    )

    print()
    print(
        "Pneumonia artifact verification: PASS"
    )


# ============================================================
# CARDIOMEGALY SMOKE TEST
# ============================================================

@torch.no_grad()
def verify_cardiomegaly(
    model,
    df
):

    print()
    print("=" * 70)
    print("CARDIOMEGALY XRV SMOKE TEST")
    print("=" * 70)

    values = []

    for i, row in df.iterrows():

        image_path = get_image_path(
            row
        )

        image_tensor = load_xrv_image(
            image_path
        )

        batch = (
            image_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        output = model(
            batch
        )

        # ----------------------------------------------------
        # TorchXRayVision DenseNet normally returns sigmoid
        # probabilities because DenseNet defaults to
        # apply_sigmoid=True.
        #
        # To remain robust across configurations, detect the
        # range and only apply sigmoid when necessary.
        # ----------------------------------------------------

        raw_value = float(
            output[
                0,
                CARDIOMEGALY_XRV_INDEX
            ].item()
        )

        if (
            raw_value < 0.0
            or raw_value > 1.0
        ):

            probability = float(
                torch.sigmoid(
                    output[
                        0,
                        CARDIOMEGALY_XRV_INDEX
                    ]
                ).item()
            )

        else:

            probability = raw_value

        values.append(
            probability
        )

        print(
            f"{image_path.name:40s} "
            f"cardiomegaly={probability:.6f}"
        )

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if not np.all(
        np.isfinite(values)
    ):

        raise RuntimeError(
            "Cardiomegaly XRV produced "
            "non-finite predictions."
        )

    print()
    print(
        f"Probability min: "
        f"{values.min():.6f}"
    )

    print(
        f"Probability max: "
        f"{values.max():.6f}"
    )

    print()
    print(
        "Cardiomegaly XRV smoke test: PASS"
    )


# ============================================================
# PREPROCESSING SMOKE TEST
# ============================================================

def verify_preprocessing(
    df
):

    print()
    print("=" * 70)
    print("PREPROCESSING SMOKE TEST")
    print("=" * 70)

    row = df.iloc[0]

    image_path = get_image_path(
        row
    )

    tensor = load_xrv_image(
        image_path
    )

    print(
        f"Image: {image_path.name}"
    )

    print(
        f"Final tensor shape: "
        f"{tuple(tensor.shape)}"
    )

    print(
        f"Tensor dtype: "
        f"{tensor.dtype}"
    )

    print(
        f"Tensor min: "
        f"{tensor.min().item():.6f}"
    )

    print(
        f"Tensor max: "
        f"{tensor.max().item():.6f}"
    )

    if tensor.shape != (
        1,
        224,
        224
    ):

        raise RuntimeError(
            "Preprocessing smoke test failed: "
            f"shape={tuple(tensor.shape)}"
        )

    if not torch.isfinite(
        tensor
    ).all():

        raise RuntimeError(
            "Preprocessing smoke test failed: "
            "non-finite values."
        )

    print()
    print(
        "Preprocessing smoke test: PASS"
    )


# ============================================================
# SAVE FEATURE MATRIX
# ============================================================

def save_feature_matrix(
    X
):

    path = (
        OUTPUT_DIR /
        "step11_xrv_features_24x1024.npy"
    )

    np.save(
        path,
        X
    )

    print()
    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXACT-CXR — STEP 11")
    print("BUILD AND VERIFY FINAL MODEL ARTIFACTS")
    print("=" * 70)

    print()
    print(
        "IMPORTANT: This step creates experimental "
        "research artifacts."
    )

    print(
        "It does NOT create clinically validated models."
    )

    # ========================================================
    # STEP 10 CHECK
    # ========================================================

    if not STEP10_CONFIG.exists():

        raise FileNotFoundError(
            "Step 10 configuration missing:\n"
            f"{STEP10_CONFIG}"
        )

    print()
    print(
        f"OK: {STEP10_CONFIG.name}"
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    df = load_dataset()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_xrv_model()

    # ========================================================
    # PREPROCESSING TEST
    # ========================================================

    verify_preprocessing(
        df
    )

    # ========================================================
    # BUILD FEATURES
    # ========================================================

    (
        X,
        label_lists,
        image_paths
    ) = build_feature_matrix(
        df,
        model
    )

    # ========================================================
    # FEATURE DIMENSION CHECK
    # ========================================================

    feature_dimension = int(
        X.shape[1]
    )

    print()
    print(
        f"Detected XRV feature dimension: "
        f"{feature_dimension}"
    )

    if feature_dimension != 1024:

        raise RuntimeError(
            "Expected 1024-dimensional "
            f"XRV features, got "
            f"{feature_dimension}."
        )

    # ========================================================
    # PNEUMONIA TARGET
    # ========================================================

    pneumonia_y = make_binary_target(
        label_lists,
        "pneumonia"
    )

    # ========================================================
    # TRAIN PNEUMONIA FEATURE MODEL
    # ========================================================

    pneumonia_artifact = (
        train_pneumonia_model(
            X,
            label_lists
        )
    )

    # ========================================================
    # SAVE PNEUMONIA MODEL
    # ========================================================

    pneumonia_path = (
        save_pneumonia_model(
            pneumonia_artifact
        )
    )

    # ========================================================
    # SAVE XRV METADATA
    # ========================================================

    xrv_metadata_path = (
        save_xrv_metadata(
            model
        )
    )

    # ========================================================
    # VERIFY PNEUMONIA ARTIFACT
    # ========================================================

    verify_pneumonia_artifact(
        pneumonia_path,
        X
    )

    # ========================================================
    # VERIFY CARDIOMEGALY
    # ========================================================

    verify_cardiomegaly(
        model,
        df
    )

    # ========================================================
    # SAVE FINAL CONFIG
    # ========================================================

    final_config_path = (
        save_final_config(
            pneumonia_path,
            xrv_metadata_path,
            feature_dimension
        )
    )

    # ========================================================
    # SAVE FEATURE MATRIX
    # ========================================================

    feature_path = (
        save_feature_matrix(
            X
        )
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 11 COMPLETE")
    print("=" * 70)

    print()
    print(
        "EXPERIMENTAL MODEL ARTIFACTS:"
    )

    print()
    print(
        "1. Cardiomegaly"
    )

    print(
        "   Model: pretrained TorchXRayVision"
    )

    print(
        "   XRV target index: 10"
    )

    print(
        "   Threshold: 0.45"
    )

    print(
        "   Step 10 AUROC: 0.95"
    )

    print()
    print(
        "2. Pneumonia"
    )

    print(
        "   Model: XRV frozen features "
        "+ Logistic Regression"
    )

    print(
        "   Feature dimension: "
        f"{feature_dimension}"
    )

    print(
        "   C: 0.001"
    )

    print(
        "   class_weight: balanced"
    )

    print(
        "   Threshold: 0.40"
    )

    print(
        "   Step 10 AUROC: 0.8571"
    )

    print()
    print(
        "FILES SAVED:"
    )

    print(
        pneumonia_path
    )

    print(
        xrv_metadata_path
    )

    print(
        final_config_path
    )

    print(
        feature_path
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These models are experimental."
    )

    print(
        "They were evaluated using only 24 images."
    )

    print(
        "They must NOT be represented "
        "as clinically validated."
    )

    print()
    print(
        "STEP 11 FINISHED SUCCESSFULLY."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

