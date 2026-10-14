!pip -q install scikit-image

import numpy as np
import pandas as pd
from pathlib import Path
from google.colab import files
from skimage.feature import hog
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from zipfile import ZipFile

RANDOM_STATE = 42

# 1. UPLOAD DATA
print("Upload train, test, and sample_submission CSVs")
uploaded = files.upload()

train_file = next(f for f in uploaded if f.startswith("train"))
test_file = next(f for f in uploaded if f.startswith("test"))
sample_file = next(
    f for f in uploaded if f.startswith("sample_submission")
)

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
sample = pd.read_csv(sample_file)

y = train["Label"].to_numpy()

# 2. CONVERT RGBA DATA TO 32x32 GRAYSCALE
pixel_cols = [f"x{i}" for i in range(0, 16384, 4)]

def get_images(df):
    arr = df[pixel_cols].to_numpy(dtype=np.float32)
    arr = arr.reshape(-1, 64, 64)

    # Downsample 64x64 -> 32x32
    arr = arr.reshape(-1, 32, 2, 32, 2)
    arr = arr.mean(axis=(2, 4))

    # Normalize each image separately
    mean = arr.mean(axis=(1, 2), keepdims=True)
    std = arr.std(axis=(1, 2), keepdims=True)

    return (arr - mean) / (std + 1e-6)

train_images = get_images(train)
test_images = get_images(test)

# 3. EXTRACT TWO DIFFERENT FEATURE SETS
def profiles(images):
    rows = images.mean(axis=2)
    cols = images.mean(axis=1)
    return np.concatenate([rows, cols], axis=1)

def hog_features(images):
    return np.asarray([
        hog(
            img,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        for img in images
    ], dtype=np.float32)

feature_sets = {
    "profiles": (
        profiles(train_images),
        profiles(test_images)
    ),
    "hog": (
        hog_features(train_images),
        hog_features(test_images)
    )
}

# 4. TARGETED GRID SEARCH
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

predictions = {}

for name, (X_train, X_test) in feature_sets.items():

    print(f"\n--- {name.upper()} MODEL ---")

    n_features = X_train.shape[1]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf"))
    ])

    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": [
            "scale",
            0.25 / n_features,
            1.0 / n_features,
            4.0 / n_features
        ]
    }

    search = GridSearchCV(
        model,
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y)

    print("Best parameters:", search.best_params_)
    print("Best CV ROC-AUC:", round(search.best_score_, 5))

    # Fit probability-enabled version of selected model
    final_model = clone(search.best_estimator_)
    final_model.set_params(svm__probability=True)
    final_model.fit(X_train, y)

    class_1_index = list(final_model.classes_).index(1)

    probabilities = final_model.predict_proba(
        X_test
    )[:, class_1_index]

    # Match predictions to sample submission IDs
    assert test["ID"].is_unique
    assert sample["ID"].is_unique
    assert set(test["ID"]) == set(sample["ID"])

    id_to_probability = pd.Series(
        probabilities,
        index=test["ID"]
    )

    predictions[name] = (
        sample["ID"].map(id_to_probability).to_numpy()
    )

    submission = sample.copy()
    submission["Probability"] = predictions[name]

    submission.to_csv(
        f"submission_{name}.csv",
        index=False
    )

# 5. CREATE BLENDED SUBMISSION

# Include previous RBF model if its submission exists
previous_file = Path("submission_rbf_svm.csv")

if previous_file.exists():

    previous = pd.read_csv(previous_file)

    previous_probs = sample["ID"].map(
        previous.set_index("ID")["Probability"]
    ).to_numpy()

    assert np.isfinite(previous_probs).all()

    blended = (
        0.50 * previous_probs
        + 0.25 * predictions["profiles"]
        + 0.25 * predictions["hog"]
    )

    print("\nBlend includes previous RBF model.")

else:
    blended = (
        0.50 * predictions["profiles"]
        + 0.50 * predictions["hog"]
    )

    print("\nBlend contains the two new models.")

submission_blend = sample.copy()
submission_blend["Probability"] = blended

submission_blend.to_csv(
    "submission_blend.csv",
    index=False
)

# 6. PACKAGE SUBMISSIONS

output_files = [
    "submission_profiles.csv",
    "submission_hog.csv",
    "submission_blend.csv"
]

for filename in output_files:
    df = pd.read_csv(filename)

    assert df.columns.tolist() == ["ID", "Probability"]
    assert len(df) == len(sample)
    assert df["Probability"].between(0, 1).all()

with ZipFile("svm_experiments.zip", "w") as archive:
    for filename in output_files:
        archive.write(filename)

print("\nAll submissions generated successfully.")
print("Files:", output_files)

files.download("svm_experiments.zip")
