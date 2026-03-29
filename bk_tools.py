import glob
import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def packup_details(f):
    # Extract details from the file path and name
    parts = f.split("/")[-1].split("_")
    example = parts[0] # Generally -> SOB
    is_malign = (1 if parts[1] == "M" else 0 ) # M -> Malignant, B -> Benign
    names = parts[2].split("-")
    class_tumor = names[0] # A, F, P, S
    year = int(names[1]) + 2000 # 2014
    patient_id = names[2] # 9212
    zoom = int(names[3]) # 200
    file_id = names[4].split(".")[0] # 001
    # Pack the details into a dictionary
    return {
        "patient_id":patient_id,
        "file_id":file_id,
        "example":example,
        "class":class_tumor,
        "year":year,
        "zoom":zoom, 
        "file_path":f,
        "is_malign":is_malign
    }

def print_file_details(f):
    # Get the file name and split it to extract details (Only for testing purposes)
    parts = f.split("/")[-1].split("_")
    print("Type of example: ", parts[0])
    print("State: ",parts[1], " (", 1 if parts[1] == 'M' else 0, ")")
    nm = parts[2].split("-")
    print("Class: ", nm[0])
    print("Year: ", nm[1])
    print("Patient ID: ", nm[2])
    print("Zoom: ", nm[3])
    print("File ID: ", nm[4].split(".")[0])

def prepare_data_table(rootpath: str = "BreaKHis_v1/histology_slides/breast") -> pd.DataFrame:
    # Use glob to find all PNG files in the directory and subdirectories
    files = glob.glob(os.path.join(rootpath,"**/**.png"), recursive=True)
    # Pack the details of each file into a list of dictionaries
    datas = [packup_details(f) for f in files]
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(datas)
    # Check the structure of the DataFrame
    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns)
    return df

def prepare_data_splitting(df: pd.DataFrame, chosen_zoom: int = 200, test_val_size: int = 0.2 ):   
    __init__() # Set seeds for reproducibility

    # Choose the zoom level to work with 
    df_zoom = df[df["zoom"] == chosen_zoom].copy()
    # Get unique patients and their malignancy status for stratification
    unique_patients = df_zoom[["patient_id","is_malign"]].drop_duplicates()
    # Train - Val/Test split    
    train_p, tmp_p_ids = train_test_split(
        unique_patients['patient_id'], 
        test_size=test_val_size, 
        stratify=unique_patients['is_malign'],
        random_state=42
    )
    # Val/Test split
    tmp_df = unique_patients[unique_patients['patient_id'].isin(tmp_p_ids)]
    val_p, test_p = train_test_split(
        tmp_df['patient_id'],
        test_size=0.5,
        stratify=tmp_df['is_malign'],
        random_state=42
    )
    # Create DataFrames for each split
    train_df = df_zoom[df_zoom['patient_id'].isin(train_p)].reset_index(drop=True)
    val_df = df_zoom[df_zoom['patient_id'].isin(val_p)].reset_index(drop=True)
    test_df = df_zoom[df_zoom['patient_id'].isin(test_p)].reset_index(drop=True)

    # Filter the DataFrames for the chosen zoom level (if not already done)
    train_df_200 = train_df[train_df['zoom'] == chosen_zoom]
    val_df_200 = val_df[val_df['zoom'] == chosen_zoom]
    test_df_200 = test_df[test_df['zoom'] == chosen_zoom]

    # Check for patient ID overlaps to ensure no data leakage
    print("\nPatient split check:")
    print(train_df["patient_id"].nunique(), val_df["patient_id"].nunique(), test_df["patient_id"].nunique())
    print(set(train_df["patient_id"]) & set(val_df["patient_id"]))
    print(set(train_df["patient_id"]) & set(test_df["patient_id"]))
    print(set(val_df["patient_id"]) & set(test_df["patient_id"]))

    return train_df_200, val_df_200, test_df_200


def __init__():
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_data_splitting_class(
    df: pd.DataFrame,
    chosen_zoom: int = 200,
    test_val_size: float = 0.2,
    seed: int = 42
):
    """
    Patient-wise data splitting for multiclass classification.

    Steps:
    1. Filter data by chosen zoom level.
    2. Remove patients that have multiple class labels.
    3. Perform patient-level split to avoid data leakage.
    4. Use stratification when possible.
    5. Fall back to non-stratified validation/test split if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Must contain:
        ['patient_id', 'class', 'zoom']
    chosen_zoom : int, default=200
        Zoom level to filter.
    test_val_size : float, default=0.2
        Fraction reserved for temporary split (val + test).
        Example:
            0.2 -> 80% train, 10% val, 10% test
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    train_df : pd.DataFrame
    val_df   : pd.DataFrame
    test_df  : pd.DataFrame
    """

    set_seeds(seed)

    # --------------------------------------------------
    # 1. Filter by zoom
    # --------------------------------------------------
    df_zoom = df[df["zoom"] == chosen_zoom].copy()

    if df_zoom.empty:
        raise ValueError(f"No samples found for zoom={chosen_zoom}")

    print(f"\nSelected zoom: {chosen_zoom}")
    print(f"Total samples at selected zoom: {len(df_zoom)}")
    print(f"Total unique patients at selected zoom: {df_zoom['patient_id'].nunique()}")

    print("\nInitial image-level class distribution:")
    print(df_zoom["class"].value_counts())

    # --------------------------------------------------
    # 2. Remove patients with multiple classes
    # --------------------------------------------------
    patient_class_counts = df_zoom.groupby("patient_id")["class"].nunique()
    problematic_patients = patient_class_counts[patient_class_counts > 1].index.tolist()

    if len(problematic_patients) > 0:
        print(f"\nRemoving {len(problematic_patients)} patient(s) with multiple classes.")
        print("Problematic patient IDs:", problematic_patients)

        df_zoom = df_zoom[~df_zoom["patient_id"].isin(problematic_patients)].copy()

    if df_zoom.empty:
        raise ValueError("No samples left after removing problematic patients.")

    # --------------------------------------------------
    # 3. Create patient-level dataframe
    #    After removal, each patient has exactly one class
    # --------------------------------------------------
    patient_level_df = df_zoom[["patient_id", "class"]].drop_duplicates().reset_index(drop=True)

    print("\nClean patient-level class distribution:")
    print(patient_level_df["class"].value_counts())

    # Safety check
    check_counts = patient_level_df.groupby("patient_id")["class"].nunique()
    if (check_counts > 1).any():
        raise ValueError("Some patients still have multiple classes after cleaning.")

    # --------------------------------------------------
    # 4. First split: Train / Temp
    # --------------------------------------------------
    first_split_counts = patient_level_df["class"].value_counts()

    if first_split_counts.min() >= 2:
        print("\nFirst split: stratified")
        train_p, tmp_p = train_test_split(
            patient_level_df["patient_id"],
            test_size=test_val_size,
            stratify=patient_level_df["class"],
            random_state=seed
        )
    else:
        print("\nFirst split: non-stratified fallback")
        train_p, tmp_p = train_test_split(
            patient_level_df["patient_id"],
            test_size=test_val_size,
            random_state=seed
        )

    tmp_df = patient_level_df[patient_level_df["patient_id"].isin(tmp_p)].copy()

    print("\nTemporary patient-level class distribution (val + test pool):")
    print(tmp_df["class"].value_counts())

    # --------------------------------------------------
    # 5. Second split: Val / Test
    # --------------------------------------------------
    second_split_counts = tmp_df["class"].value_counts()

    if len(tmp_df) < 2:
        raise ValueError("Temporary split is too small to create validation and test sets.")

    if len(second_split_counts) > 0 and second_split_counts.min() >= 2:
        print("\nSecond split: stratified")
        val_p, test_p = train_test_split(
            tmp_df["patient_id"],
            test_size=0.5,
            stratify=tmp_df["class"],
            random_state=seed
        )
    else:
        print("\nSecond split: non-stratified fallback")
        val_p, test_p = train_test_split(
            tmp_df["patient_id"],
            test_size=0.5,
            random_state=seed
        )

    # --------------------------------------------------
    # 6. Build final dataframes
    # --------------------------------------------------
    train_df = df_zoom[df_zoom["patient_id"].isin(train_p)].reset_index(drop=True)
    val_df = df_zoom[df_zoom["patient_id"].isin(val_p)].reset_index(drop=True)
    test_df = df_zoom[df_zoom["patient_id"].isin(test_p)].reset_index(drop=True)

    # --------------------------------------------------
    # 7. Leakage checks
    # --------------------------------------------------
    train_patients = set(train_df["patient_id"].unique())
    val_patients = set(val_df["patient_id"].unique())
    test_patients = set(test_df["patient_id"].unique())

    print("\nPatient split check:")
    print("Train patients:", len(train_patients))
    print("Val patients  :", len(val_patients))
    print("Test patients :", len(test_patients))

    print("Train-Val overlap :", train_patients & val_patients)
    print("Train-Test overlap:", train_patients & test_patients)
    print("Val-Test overlap  :", val_patients & test_patients)

    # --------------------------------------------------
    # 8. Final class distributions
    # --------------------------------------------------
    print("\nFinal image-level class distribution:")
    print("\nTrain:")
    print(train_df["class"].value_counts())

    print("\nValidation:")
    print(val_df["class"].value_counts())

    print("\nTest:")
    print(test_df["class"].value_counts())

    print("\nFinal sample counts:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    return train_df, val_df, test_df



def prepare_best_patient_split(
    df: pd.DataFrame,
    chosen_zoom: int = 200,
    test_val_size: float = 0.2,
    n_trials: int = 200
):
    df_zoom = df[df["zoom"] == chosen_zoom].copy()

    if df_zoom.empty:
        raise ValueError(f"No samples found for zoom={chosen_zoom}")

    # remove patients with multiple classes
    patient_class_counts = df_zoom.groupby("patient_id")["class"].nunique()
    problematic_patients = patient_class_counts[patient_class_counts > 1].index.tolist()
    df_zoom = df_zoom[~df_zoom["patient_id"].isin(problematic_patients)].copy()

    patient_level_df = df_zoom[["patient_id", "class"]].drop_duplicates().reset_index(drop=True)

    all_classes = set(patient_level_df["class"].unique())
    best_score = -1
    best_split = None
    best_info = None

    for seed in range(n_trials):
        try:
            train_p, tmp_p = train_test_split(
                patient_level_df["patient_id"],
                test_size=test_val_size,
                stratify=patient_level_df["class"],
                random_state=seed
            )

            tmp_df = patient_level_df[patient_level_df["patient_id"].isin(tmp_p)].copy()

            # second split: stratify if possible, otherwise fallback
            second_counts = tmp_df["class"].value_counts()
            if len(second_counts) > 0 and second_counts.min() >= 2:
                val_p, test_p = train_test_split(
                    tmp_df["patient_id"],
                    test_size=0.5,
                    stratify=tmp_df["class"],
                    random_state=seed
                )
            else:
                val_p, test_p = train_test_split(
                    tmp_df["patient_id"],
                    test_size=0.5,
                    random_state=seed
                )

            train_df = df_zoom[df_zoom["patient_id"].isin(train_p)].reset_index(drop=True)
            val_df = df_zoom[df_zoom["patient_id"].isin(val_p)].reset_index(drop=True)
            test_df = df_zoom[df_zoom["patient_id"].isin(test_p)].reset_index(drop=True)

            train_classes = set(train_df["class"].unique())
            val_classes = set(val_df["class"].unique())
            test_classes = set(test_df["class"].unique())

            # score: class coverage in all splits
            score = len(train_classes) + len(val_classes) + len(test_classes)

            # strong bonus if every split has all classes
            if train_classes == all_classes:
                score += 100
            if val_classes == all_classes:
                score += 100
            if test_classes == all_classes:
                score += 100

            if score > best_score:
                best_score = score
                best_split = (train_df, val_df, test_df)
                best_info = {
                    "seed": seed,
                    "train_classes": sorted(train_classes),
                    "val_classes": sorted(val_classes),
                    "test_classes": sorted(test_classes),
                    "all_classes": sorted(all_classes),
                    "score": score
                }

        except Exception:
            continue

    if best_split is None:
        raise ValueError("No valid split could be created.")

    print("Best split info:")
    print(best_info)

    return best_split


def prepare_coverage_optimized_split(
    df: pd.DataFrame,
    chosen_zoom: int = 200,
    test_size: float = 0.15,
    val_size: float = 0.15,
    n_trials: int = 500,
    random_state_start: int = 0
):
    df_zoom = df[df["zoom"] == chosen_zoom].copy()

    if df_zoom.empty:
        raise ValueError(f"No samples found for zoom={chosen_zoom}")

    # Patient -> set of classes
    patient_class_map = (
        df_zoom.groupby("patient_id")["class"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
    )

    all_classes = sorted(df_zoom["class"].unique())
    all_class_set = set(all_classes)

    best_score = -np.inf
    best_split = None
    best_info = None

    patient_ids = patient_class_map["patient_id"].values

    for seed in range(random_state_start, random_state_start + n_trials):
        try:
            # First split: train vs temp
            train_ids, temp_ids = train_test_split(
                patient_ids,
                test_size=(test_size + val_size),
                random_state=seed,
                shuffle=True
            )

            # Second split: val vs test
            relative_test_size = test_size / (test_size + val_size)

            val_ids, test_ids = train_test_split(
                temp_ids,
                test_size=relative_test_size,
                random_state=seed,
                shuffle=True
            )

            train_df = df_zoom[df_zoom["patient_id"].isin(train_ids)].reset_index(drop=True)
            val_df = df_zoom[df_zoom["patient_id"].isin(val_ids)].reset_index(drop=True)
            test_df = df_zoom[df_zoom["patient_id"].isin(test_ids)].reset_index(drop=True)

            train_classes = set(train_df["class"].unique())
            val_classes = set(val_df["class"].unique())
            test_classes = set(test_df["class"].unique())

            # No leakage check
            if (
                set(train_df["patient_id"]) & set(val_df["patient_id"]) or
                set(train_df["patient_id"]) & set(test_df["patient_id"]) or
                set(val_df["patient_id"]) & set(test_df["patient_id"])
            ):
                continue

            # Coverage score
            score = 0

            # reward total coverage
            score += len(train_classes) * 3
            score += len(val_classes) * 5
            score += len(test_classes) * 5

            # big reward if all classes appear
            if train_classes == all_class_set:
                score += 100
            if val_classes == all_class_set:
                score += 200
            if test_classes == all_class_set:
                score += 200

            # penalize missing classes, especially val/test
            score -= (len(all_class_set - train_classes) * 10)
            score -= (len(all_class_set - val_classes) * 30)
            score -= (len(all_class_set - test_classes) * 30)

            # encourage more balanced image counts across splits
            score -= abs(len(val_df) - len(test_df)) * 0.05

            if score > best_score:
                best_score = score
                best_split = (train_df, val_df, test_df)
                best_info = {
                    "seed": seed,
                    "score": score,
                    "train_classes": sorted(train_classes),
                    "val_classes": sorted(val_classes),
                    "test_classes": sorted(test_classes),
                    "missing_train": sorted(all_class_set - train_classes),
                    "missing_val": sorted(all_class_set - val_classes),
                    "missing_test": sorted(all_class_set - test_classes),
                    "train_counts": train_df["class"].value_counts().to_dict(),
                    "val_counts": val_df["class"].value_counts().to_dict(),
                    "test_counts": test_df["class"].value_counts().to_dict(),
                }

        except Exception:
            continue

    if best_split is None:
        raise ValueError("No valid split found.")

    print("Best split info:")
    print(best_info)

    return best_split