import glob
import os
from pathlib import Path
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
path_db = BASE_DIR / "BreaKHis_v1" / "histology_slides" / "breast"


def packup_details(f):
    p = Path(f)

    # only filename, cross-platform safe
    filename = p.name                          # SOB_B_A-14-22549G-100-001.png
    stem = p.stem                             # SOB_B_A-14-22549G-100-001

    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")

    example = parts[0]                        # SOB
    is_malign = 1 if parts[1] == "M" else 0   # M -> malignant, B -> benign

    names = parts[2].split("-")
    if len(names) < 5:
        raise ValueError(f"Unexpected tumor name format: {filename}")

    class_tumor = names[0]
    year = int(names[1]) + 2000
    patient_id = names[2]
    zoom = int(names[3])
    file_id = names[4]

    return {
        "patient_id": patient_id,
        "file_id": file_id,
        "example": example,
        "class": class_tumor,
        "year": year,
        "zoom": zoom,
        "file_path": str(p),
        "is_malign": is_malign,
    }


def print_file_details(f):
    p = Path(f)
    filename = p.name
    stem = p.stem

    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")

    print("Type of example:", parts[0])
    print("State:", parts[1], "(", 1 if parts[1] == "M" else 0, ")")

    nm = parts[2].split("-")
    if len(nm) < 5:
        raise ValueError(f"Unexpected tumor name format: {filename}")

    print("Class:", nm[0])
    print("Year:", nm[1])
    print("Patient ID:", nm[2])
    print("Zoom:", nm[3])
    print("File ID:", nm[4])


def prepare_data_table(rootpath=path_db) -> pd.DataFrame:
    rootpath = Path(rootpath)

    # cross-platform recursive PNG search
    files = list(rootpath.rglob("*.png"))

    if not files:
        raise FileNotFoundError(f"No PNG files found under: {rootpath}")

    datas = []
    bad_files = []

    for f in files:
        try:
            datas.append(packup_details(f))
        except Exception as e:
            bad_files.append((str(f), str(e)))

    df = pd.DataFrame(datas)

    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns.tolist())
    print("Parsed files:", len(datas))
    print("Failed files:", len(bad_files))

    if bad_files:
        print("\nFirst 10 problematic files:")
        for fp, err in bad_files[:10]:
            print(fp, "->", err)

    return df

def prepare_data_splitting(df: pd.DataFrame, chosen_zoom: int = 200, test_val_size: int = 0.2 ):   
    set_seeds() # Set seeds for reproducibility

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


def set_seeds(seed: int = 42):
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
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


def prepare_data_splitting_v1(
    df: pd.DataFrame,
    chosen_zoom: int = 200,
    temp_size: float = 0.4,
    seed: int = 42
):

    df = df[df["zoom"] == chosen_zoom].copy()
    list_df_by_class = [df[df["class"] == c] for c in df["class"].unique()]
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for class_df in list_df_by_class:
        train_df, temp_df = train_test_split(
            class_df,
            test_size=temp_size,
            random_state=seed,
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=seed
        )

        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)

    final_train_df = pd.concat(train_dfs).reset_index(drop=True)
    final_val_df = pd.concat(val_dfs).reset_index(drop=True)
    final_test_df = pd.concat(test_dfs).reset_index(drop=True)

    assert final_test_df["patient_id"].isin(final_val_df["patient_id"]).any() == False
    assert final_test_df["patient_id"].isin(final_train_df["patient_id"]).any() == False
    assert final_val_df["patient_id"].isin(final_train_df["patient_id"]).any() == False

    return final_train_df, final_val_df, final_test_df