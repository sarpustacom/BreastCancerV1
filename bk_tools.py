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


def prepare_image_level_split(
    df: pd.DataFrame,
    chosen_zoom: int = 200,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42
):
    """
    Image-level stratified split for multiclass classification.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ['class', 'zoom'] and file path column.
    chosen_zoom : int
        Zoom level to filter.
    test_size : float
        Final test ratio.
    val_size : float
        Final validation ratio.
    seed : int
        Random seed.

    Returns
    -------
    train_df, val_df, test_df
    """
    set_seeds(seed)

    df_zoom = df[df["zoom"] == chosen_zoom].copy().reset_index(drop=True)

    if df_zoom.empty:
        raise ValueError(f"No samples found for zoom={chosen_zoom}")

    print(f"Selected zoom: {chosen_zoom}")
    print(f"Total images: {len(df_zoom)}")
    print("\nClass distribution before split:")
    print(df_zoom["class"].value_counts())

    # First split: train vs temp
    train_df, temp_df = train_test_split(
        df_zoom,
        test_size=(val_size + test_size),
        stratify=df_zoom["class"],
        random_state=seed,
        shuffle=True
    )

    # Relative test size inside temp
    relative_test_size = test_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["class"],
        random_state=seed,
        shuffle=True
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\nSplit sizes:")
    print("Train:", len(train_df))
    print("Val  :", len(val_df))
    print("Test :", len(test_df))

    print("\nTrain class distribution:")
    print(train_df["class"].value_counts())

    print("\nVal class distribution:")
    print(val_df["class"].value_counts())

    print("\nTest class distribution:")
    print(test_df["class"].value_counts())

    # Optional leakage check
    if "patient_id" in df_zoom.columns:
        train_patients = set(train_df["patient_id"].unique())
        val_patients = set(val_df["patient_id"].unique())
        test_patients = set(test_df["patient_id"].unique())

        print("\nPatient overlap check (expected in image-level split):")
        print("Train-Val overlap :", len(train_patients & val_patients))
        print("Train-Test overlap:", len(train_patients & test_patients))
        print("Val-Test overlap  :", len(val_patients & test_patients))

    return train_df, val_df, test_df


def prepare_data_splitting_class_v2(
    df: pd.DataFrame,
    chosen_zoom: int = None,
    temp_size: float = 0.2,
    seed: int = 42
):
    """
    Patient-wise data splitting for multiclass classification.

    Steps:
    1. Optionally filter data by zoom level.
    2. Remove patients that have multiple class labels (with warning).
    3. Perform patient-level split to avoid data leakage.
    4. Use stratification when possible, fall back otherwise.
    5. Assign integer labels and return class mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Must contain:
        ['patient_id', 'class', 'zoom', 'file_path']
    chosen_zoom : int or None, default=None
        Zoom level to filter. If None, all zoom levels are used.
    temp_size : float, default=0.2
        Fraction reserved for val + test pool.
        Example:
            0.2 -> ~80% train, ~10% val, ~10% test
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    train_df : pd.DataFrame
    val_df   : pd.DataFrame
    test_df  : pd.DataFrame
    class_to_idx : dict
        Mapping from class name to integer label.
    """

    set_seeds(seed)

    # --------------------------------------------------
    # 1. Zoom filtresi (opsiyonel)
    # --------------------------------------------------
    if chosen_zoom is not None:
        df_zoom = df[df["zoom"] == chosen_zoom].copy().reset_index(drop=True)
        if df_zoom.empty:
            raise ValueError(f"No samples found for zoom={chosen_zoom}")
        print(f"\nSelected zoom: {chosen_zoom}")
    else:
        df_zoom = df.copy().reset_index(drop=True)
        print("\nNo zoom filter applied — using all zoom levels.")

    print(f"Total samples : {len(df_zoom)}")
    print(f"Unique patients: {df_zoom['patient_id'].nunique()}")
    print("\nInitial image-level class distribution:")
    print(df_zoom["class"].value_counts())

    # --------------------------------------------------
    # 2. Birden fazla sınıfı olan hastaları çıkar
    # --------------------------------------------------
    patient_class_counts = df_zoom.groupby("patient_id")["class"].nunique()
    problematic_patients = patient_class_counts[patient_class_counts > 1].index.tolist()

    if len(problematic_patients) > 0:
        removed_images = df_zoom[df_zoom["patient_id"].isin(problematic_patients)].shape[0]
        print(f"\n[WARNING] {len(problematic_patients)} patient(s) with multiple classes found.")
        print(f"          Removing {removed_images} images ({removed_images/len(df_zoom)*100:.1f}% of data).")
        print(f"          Problematic patient IDs: {problematic_patients}")
        df_zoom = df_zoom[~df_zoom["patient_id"].isin(problematic_patients)].copy().reset_index(drop=True)
    else:
        print("\nNo patients with multiple classes found.")

    if df_zoom.empty:
        raise ValueError("No samples left after removing problematic patients.")

    # --------------------------------------------------
    # 3. Hasta bazlı dataframe oluştur
    # --------------------------------------------------
    patient_level_df = (
        df_zoom[["patient_id", "class"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print("\nClean patient-level class distribution:")
    print(patient_level_df["class"].value_counts())

    # Güvenlik kontrolü
    check = patient_level_df.groupby("patient_id")["class"].nunique()
    if (check > 1).any():
        raise ValueError("Some patients still have multiple classes after cleaning.")

    # --------------------------------------------------
    # 4. Birinci split: Train / Temp
    # --------------------------------------------------
    class_counts = patient_level_df["class"].value_counts()

    if class_counts.min() >= 2:
        print("\nFirst split: stratified")
        train_p, tmp_p = train_test_split(
            patient_level_df["patient_id"],
            test_size=temp_size,
            stratify=patient_level_df["class"],
            random_state=seed
        )
    else:
        print("\nFirst split: non-stratified fallback (some classes have < 2 patients)")
        train_p, tmp_p = train_test_split(
            patient_level_df["patient_id"],
            test_size=temp_size,
            random_state=seed
        )

    tmp_patients = patient_level_df[patient_level_df["patient_id"].isin(tmp_p)].copy()

    print("\nTemp pool patient-level class distribution (val + test):")
    print(tmp_patients["class"].value_counts())

    # --------------------------------------------------
    # 5. İkinci split: Val / Test
    # --------------------------------------------------
    if len(tmp_patients) < 2:
        raise ValueError("Temp pool is too small to split into validation and test sets.")

    tmp_counts = tmp_patients["class"].value_counts()

    if tmp_counts.min() >= 2:
        print("\nSecond split: stratified")
        val_p, test_p = train_test_split(
            tmp_patients["patient_id"],
            test_size=0.5,
            stratify=tmp_patients["class"],
            random_state=seed
        )
    else:
        print("\nSecond split: non-stratified fallback (some classes have < 2 patients in temp pool)")
        val_p, test_p = train_test_split(
            tmp_patients["patient_id"],
            test_size=0.5,
            random_state=seed
        )

    # --------------------------------------------------
    # 6. Final dataframe'leri oluştur
    # --------------------------------------------------
    train_df = df_zoom[df_zoom["patient_id"].isin(train_p)].reset_index(drop=True)
    val_df   = df_zoom[df_zoom["patient_id"].isin(val_p)].reset_index(drop=True)
    test_df  = df_zoom[df_zoom["patient_id"].isin(test_p)].reset_index(drop=True)

    # --------------------------------------------------
    # 7. Sızıntı kontrolü
    # --------------------------------------------------
    train_patients = set(train_df["patient_id"].unique())
    val_patients   = set(val_df["patient_id"].unique())
    test_patients  = set(test_df["patient_id"].unique())

    tv_overlap  = train_patients & val_patients
    tt_overlap  = train_patients & test_patients
    vt_overlap  = val_patients  & test_patients

    print("\nPatient overlap check (must all be empty):")
    print(f"  Train-Val overlap : {tv_overlap  if tv_overlap  else 'None — OK'}")
    print(f"  Train-Test overlap: {tt_overlap  if tt_overlap  else 'None — OK'}")
    print(f"  Val-Test overlap  : {vt_overlap  if vt_overlap  else 'None — OK'}")

    if tv_overlap or tt_overlap or vt_overlap:
        raise ValueError("Data leakage detected! Patient overlap between splits.")

    # --------------------------------------------------
    # 8. Label mapping
    # --------------------------------------------------
    class_names  = sorted(df_zoom["class"].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df["label"] = train_df["class"].map(class_to_idx)
    val_df["label"]   = val_df["class"].map(class_to_idx)
    test_df["label"]  = test_df["class"].map(class_to_idx)

    # --------------------------------------------------
    # 9. Özet rapor
    # --------------------------------------------------
    print("\n" + "="*50)
    print("SPLIT SUMMARY")
    print("="*50)
    print(f"{'Split':<12} {'Patients':>10} {'Images':>10}")
    print("-"*35)
    print(f"{'Train':<12} {len(train_patients):>10} {len(train_df):>10}")
    print(f"{'Validation':<12} {len(val_patients):>10} {len(val_df):>10}")
    print(f"{'Test':<12} {len(test_patients):>10} {len(test_df):>10}")
    print(f"{'Total':<12} {len(train_patients|val_patients|test_patients):>10} {len(df_zoom):>10}")
    print("="*50)

    print("\nClass label mapping:")
    for cls, idx in class_to_idx.items():
        print(f"  {cls} -> {idx}")

    print("\nImage-level class distribution per split:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{split_name}:")
        print(split_df["class"].value_counts().to_string())

    return train_df, val_df, test_df, class_to_idx


def prepare_data_splitting_class_v3(
    df: pd.DataFrame,
    chosen_zoom: int = None,
    temp_size: float = 0.2,
    seed: int = 42,
    expected_classes: list = None
):
    """
    V3: Patient-wise data splitting with guaranteed stratification and static label mapping.
    """
    set_seeds(seed)

    # BreaKHis veri seti için varsayılan sınıflar (Statik Mapping için zorunlu)
    if expected_classes is None:
        expected_classes = ['A', 'DC', 'F', 'LC', 'MC', 'PC', 'PT', 'TA']

    # --------------------------------------------------
    # 1. Robust Zoom Filtresi (Tip uyuşmazlığını çözer)
    # --------------------------------------------------
    if chosen_zoom is not None:
        # "200" ve "200X" gibi string/int karmaşasını çözmek için string contains kullanılır
        zoom_str = str(chosen_zoom).replace("X", "").replace("x", "")
        df_zoom = df[df["zoom"].astype(str).str.contains(zoom_str, case=False, na=False)].copy().reset_index(drop=True)
        
        if df_zoom.empty:
            raise ValueError(f"[HATA] Zoom seviyesi '{chosen_zoom}' için hiçbir veri bulunamadı!")
        print(f"\n[BİLGİ] Seçilen zoom seviyesi: {chosen_zoom}")
    else:
        df_zoom = df.copy().reset_index(drop=True)
        print("\n[BİLGİ] Zoom filtresi uygulanmadı — tüm veriler kullanılıyor.")

    print(f"Toplam Görüntü: {len(df_zoom)}")
    print(f"Eşsiz Hasta Sayısı: {df_zoom['patient_id'].nunique()}")

    # --------------------------------------------------
    # 2. Birden fazla sınıfı olan hastaları çıkar
    # --------------------------------------------------
    patient_class_counts = df_zoom.groupby("patient_id")["class"].nunique()
    problematic_patients = patient_class_counts[patient_class_counts > 1].index.tolist()

    if len(problematic_patients) > 0:
        removed_images = df_zoom[df_zoom["patient_id"].isin(problematic_patients)].shape[0]
        print(f"\n[UYARI] {len(problematic_patients)} hasta birden fazla sınıfa sahip. Temizleniyor...")
        df_zoom = df_zoom[~df_zoom["patient_id"].isin(problematic_patients)].copy().reset_index(drop=True)
    
    if df_zoom.empty:
        raise ValueError("[HATA] Sorunlu hastalar çıkarıldıktan sonra veri kalmadı.")

    # --------------------------------------------------
    # 3. Hasta bazlı dataframe oluştur
    # --------------------------------------------------
    patient_level_df = df_zoom[["patient_id", "class"]].drop_duplicates().reset_index(drop=True)

    # --------------------------------------------------
    # 4. V3 ÖZEL: Sınıf Bazlı Garantili Stratified Split
    # --------------------------------------------------
    print("\n[BİLGİ] Garantili Katmanlı (Stratified) Bölme işlemi başlatılıyor...")
    train_p, val_p, test_p = [], [], []

    for cls in patient_level_df["class"].unique():
        p_list = patient_level_df[patient_level_df["class"] == cls]["patient_id"].tolist()
        np.random.shuffle(p_list) # Seed ile karıştır
        n = len(p_list)

        if n == 1:
            # Sadece 1 hasta varsa mecburen Train'e gider
            train_p.extend(p_list)
            print(f"  -> UYARI: '{cls}' sınıfında sadece 1 hasta var. Sadece Train setine eklendi.")
        elif n == 2:
            # 2 hasta varsa Train ve Val paylaşır, Test'e kalmaz
            train_p.append(p_list[0])
            val_p.append(p_list[1])
            print(f"  -> UYARI: '{cls}' sınıfında 2 hasta var. Train ve Val setine dağıtıldı (Test=0).")
        elif n == 3:
            # 3 hasta varsa her sete 1'er tane garanti verilir
            train_p.append(p_list[0])
            val_p.append(p_list[1])
            test_p.append(p_list[2])
        else:
            # 4 ve üzeri hasta varsa oranlara göre dağıt
            val_count = max(1, int(n * (temp_size / 2)))
            test_count = max(1, int(n * (temp_size / 2)))
            train_count = n - val_count - test_count

            train_p.extend(p_list[:train_count])
            val_p.extend(p_list[train_count:train_count+val_count])
            test_p.extend(p_list[train_count+val_count:])

    # --------------------------------------------------
    # 5. Final dataframe'leri oluştur
    # --------------------------------------------------
    train_df = df_zoom[df_zoom["patient_id"].isin(train_p)].reset_index(drop=True)
    val_df   = df_zoom[df_zoom["patient_id"].isin(val_p)].reset_index(drop=True)
    test_df  = df_zoom[df_zoom["patient_id"].isin(test_p)].reset_index(drop=True)

    # --------------------------------------------------
    # 6. Sızıntı (Leakage) kontrolü
    # --------------------------------------------------
    train_patients = set(train_df["patient_id"].unique())
    val_patients   = set(val_df["patient_id"].unique())
    test_patients  = set(test_df["patient_id"].unique())

    if (train_patients & val_patients) or (train_patients & test_patients) or (val_patients & test_patients):
        raise ValueError("[KARTAL GÖZÜ HATA] Veri sızıntısı (Leakage) tespit edildi! Çakışan hastalar var.")
    print("\n[BAŞARILI] Veri sızıntısı (Leakage) testi geçildi. Kümeler tamamen izole.")

    # --------------------------------------------------
    # 7. V3 ÖZEL: Statik Label Mapping
    # --------------------------------------------------
    # Veri setinde o an bir sınıf olmasa bile ağın 8 çıkışlı kalmasını sağlar.
    expected_classes = sorted(expected_classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(expected_classes)}
    
    train_df["label"] = train_df["class"].map(class_to_idx)
    val_df["label"]   = val_df["class"].map(class_to_idx)
    test_df["label"]  = test_df["class"].map(class_to_idx)

    # --------------------------------------------------
    # 8. Özet Rapor
    # --------------------------------------------------
    print("\n" + "="*50)
    print("V3 SPLIT SUMMARY (Görüntü Sayıları)")
    print("="*50)
    print(f"{'Split':<12} {'Hastalar':>10} {'Görüntüler':>10}")
    print("-"*35)
    print(f"{'Train':<12} {len(train_patients):>10} {len(train_df):>10}")
    print(f"{'Validation':<12} {len(val_patients):>10} {len(val_df):>10}")
    print(f"{'Test':<12} {len(test_patients):>10} {len(test_df):>10}")
    print("="*50)

    print("\n[BİLGİ] Sınıf - İndeks Eşleşmesi (Statik 8 Sınıf):")
    for cls, idx in class_to_idx.items():
        print(f"  {cls} -> {idx}")

    return train_df, val_df, test_df, class_to_idx