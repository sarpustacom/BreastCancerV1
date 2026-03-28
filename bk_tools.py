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
