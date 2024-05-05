import os
import csv
import logging 
import pandas as pd
from typing import List

def save_df_as_csv(df: pd.DataFrame, fpath: str) -> str:
    """safer version of saving pandas dataframe as local csv
    Arguments:
        df (pd.DataFrame): Dataframe to be saved
        fpath (str): Destination file path
    Returns:
        str: Destination file path
    """
    fpath_abs = os.path.abspath(fpath)
    with open(fpath_abs, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(df.columns)
        writer.writerows(df.values.tolist())
    logging.info(f"Save dataframe to file: {fpath_abs}")
    return fpath_abs

def stratified_split(df: pd.DataFrame, splitby: List[str], frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified splitting based on multiple columns.
    
    Arguments:
        df (pd.DataFrame): Dataframe to be processed
        groupby (List[str]): Columns to groupby for straitification
        frac (float): ratio of the first stratified sample
    Returns:
        pd.DataFrame: sampled split (df * frac)
        pd.DataFrame: remained split  df - (df * frac)
    """
    df1 = df.groupby(splitby, dropna=True).sample(frac=frac)
    df2 = df.drop(df1.index).reset_index(drop=True)
    return df1.reset_index(drop=True), df2