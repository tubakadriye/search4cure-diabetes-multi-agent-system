import numpy as np
import pandas as pd


def combine_all_attributes(df, exclude_columns=None):
    """
    Combine all attributes (optionally excluding some) of a DataFrame row into a single column.

    Parameters:
    - df: pandas DataFrame
    - exclude_columns: list of column names to exclude from the combination

    Returns:
    - df: DataFrame with a new 'combined_info' column
    """
    exclude_columns = exclude_columns or []

    def combine_row(row):
        combined = []
        for attr in row.index:
            if attr in exclude_columns:
                continue
            value = row[attr]
            if isinstance(value, (pd.Series, np.ndarray, list)):
                # Handle array-like objects
                if len(value) > 0 and not pd.isna(value).all():
                    combined.append(f"{attr.capitalize()}: {value!s}")
            elif not pd.isna(value):
                combined.append(f"{attr.capitalize()}: {value!s}")
        return " ".join(combined)

    df["combined_info"] = df.apply(combine_row, axis=1)
    return df