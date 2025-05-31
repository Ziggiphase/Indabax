import pandas as pd
import numpy as np
from itertools import combinations
def FeatureInteraction(df, col):
    new_features = pd.DataFrame()

    for f1, f2 in combinations(col, 2):
        # Multiply features
        new_features[f'{f1}_x_{f2}'] = df[f1] * df[f2]

        # Divide features (skip division by 0)
        new_features[f'{f1}_div_{f2}'] = np.where(df[f2] != 0, df[f1] / df[f2], np.nan)
        new_features[f'{f2}_div_{f1}'] = np.where(df[f1] != 0, df[f2] / df[f1], np.nan)

        # Add features
        new_features[f'{f1}_plus_{f2}'] = df[f1] + df[f2]

        # Subtract features (both directions)
        new_features[f'{f1}_minus_{f2}'] = df[f1] - df[f2]
        new_features[f'{f2}_minus_{f1}'] = df[f2] - df[f1]

    # Optionally, you can add back the original features to the dataframe
    df_with_new_features = pd.concat([df, new_features], axis=1)

    return df_with_new_features

def SND_Feature(df, col):
    new_df = pd.DataFrame()

    for f1, f2 in combinations(col, 2):
        new_df[f"SND_{f1}_{f2}"] = (df[f1] - df[f2])/df[f1] + df[f2]

    df_with_new_features = pd.concat([df, new_df], axis=1)

    return df_with_new_features