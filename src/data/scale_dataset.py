import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.preprocessing import StandardScaler


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=False), required=0)
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # input_filepath = click.prompt(
    #     'Enter the file path for the extracted training data',
    #     type=click.Path(exists=True))
    input_filepath = "data/processed"
    scale_data(input_filepath)


def scale_data(input_filepath):
    X_train = pd.read_csv(f"{input_filepath}/X_train.csv")
    X_test = pd.read_csv(f"{input_filepath}/X_test.csv")
    # Select only numeric columns for scaling
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    non_numeric_cols = X_train.columns.difference(numeric_cols)
    scaler = StandardScaler()
    X_train_scaled_numeric = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled_numeric = scaler.transform(X_test[numeric_cols])
    # Convert back to DataFrame with original columns
    X_train_scaled = pd.DataFrame(X_train_scaled_numeric,
                                  columns=numeric_cols,
                                  index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_numeric,
                                 columns=numeric_cols,
                                 index=X_test.index)
    # Concatenate non-numeric columns
    X_train_final = pd.concat([X_train_scaled, X_train[non_numeric_cols]],
                              axis=1)
    X_test_final = pd.concat([X_test_scaled, X_test[non_numeric_cols]], axis=1)
    # Save scaled datasets
    X_train_final.to_csv(f"{input_filepath}/X_train_scaled.csv", index=False)
    X_test_final.to_csv(f"{input_filepath}/X_test_scaled.csv", index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
