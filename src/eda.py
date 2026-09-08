import pandas as pd


def load_data(train_path, test_path):

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


def basic_eda(train, test):

    print("TRAIN SHAPE:", train.shape)
    print("TEST SHAPE:", test.shape)

    print("\nTRAIN COLUMNS:")
    print(train.columns.tolist())

    print("\nTEST COLUMNS:")
    print(test.columns.tolist())

    print("\nTRAIN MISSING VALUES:")
    print(train.isnull().sum())

    print("\nTEST MISSING VALUES:")
    print(test.isnull().sum())

    print("\nMODALITY:")
    print(train["modality"].value_counts())

    print("\nBODY PART:")
    print(train["body_part"].value_counts())

    print("\nMODALITY × BODY PART:")
    print(
        pd.crosstab(
            train["modality"],
            train["body_part"]
        )
    )


if __name__ == "__main__":

    train, test = load_data(
        "data/train.csv",
        "data/test.csv"
    )

    basic_eda(train, test)