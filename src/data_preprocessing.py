import pandas as pd

from sklearn.preprocessing import StandardScaler

def load_data():

    df = pd.read_csv(
    "../data/CC GENERAL.csv"
)

    return df


def preprocess_data(df):

    df.fillna(
        df.median(numeric_only=True),
        inplace=True
    )

    X = df.drop(
        columns=["CUST_ID"]
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler