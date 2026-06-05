from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from src.data_preprocessing import (
    load_data,
    preprocess_data
)

df = load_data()

X_scaled, scaler = preprocess_data(df)

model = DBSCAN(
    eps=1.5,
    min_samples=10
)

clusters = model.fit_predict(
    X_scaled
)

if len(set(clusters)) > 1:

    score = silhouette_score(
        X_scaled,
        clusters
    )

    print(
        f"Silhouette Score : {score:.3f}"
    )

print(
    f"Noise Points : {sum(clusters==-1)}"
)