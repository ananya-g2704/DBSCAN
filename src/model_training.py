from sklearn.cluster import DBSCAN

from data_preprocessing import (
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

print("Clusters Found")

print(set(clusters))

noise_points = sum(
    clusters == -1
)

print(
    f"Noise Points : {noise_points}"
)