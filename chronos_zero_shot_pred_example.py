import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline, Chronos2Pipeline
import os

#import data
raw_data = pd.read_csv("240923 - suivi marché depuis 2012.csv", sep=";")

# Use only 1 GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the Chronos-2 pipeline
# pip install 'chronos-forecasting>=2.0' 'pandas[pyarrow]' 'matplotlib'
# GPU recommended for faster inference, but CPU is also supported
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="auto")

# prepare data
data = raw_data.iloc[:, :2]
data = data.rename(columns={
    "Date ": "timestamp",
    " BASELOAD CALENDARS ": "target"
})
data = data.drop([0])
data["target"] = data["target"].str.replace(",", ".", regex=False).astype(float)

# Convertir le timestamp au bon format
data["timestamp"] = pd.to_datetime(data["timestamp"], format="%d/%m/%Y")

# Resample en journalier et interpolation
data = (
    data.set_index("timestamp")
        .resample("D")
        .interpolate()
        .reset_index()
)

data["item_id"] = "FRA" # Colonne nécessaire au modèle

# Séparation en context (historique de 30 jours) et expected (période de 15 jours à prédire)
start_date = pd.to_datetime("2021-01-01") # Date à partir de laquelle on utilise les données
cut_date = pd.to_datetime("2021-02-01") # Date à laquelle on va commencer la prédiction
#end_date = data["timestamp"].max()
end_date = pd.to_datetime("2021-02-15") # Date à laquelle on arrête la prédiction
context_df = data[data["timestamp"] < cut_date].copy()
context_df = context_df.tail(30) # Garder seulement les 30 derniers jours d'historique
expected_df = data[data["timestamp"] >= cut_date].copy()
expected_df = expected_df[expected_df["timestamp"] <= end_date].copy()
nb_jours = (end_date - cut_date).days + 1 # Nombre de jours prédits
#expected_df = expected_df.head(nb_jours)

# Prédiction Chronos
predicted_df = pipeline.predict_df(
    context_df,
    prediction_length=nb_jours,
    quantile_levels=[0.1, 0.5, 0.9]
)


##visualisation des résultats
plt.figure(figsize=(12, 5))

# 30 jours de contexte
plt.plot(context_df["timestamp"], context_df["target"], label="Context", linewidth=2)
# 15 jours attendus
plt.plot(expected_df["timestamp"], expected_df["target"], label="Expected", linewidth=2)
# 15 jours prédits
plt.plot(predicted_df["timestamp"], predicted_df["predictions"], label="Predicted", linewidth=2)

# Bandes d'incertitude
plt.fill_between(
    predicted_df["timestamp"],
    predicted_df["0.1"],
    predicted_df["0.9"],
    alpha=0.3,
    label="Prediction interval (10-90%)"
)

plt.legend()
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Expected vs Predicted")
plt.grid(True)
plt.show()

