import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline, ChronosPipeline
import torch
from pathlib import Path
import os

#import data
raw_data = pd.read_csv("240923 - suivi marché depuis 2012.csv", sep=";")

#Modèle Chronos Fine-Tuned
OUTPUT_DIR = "./output/finetuned_model"
# On cherche automatiquement le dernier dossier "run" et le dernier "checkpoint"
# car ton entraînement a créé "run-2/checkpoint-1000"
runs = sorted(Path(OUTPUT_DIR).glob("run-*"), key=os.path.getmtime)
if not runs:
    raise FileNotFoundError(f"Aucun dossier 'run' trouvé dans {OUTPUT_DIR}")
last_run = runs[-1]
checkpoints = sorted(last_run.glob("checkpoint-*"), key=os.path.getmtime)
if not checkpoints:
    raise FileNotFoundError(f"Aucun checkpoint trouvé dans {last_run}")
MODEL_PATH = checkpoints[-1] # Le dernier checkpoint (ex: checkpoint-1000)

#Charger le modèle fine-tuné
pipeline = ChronosPipeline.from_pretrained(
    MODEL_PATH,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)
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

#prédiction Chronos fine-tuned
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

#bandes d'incertitude
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

