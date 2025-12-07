import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline, Chronos2Pipeline
import os
from chronos_eval import calculate_mase, calculate_mda
import torch
import warnings

#ignorer spécifiquement les warnings contenant "pin_memory"
warnings.filterwarnings("ignore", message=".*pin_memory.*")

#import data
raw_data = pd.read_csv("240923 - suivi marché depuis 2012.csv", sep=";")

#prepare data
data = raw_data.iloc[:, :2] #on garde seulement les 2 premières colonnes (date et prix pour année suivante en base)
data = data.rename(columns={"Date ": "timestamp"," BASELOAD CALENDARS ": "target"}) #renommer les colonnes pour le modèle
data = data.drop([0]) #supprimer la 1ère ligne (titres)
data["target"] = data["target"].str.replace(",", ".", regex=False).astype(float)  #remplacer les virgules par des points et convertir en float
data["timestamp"] = pd.to_datetime(data["timestamp"], format="%d/%m/%Y") #convertir le timestamp au bon format
data = (data.set_index("timestamp").resample("D").interpolate().reset_index()) #resample en journalier et interpolation des valeurs manquantes (week-ends)
data["item_id"] = "FRA" #colonne nécessaire au modèle

#Modèle Chronos
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use only 1 GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Load the Chronos-2 pipeline
# pip install 'chronos-forecasting>=2.0' 'pandas[pyarrow]' 'matplotlib'
# GPU recommended for faster inference, but CPU is also supported
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device)

#création du jeu de test pour évaluation des performances en zero-shot
start_date = pd.to_datetime("2021-01-01") #date à partir de laquelle on utilise les données
end_date = pd.to_datetime("2021-03-01") #date à laquelle on arrête la prédiction
nb_jours_context = 30 #nombre de jours dans le contexte (historique)
nb_jours_pred = 14 #nombre de jours à prédire
test_df = data[(data["timestamp"] >= start_date)].copy()
test_df = test_df[test_df["timestamp"] <= end_date].copy()
metrics = {"mase": [], "mda": []}

#évaluation en zero-shot sur des fenêtres glissantes
for day in range(nb_jours_context + 1, len(test_df) - nb_jours_pred + 1):
    context_df = test_df.iloc[day - nb_jours_context: day]
    expected_df = test_df.iloc[day: day + nb_jours_pred]

    #Prédiction Chronos
    predicted_df = pipeline.predict_df(
        context_df,
        prediction_length=nb_jours_pred,
        quantile_levels=[0.1, 0.5, 0.9]
    )

    #évaluation des performances
    y_true = expected_df["target"].values
    y_pred = predicted_df["predictions"].values
    y_hist = context_df["target"].values

    mase = calculate_mase(y_true, y_pred, y_hist, seasonality=1)
    metrics["mase"].append(mase)

    last_history_value = y_hist[-1]
    mda = calculate_mda(y_true, y_pred, last_history_value)
    metrics["mda"].append(mda)
    

#résultats finaux
print(f"\nZero-Shot Evaluation over {len(metrics['mase'])} tests:")
print(f"Average MASE: {np.mean(metrics['mase']):.4f}")
print(f"Average MDA: {np.mean(metrics['mda']):.4f}")

"""Zero-Shot Evaluation over 16 tests:
Average MASE: 2.5785
Average MDA: 0.4777"""