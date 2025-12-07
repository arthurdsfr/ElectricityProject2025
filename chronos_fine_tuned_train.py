import pandas as pd
import numpy as np
import yaml
from gluonts.dataset.arrow import ArrowWriter
from pathlib import Path
import os

#configuration
DATA_PATH = "240923 - suivi marché depuis 2012.csv"
OUTPUT_DIR = "./chronos_finetuning_data"
#date de coupure : tout ce qui est avant sert à entraîner
TRAIN_CUTOFF_DATE = "2020-12-31" 

#paramètres du modèle
CONTEXT_LENGTH = 30  # Historique vu par le modèle
PREDICTION_LENGTH = 15 # Horizon de prévision

#création du dossier de sortie
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

#data preparation
print("Chargement et nettoyage des données...")
raw_data = pd.read_csv(DATA_PATH, sep=";")

data = raw_data.iloc[:, :2] #on garde seulement les 2 premières colonnes (date et prix pour année suivante en base)
data = data.rename(columns={"Date ": "timestamp"," BASELOAD CALENDARS ": "target"}) #renommer les colonnes pour le modèle
data = data.drop([0]) #supprimer la 1ère ligne (titres)
data["target"] = data["target"].str.replace(",", ".", regex=False).astype(float)  #remplacer les virgules par des points et convertir en float
data["timestamp"] = pd.to_datetime(data["timestamp"], format="%d/%m/%Y") #convertir le timestamp au bon format
data = (data.set_index("timestamp").resample("D").interpolate().reset_index()) #resample en journalier et interpolation des valeurs manquantes (week-ends)
data["item_id"] = "FRA" #colonne nécessaire au modèle

#création du fichier de données d'entraînement
#on garde que les données jusqu'à la date de coupure pour l'entraînement
df_train = data[data["timestamp"] <= TRAIN_CUTOFF_DATE].copy()

print(f"Données totales : {len(data)} jours")
print(f"Données d'entraînement (jusqu'au {TRAIN_CUTOFF_DATE}) : {len(df_train)} jours")

#conversion au format GluonTS attendu par le modèle
training_data = [
    {
        "start": pd.Timestamp(df_train["timestamp"].iloc[0]),
        "target": df_train["target"].values.astype(np.float32) # Important: float32
    }
]

#sauvegarde en fichier .arrow
arrow_path = os.path.join(OUTPUT_DIR, "train.arrow")
ArrowWriter(compression="lz4").write_to_file(
    training_data,
    path=arrow_path
)
print(f"Fichier de données généré : {arrow_path}")

#création du fichier de configuration YAML attendu par le modèle
config_path = os.path.join(OUTPUT_DIR, "config.yaml")

training_config = {
    #chemin vers le fichier arrow
    "training_data_paths": [arrow_path],
    #probabilité d'utiliser ce fichier (1.0 = 100% car c'est le seul)
    "probability": [1.0],
    #paramètres de dimension
    "context_length": CONTEXT_LENGTH,
    "prediction_length": PREDICTION_LENGTH,
    "min_past": CONTEXT_LENGTH, #il faut au moins 30 jours pour commencer à apprendre
    #paramètres d'optimisation
    "max_steps": 1000,
    "learning_rate": 0.001,   #vitesse d'apprentissage (faible pour pas casser le modèle pré-entraîné)
    "shuffle_buffer_length": 1000,
    "model_id": "amazon/chronos-t5-small", #modèle pré-entraîné à fine-tuner
    "random_init": False,     #False = Fine-Tuning (on garde les connaissances)
    "dataloader_num_workers": 0, 
    "torch_compile": False
}

with open(config_path, "w") as f:
    yaml.dump(training_config, f, sort_keys=False)

print(f"Fichier de configuration généré : {config_path}")

print("Prêt pour le fine-tuning du modèle")
print("Copier et exécuter la commande suivante :")
print(f"\npython chronos-forecasting/scripts/training/train.py --config {config_path} --output-dir ./output/finetuned_model")