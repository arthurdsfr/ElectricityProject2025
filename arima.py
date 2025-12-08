import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from chronos_eval import calculate_mase, calculate_mda 

# Configuration graphique
plt.style.use("ggplot")

# ==========================================
# 1. CHARGEMENT ET NETTOYAGE
# ==========================================
print("--- Chargement des données ---")
df = pd.read_csv("240923 - suivi marché depuis 2012.csv", sep=";", header=1, decimal=",", engine="python", dtype=str)
df = df.iloc[:, :2]
df.columns = ["date", "price"]
df["date"] = pd.to_datetime(df["date"].str.strip().str.replace('"', ''), format="%d/%m/%Y")
df["price"] = df["price"].str.replace(",", ".").astype(float)

# Filtre temporel
df = df.set_index("date").sort_index().loc[:"2021-02-24"]

# Interpolation
idx_full = pd.date_range(df.index.min(), df.index.max(), freq="D")
df_full = df.reindex(idx_full)
df_full["price"] = df_full["price"].interpolate(method="linear")

print(f"Données prêtes : {len(df_full)} jours.")

# ==========================================
# 2. FONCTIONS UTILES
# ==========================================

def create_seasonal_profile(df_train):
    """Calcule le profil saisonnier moyen."""
    return df_train.groupby(df_train.index.dayofyear)["price"].mean()

def get_seasonality(dates, profile):
    """Récupère la valeur du profil pour une liste de dates (Correction ffill incluse)."""
    return profile.reindex(dates.dayofyear).ffill().values

# ==========================================
# 3. PIPELINE DUEL (ARIMA vs NAÏF SIMPLE)
# ==========================================

def run_benchmark_duel(df, horizon=15):
    # Configuration
    start_test_date = pd.to_datetime("2021-01-30")
    
    # Train / Test
    train_hist = df.loc[:start_test_date - pd.Timedelta(days=1)]
    test_data_full = df.loc[start_test_date:]
    
    print(f"\n--- Initialisation ---")
    print(f"Horizon : {horizon} jours")
    
    # --- PRÉPARATION ARIMA ---
    print("1. Calcul du profil saisonnier historique...")
    seasonal_profile = create_seasonal_profile(train_hist)
    
    df["season"] = get_seasonality(df.index, seasonal_profile)
    df["resid"] = df["price"] - df["season"]
    
    # Entraînement sur résidus (Train seulement)
    train_resid = df.loc[:start_test_date - pd.Timedelta(days=1), "resid"]
    
    print("2. Entraînement ARIMA sur résidus...")
    model = auto_arima(
        train_resid,
        seasonal=False,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        d=None,
        suppress_warnings=True
    )
    print(f"   -> Modèle : ARIMA{model.order}")

    # --- BOUCLE DE TEST ---
    max_loop_date = test_data_full.index.max() - pd.Timedelta(days=horizon-1)
    dates_to_loop = test_data_full.loc[:max_loop_date].index
    
    # Stockage
    results = {
        "ARIMA":       {"mase": [], "mda": []},
        "Naif_Simple": {"mase": [], "mda": []}
    }
    
    # Pour le graphique final
    plot_data = [] 
    
    print(f"\nLancement du duel sur {len(dates_to_loop)} fenêtres...")
    
    for i, current_date in enumerate(dates_to_loop):
        # A. Cibles et Historique
        target_dates = pd.date_range(start=current_date, periods=horizon, freq='D')
        y_true = df.loc[target_dates, "price"].values
        
        # Historique connu à l'instant t
        history_series = df.loc[:current_date - pd.Timedelta(days=1), "price"]
        y_hist_full = history_series.values
        last_known_val = y_hist_full[-1]

        # B. PRÉDICTIONS
        
        # 1. ARIMA (Résidu + Saisonnalité)
        pred_resid = model.predict(n_periods=horizon, return_conf_int=False)
        if hasattr(pred_resid, 'values'): pred_resid = pred_resid.values
        future_season = get_seasonality(target_dates, seasonal_profile)
        pred_arima = pred_resid + future_season
        
        # 2. Naïf Simple (Persistence)
        pred_naive = np.full(horizon, last_known_val)
        
        # --- C. AFFICHAGE DES 15 VALEURS (Demande Utilisateur) ---
        print(f"\n--- Fenêtre {i+1} : Départ le {current_date.date()} ---")
        # On crée un petit DataFrame temporaire pour afficher proprement
        df_print = pd.DataFrame({
            "Date": target_dates,
            "Réel": y_true,
            "ARIMA": pred_arima,
            "Naïf": pred_naive
        })
        # Affichage formaté (arrondi à 2 décimales)
        print(df_print.to_string(index=False, float_format="%.2f"))

        # D. SCORING
        models_preds = {
            "ARIMA": pred_arima,
            "Naif_Simple": pred_naive
        }
        
        for name, preds in models_preds.items():
            mase = calculate_mase(y_true, preds, y_hist_full, seasonality=1)
            mda = calculate_mda(y_true, preds, last_known_val)
            
            results[name]["mase"].append(mase)
            results[name]["mda"].append(mda)
            
        # Stockage pour le graphique
        plot_data.append({
            "dates": target_dates,
            "arima_preds": pred_arima
        })

        # E. UPDATE MODELE
        true_resid_today = df.loc[current_date, "price"] - df.loc[current_date, "season"]
        model.update([true_resid_today])

    return results, plot_data

# ==========================================
# 4. RÉSULTATS FINAUX & PLOT
# ==========================================
if __name__ == "__main__":
    horizon = 15
    res, plot_data = run_benchmark_duel(df_full, horizon=horizon)

    # --- TABLEAU RÉCAPITULATIF ---
    print("\n" + "="*60)
    print(f" RÉSULTATS DUEL (Horizon {horizon} jours)")
    print("="*60)
    print(f"{'MODÈLE':<20} | {'MASE Moyen':<15} | {'MDA Moyen':<15}")
    print("-" * 60)

    summary = {}
    for model in res:
        summary[model] = {
            "mase": np.mean(res[model]["mase"]),
            "mda": np.mean(res[model]["mda"])
        }
        print(f"{model:<20} | {summary[model]['mase']:.4f}          | {summary[model]['mda']:.2%}")

    print("-" * 60)
    
    # Verdict
    if summary["ARIMA"]["mase"] < summary["Naif_Simple"]["mase"]:
        print("✅ VICTOIRE : Ton modèle bat le Naïf Simple !")
    else:
        print("❌ DÉFAITE : Le modèle Naïf Simple est meilleur.")

    # --- PLOTTING (Spaghetti Plot) ---
    plt.figure(figsize=(12, 6))

    # 1. Tracer la Réalité
    # On prend une période assez large pour voir le contexte
    plot_start = "2021-01-20"
    x_real = df_full.loc[plot_start:].index.values
    y_real = df_full.loc[plot_start:]["price"].values

    plt.plot(x_real, y_real, label="Réalité", color="black", linewidth=2.5, zorder=10)

    # 2. Tracer les fenêtres ARIMA
    print(f"\nGénération du graphique avec {len(plot_data)} fenêtres...")
    for i, data in enumerate(plot_data):
        dates = data["dates"]
        preds = data["arima_preds"]
        
        # Astuce légende : on ne l'affiche qu'une fois
        lbl = "Prédictions ARIMA (15j)" if i == 0 else "_nolegend_"
        plt.plot(dates, preds, color="#E24A33", alpha=0.5, linewidth=1, label=lbl)

    plt.title(f"Visualisation des fenêtres de prédiction ARIMA ({horizon} jours)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()