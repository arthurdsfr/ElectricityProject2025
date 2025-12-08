import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from chronos_eval import calculate_mase, calculate_mda 

#configuration graphique
plt.style.use("ggplot")

#preprocessing
df = pd.read_csv("240923 - suivi marché depuis 2012.csv", sep=";", header=1, decimal=",", engine="python", dtype=str)
df = df.iloc[:, :2]
df.columns = ["date", "price"]
df["date"] = pd.to_datetime(df["date"].str.strip().str.replace('"', ''), format="%d/%m/%Y")
df["price"] = df["price"].str.replace(",", ".").astype(float)

#mask guerre ukraine
df = df.set_index("date").sort_index().loc[:"2021-02-24"]

#interpolation des weekends et jours fériés
idx_full = pd.date_range(df.index.min(), df.index.max(), freq="D")
df_full = df.reindex(idx_full)
df_full["price"] = df_full["price"].interpolate(method="linear")

def create_seasonal_profile(df_train):
    return df_train.groupby(df_train.index.dayofyear)["price"].mean()

def get_seasonality(dates, profile):
    return profile.reindex(dates.dayofyear).ffill().values

def run_benchmark_duel(df, horizon=15):
    start_test_date = pd.to_datetime("2021-01-30")
    #train / test
    train_hist = df.loc[:start_test_date - pd.Timedelta(days=1)]
    test_data_full = df.loc[start_test_date:]
    
    seasonal_profile = create_seasonal_profile(train_hist)
    
    df["season"] = get_seasonality(df.index, seasonal_profile)
    df["resid"] = df["price"] - df["season"]
    
    #entraînement sur résidus
    train_resid = df.loc[:start_test_date - pd.Timedelta(days=1), "resid"]
    
    model = auto_arima(
        train_resid,
        seasonal=False,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        d=None,
        suppress_warnings=True
    )

    max_loop_date = test_data_full.index.max() - pd.Timedelta(days=horizon-1)
    dates_to_loop = test_data_full.loc[:max_loop_date].index

    results = {
        "ARIMA":       {"mase": [], "mda": []},
        "Persistance": {"mase": [], "mda": []}
    }
    
    plot_data = [] 
    
    for i, current_date in enumerate(dates_to_loop):
    
        target_dates = pd.date_range(start=current_date, periods=horizon, freq='D')
        y_true = df.loc[target_dates, "price"].values
        
        history_series = df.loc[:current_date - pd.Timedelta(days=1), "price"]
        y_hist_full = history_series.values
        last_known_val = y_hist_full[-1]
        
        pred_resid = model.predict(n_periods=horizon, return_conf_int=False)
        if hasattr(pred_resid, 'values'): pred_resid = pred_resid.values
        future_season = get_seasonality(target_dates, seasonal_profile)
        pred_arima = pred_resid + future_season #ajout de la saisonnalité
        
        pred_naive = np.full(horizon, last_known_val) #ajout du modèle persistance (15j constant)
        
        df_print = pd.DataFrame({
            "Date": target_dates,
            "Réel": y_true,
            "ARIMA": pred_arima,
            "Persistance": pred_naive
        })

        print(df_print.to_string(index=False, float_format="%.2f")) 

        #évaluation
        models_preds = {
            "ARIMA": pred_arima,
            "Persistance": pred_naive
        }
        
        for name, preds in models_preds.items():
            mase = calculate_mase(y_true, preds, y_hist_full, seasonality=1)
            mda = calculate_mda(y_true, preds, last_known_val)
            
            results[name]["mase"].append(mase)
            results[name]["mda"].append(mda)
            
        plot_data.append({
            "dates": target_dates,
            "arima_preds": pred_arima
        })

        true_resid_today = df.loc[current_date, "price"] - df.loc[current_date, "season"]
        model.update([true_resid_today])

    return results, plot_data

if __name__ == "__main__":
    horizon = 15
    res, plot_data = run_benchmark_duel(df_full, horizon=horizon)
    print(f" Résultats (avec modèle persistance) (Horizon {horizon} jours)")
    print(f"{'Modèle':<20} | {'MASE Moyen':<15} | {'MDA Moyen':<15}")

    summary = {}
    for model in res:
        summary[model] = {
            "mase": np.mean(res[model]["mase"]),
            "mda": np.mean(res[model]["mda"])
        }
        print(f"{model:<20} | {summary[model]['mase']:.4f}          | {summary[model]['mda']:.2%}")


    plt.figure(figsize=(12, 6))
    plot_start = "2021-01-20"
    x_real = df_full.loc[plot_start:].index.values
    y_real = df_full.loc[plot_start:]["price"].values

    plt.plot(x_real, y_real, label="Réalité", color="black", linewidth=2.5, zorder=10)

    for i, data in enumerate(plot_data):
        dates = data["dates"]
        preds = data["arima_preds"]
        lbl = "Prédictions ARIMA (15j)" if i == 0 else "_nolegend_" #pour légende unique
        plt.plot(dates, preds, color="red", alpha=0.5, linewidth=1, label=lbl)

    plt.title(f"Visualisation des fenêtres de prédiction ARIMA ({horizon} jours)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()