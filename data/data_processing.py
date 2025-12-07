import pandas as pd
import matplotlib.pyplot as plt

# --- Load electricity price data ---
file_path = 'data/240923 - suivi marché depuis 2012.csv'

# Read CSV (skip duplicate header row, take first 2 columns)
df_raw = pd.read_csv(file_path, delimiter=',', skiprows=1, usecols=[0, 1])
df_raw.columns = ['Date', 'Price']

# Convert Date column
df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%d/%m/%Y', errors='coerce')
df_raw = df_raw.dropna(subset=['Date'])

# Convert Price to float
df_raw['Price'] = df_raw['Price'].astype(str).str.replace(',', '.').astype(float)

# Sort by date just to be safe
df_raw = df_raw.sort_values('Date')

# Fill missing values
df_raw['Price'] = df_raw['Price'].ffill().bfill()

# ----------------------------------------------------
#   1) FULL DATASET  (2012 → 2023 CSV)
# ----------------------------------------------------
df_elec_full = df_raw.copy()
df_elec_full = df_elec_full.set_index('Date')


# ----------------------------------------------------
#   2) TRAINING DATASET  (2012 → 2021-03-01)
# ----------------------------------------------------
start_train = pd.to_datetime("2012-01-02")
end_train   = pd.to_datetime("2021-03-01")

df_elec_train = df_raw[
    (df_raw['Date'] >= start_train) &
    (df_raw['Date'] <= end_train)
].copy()

df_elec_train = df_elec_train.set_index('Date')



plt.figure(figsize=(14, 6))

# Plot full dataset
plt.plot(df_elec_full.index, df_elec_full['Price'], label="Full Dataset (2012–2023)", alpha=0.6)

# Plot training dataset
plt.plot(df_elec_train.index, df_elec_train['Price'], label="Training Dataset (2012–2021-03-01)", linewidth=2)

plt.title("Electricity Price: Training vs Full Dataset")
plt.xlabel("Date")
plt.ylabel("Price (€)")
plt.legend()
plt.grid(True)
plt.show()


