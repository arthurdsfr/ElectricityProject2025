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

# Fill missing price values
df_raw['Price'] = df_raw['Price'].ffill().bfill()

# ----------------------------------------------------
#   1) FULL DATASET  (2012 → 2023 CSV)
# ----------------------------------------------------
df_elec_full = df_raw.copy()
df_elec_full = df_elec_full.set_index('Date')


# ----------------------------------------------------
#   2) TRAINING DATASET  (2012 → 2020-12-31)
# ----------------------------------------------------
start_train = pd.to_datetime("2012-01-02")
end_train   = pd.to_datetime("2020-12-31")

df_elec_train = df_raw[
    (df_raw['Date'] >= start_train) &
    (df_raw['Date'] <= end_train)
].copy()

df_elec_train = df_elec_train.set_index('Date')


# ----------------------------------------------------
#   3) TEST DATASET  (2021-01-01 → 2021-03-01)
#   FIX: reindex to ensure full calendar coverage
# ----------------------------------------------------
start_test = pd.to_datetime("2021-01-01")
end_test   = pd.to_datetime("2021-03-01")

df_elec_test = df_raw[
    (df_raw['Date'] >= start_test) & 
    (df_raw['Date'] <= end_test)
].copy()

df_elec_test = df_elec_test.set_index('Date')

# ---- FIX: Ensure no missing dates (important for GRU 30-day window) ----
full_test_range = pd.date_range(start=start_test, end=end_test, freq="D")
df_elec_test = df_elec_test.reindex(full_test_range)

# Fill missing prices (interpolate new days like Jan 1–3)
df_elec_test['Price'] = df_elec_test['Price'].interpolate().ffill().bfill()


# Optional: Plot
plt.figure(figsize=(14, 6))

plt.plot(df_elec_full.index, df_elec_full['Price'], label="Full Dataset (2012–2023)", alpha=0.3)
plt.plot(df_elec_train.index, df_elec_train['Price'], label="Training Dataset (2012–2020)", linewidth=2)
plt.plot(df_elec_test.index, df_elec_test['Price'], label="Fixed Test Dataset (2021-01-01 → 2021-03-01)", linewidth=2, color='red')

plt.title("Electricity Price: Training vs Test vs Full Dataset")
plt.xlabel("Date")
plt.ylabel("Price (€)")
plt.legend()
plt.grid(True)
plt.show()
