import pandas as pd
import matplotlib.pyplot as plt

# --- Load electricity price data ---
file_path = 'data/240923 - suivi marché depuis 2012.csv'

# Skip the duplicate header row and read only the first two columns
df_elec = pd.read_csv(file_path, delimiter=',', skiprows=1, usecols=[0, 1])

# Rename columns to standard names
df_elec.columns = ['Date', 'Price']

# Convert Date to datetime
df_elec['Date'] = pd.to_datetime(df_elec['Date'], format='%d/%m/%Y', errors='coerce')

# Drop any rows with invalid dates
df_elec = df_elec.dropna(subset=['Date'])

# Convert Price to float (replace comma with dot)
df_elec['Price'] = df_elec['Price'].astype(str).str.replace(',', '.').astype(float)

# Filter by date range
start_date = pd.to_datetime("2012-01-02")
end_date = pd.to_datetime("2021-03-01")
df_elec = df_elec[(df_elec['Date'] >= start_date) & (df_elec['Date'] <= end_date)]

# Optional: set Date as index
df_elec = df_elec.set_index('Date')

df_elec['Price'].plot(title="Electricity Price Over Time", figsize=(12,6))
plt.xlabel("Date")
plt.ylabel("Price (€)")
plt.show()
