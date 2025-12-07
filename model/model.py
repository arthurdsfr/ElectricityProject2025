import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class GRUForecaster:
    def __init__(self, seq_len=30, horizon=15, units=64):
        self.seq_len = seq_len
        self.horizon = horizon
        self.units = units
        self.scaler = MinMaxScaler()
        self.model = None
        self.X_test = None
        self.y_test = None

    # -------------------------------------------------
    # WINDOW CREATION
    # -------------------------------------------------
    def create_windows(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_len - self.horizon):
            X.append(data[i : i + self.seq_len])
            y.append(data[i + self.seq_len : i + self.seq_len + self.horizon])
        X = np.array(X).reshape((-1, self.seq_len, 1))
        y = np.array(y)
        return X, y

    # -------------------------------------------------
    # FIT FINAL MODEL ON ALL DATA (no split)
    # -------------------------------------------------
    def fit(self, price_series, epochs=60, batch_size=32):
        """
        For final training on the ENTIRE dataset after cross-validation.
        No validation split here.
        """
        data = price_series.reshape(-1, 1)
        scaled = self.scaler.fit_transform(data)

        X, y = self.create_windows(scaled)

        self.model = Sequential([
            Input(shape=(self.seq_len, 1)),
            GRU(self.units),
            Dropout(0.2),            
            Dense(32, activation='relu'),
            Dense(self.horizon)
        ])
        self.model.compile(optimizer="adam", loss="mse")

        es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )

    # -------------------------------------------------
    # FIT FOR ONE CV FOLD
    # -------------------------------------------------
    def fit_on_split(self, train_series, test_series, epochs=60, batch_size=32):
        """
        Used by K-fold cross validation.
        Scaling is fit only on train, applied to test.
        """
        train_scaled = self.scaler.fit_transform(train_series.reshape(-1, 1))
        test_scaled = self.scaler.transform(test_series.reshape(-1, 1))

        X_train, y_train = self.create_windows(train_scaled)
        X_test,  y_test  = self.create_windows(test_scaled)

        self.X_test = X_test
        self.y_test = y_test

        self.model = Sequential([
            Input(shape=(self.seq_len, 1)),
            GRU(self.units),
            Dropout(0.2),           
            Dense(32, activation='relu'),
            Dense(self.horizon)
        ])

        self.model.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

        batch_size = min(32, len(X_train), len(X_test))

        self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )

    # -------------------------------------------------
    # PREDICT TEST SET (for K-fold evaluation)
    # -------------------------------------------------
    def predict_test(self):
        preds = self.model.predict(self.X_test)

        # inverse transform each step independently
        preds_inv = np.array([
            self.scaler.inverse_transform(p.reshape(-1,1)).flatten()
            for p in preds
        ])

        y_true_inv = np.array([
            self.scaler.inverse_transform(t.reshape(-1,1)).flatten()
            for t in self.y_test
        ])

        return preds_inv, y_true_inv

    # -------------------------------------------------
    # EVALUATION METRICS
    # -------------------------------------------------
    def evaluate(self):
        preds, true = self.predict_test()
        mae = np.mean(np.abs(preds - true))
        rmse = np.sqrt(np.mean((preds - true)**2))
        return mae, rmse

    # -------------------------------------------------
    # PREDICT NEXT FUTURE HORIZON
    # -------------------------------------------------
    def predict_future(self, last_prices):
        seq = last_prices.reshape(-1, 1)
        seq_scaled = self.scaler.transform(seq)

        X_input = seq_scaled.reshape((1, self.seq_len, 1))
        pred_scaled = self.model.predict(X_input)

        pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return pred

    # -------------------------------------------------
    # PLOT EXAMPLE FORECAST FROM TEST SET
    # -------------------------------------------------
    def plot_example_forecast(self, index=0):
        preds, true = self.predict_test()

        plt.figure(figsize=(9, 4))
        plt.plot(true[index], label="True")
        plt.plot(preds[index], label="Predicted")
        plt.title(f"Forecast Example — Window #{index}")
        plt.legend()
        plt.show()

    # -------------------------------------------------
    # SAVE / LOAD MODEL + SCALER
    # -------------------------------------------------
    def save(self, path):
        self.model.save(path)
        np.save(path + "_scaler_scale.npy", self.scaler.scale_)
        np.save(path + "_scaler_min.npy", self.scaler.min_)

    def load(self, path):
        self.model = load_model(path)
        scale = np.load(path + "_scaler_scale.npy")
        min_ = np.load(path + "_scaler_min.npy")
        self.scaler.scale_ = scale
        self.scaler.min_ = min_
