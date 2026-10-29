import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


FEATURES = ["Open", "High", "Low", "Close", "Volume"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate an LSTM stock-price predictor."
    )
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker (default: AAPL).")
    parser.add_argument("--start", default="2020-01-01", help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Optional end date, YYYY-MM-DD (exclusive).")
    parser.add_argument("--seq-length", type=int, default=30, help="Window length, including target (default: 30).")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs for each model (default: 200).")
    parser.add_argument("--cv-splits", type=int, default=3, help="Number of chronological CV splits (default: 3).")
    return parser.parse_args()


def download_data(ticker, start, end):
    kwargs = {"tickers": ticker, "start": start, "progress": False, "auto_adjust": False}
    if end:
        kwargs["end"] = end
    df = yf.download(**kwargs)
    if df.empty:
        raise RuntimeError(f"No stock data was downloaded for {ticker}.")

    # yfinance may return MultiIndex columns even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        selected = {}
        for feature in FEATURES:
            matches = [col for col in df.columns if feature in col]
            if not matches:
                raise ValueError(f"Downloaded data does not contain {feature}.")
            selected[feature] = df.loc[:, matches[0]]
        df = pd.DataFrame(selected, index=df.index)
    else:
        missing = [feature for feature in FEATURES if feature not in df.columns]
        if missing:
            raise ValueError(f"Downloaded data is missing columns: {', '.join(missing)}")
        df = df[FEATURES]

    return df.apply(pd.to_numeric, errors="coerce").dropna()


def make_windows(scaled_features, seq_length):
    input_length = seq_length - 1
    sample_count = len(scaled_features) - input_length
    if sample_count <= 0:
        raise ValueError("Not enough rows for the selected sequence length.")

    X = np.stack(
        [scaled_features[i : i + input_length] for i in range(sample_count)]
    ).astype(np.float32)
    # Close is the fourth feature. Each target is the row immediately after its inputs.
    y = scaled_features[input_length:, FEATURES.index("Close")].reshape(-1, 1).astype(np.float32)
    return X, y


def fit_scaler_and_make_data(raw_features, train_target_end, seq_length):
    # Fit only on the chronological training period to avoid leaking validation/test data.
    scaler = StandardScaler()
    scaler.fit(raw_features[: train_target_end + 1])
    scaled = scaler.transform(raw_features).astype(np.float32)
    X, y = make_windows(scaled, seq_length)
    return scaler, X, y


class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, output_dim=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def train_model(X, y, epochs, seed):
    torch.manual_seed(seed)
    model = PredictionModel(input_dim=len(FEATURES)).to(DEVICE)
    X_tensor = torch.from_numpy(X).to(DEVICE)
    y_tensor = torch.from_numpy(y).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(X_tensor)
        loss = criterion(prediction, y_tensor)
        loss.backward()
        optimizer.step()

    return model


def inverse_close(values, scaler):
    close_index = FEATURES.index("Close")
    return values.reshape(-1) * scaler.scale_[close_index] + scaler.mean_[close_index]


def predict_lstm(model, X, scaler):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(DEVICE)
        scaled_prediction = model(X_tensor).cpu().numpy().reshape(-1)
    return inverse_close(scaled_prediction, scaler)


def baseline_prediction(X, scaler):
    # Naive baseline: next Close equals the latest Close in the input window.
    close_index = FEATURES.index("Close")
    return inverse_close(X[:, -1, close_index], scaler)


def score(actual, predicted):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
    }


def print_scores(label, actual, lstm_prediction, baseline):
    lstm_scores = score(actual, lstm_prediction)
    baseline_scores = score(actual, baseline)
    print(
        f"{label:<20} "
        f"LSTM RMSE: {lstm_scores['RMSE']:.4f} | LSTM MAE: {lstm_scores['MAE']:.4f} || "
        f"Baseline RMSE: {baseline_scores['RMSE']:.4f} | Baseline MAE: {baseline_scores['MAE']:.4f}"
    )


def main():
    args = parse_args()
    if args.seq_length < 2:
        raise ValueError("--seq-length must be at least 2.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    df = download_data(args.ticker, args.start, args.end)
    raw_features = df[FEATURES].to_numpy(dtype=np.float64)
    sample_count = len(raw_features) - args.seq_length + 1
    if sample_count < args.cv_splits + 1:
        raise ValueError("Not enough data for the selected sequence length and CV splits.")

    # Expanding-window time-series CV: each validation block follows its training data.
    splitter = TimeSeriesSplit(n_splits=args.cv_splits)
    print(f"Device: {DEVICE}")
    print(f"Ticker: {args.ticker} | rows: {len(df)} | features: {', '.join(FEATURES)}")
    print("\nChronological cross-validation:")
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.arange(sample_count)), start=1):
        train_target_end = int(train_idx[-1] + args.seq_length - 1)
        scaler, X, y = fit_scaler_and_make_data(raw_features, train_target_end, args.seq_length)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        actual_val = inverse_close(y[val_idx, 0], scaler)

        model = train_model(X_train, y_train, args.epochs, seed=42 + fold)
        lstm_val = predict_lstm(model, X_val, scaler)
        baseline_val = baseline_prediction(X_val, scaler)
        print_scores(f"Fold {fold}", actual_val, lstm_val, baseline_val)
        fold_scores.append((score(actual_val, lstm_val), score(actual_val, baseline_val)))

    for model_name, model_index in (("LSTM", 0), ("Baseline", 1)):
        averages = {
            metric: np.mean([fold[model_index][metric] for fold in fold_scores])
            for metric in ("RMSE", "MAE")
        }
        print(
            f"CV average {model_name:<11} "
            f"RMSE: {averages['RMSE']:.4f} | MAE: {averages['MAE']:.4f}"
        )

    # Final chronological 80/20 holdout evaluation and plot.
    train_size = int(0.8 * sample_count)
    if train_size < 1 or train_size >= sample_count:
        raise ValueError("Not enough data for an 80/20 train/test split.")
    train_idx = np.arange(train_size)
    test_idx = np.arange(train_size, sample_count)
    train_target_end = int(train_idx[-1] + args.seq_length - 1)
    scaler, X, y = fit_scaler_and_make_data(raw_features, train_target_end, args.seq_length)

    model = train_model(X[train_idx], y[train_idx], args.epochs, seed=123)
    actual_test = inverse_close(y[test_idx, 0], scaler)
    lstm_test = predict_lstm(model, X[test_idx], scaler)
    baseline_test = baseline_prediction(X[test_idx], scaler)
    print("\nFinal holdout (last 20%):")
    print_scores("Test", actual_test, lstm_test, baseline_test)

    # Target row for window i is i + seq_length - 1.
    target_rows = test_idx + args.seq_length - 1
    dates = df.index[target_rows]
    errors = np.abs(actual_test - lstm_test)
    test_rmse = score(actual_test, lstm_test)["RMSE"]

    fig = plt.figure(figsize=(12, 10))
    grid = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(grid[:3, 0])
    ax1.plot(dates, actual_test, color="blue", label="Actual Close")
    ax1.plot(dates, lstm_test, color="green", label="LSTM Prediction")
    ax1.plot(dates, baseline_test, color="orange", alpha=0.8, label="Naive Baseline")
    ax1.legend()
    ax1.set_title(f"{args.ticker} Stock Price Prediction")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price")

    ax2 = fig.add_subplot(grid[3, 0])
    ax2.axhline(test_rmse, color="blue", linestyle="--", label="LSTM RMSE")
    ax2.plot(dates, errors, color="red", label="LSTM Absolute Error")
    ax2.legend()
    ax2.set_title("Test Prediction Error")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Error")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
