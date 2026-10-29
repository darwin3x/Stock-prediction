# Stock Price Prediction with LSTM

A Python script that downloads historical stock data and trains a PyTorch LSTM to predict the next closing price. The script uses Open, High, Low, Close, and Volume data, compares the LSTM with a simple persistence baseline, and evaluates both with chronological cross-validation and a final holdout set.

> **Note**
> This project is for learning and experimentation. It uses historical prices and does not provide financial advice or guarantee future performance.

## Features

- **Configurable market data** — set the ticker and date range from the command line.
- **OHLCV input features** — uses Open, High, Low, Close, and Volume to predict Close.
- **LSTM and baseline comparison** — compares the LSTM with the baseline “next Close equals the latest observed Close.”
- **Time-aware validation** — reports RMSE and MAE for expanding chronological folds and a final 80/20 holdout.
- **Plots** — displays actual prices, LSTM predictions, baseline predictions, and LSTM absolute errors.
- **Automatic device selection** — uses CUDA when available and otherwise runs on the CPU.

## Tech stack

| Area | Technology |
|---|---|
| Language | Python 3 |
| Data download | `yfinance` |
| Data processing and metrics | `numpy`, `pandas`, `scikit-learn` |
| Model and training | `torch` (PyTorch) |
| Plotting | `matplotlib` |

## Installation

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
python -m pip install numpy pandas yfinance matplotlib torch scikit-learn
```

The script downloads data from Yahoo Finance at runtime, so an internet connection is required.

## Usage

Run with the defaults (AAPL, starting January 1, 2020):

```bash
python main.py
```

Choose a ticker and date range:

```bash
python main.py --ticker MSFT --start 2018-01-01 --end 2025-01-01
```

`--end` is optional and follows Yahoo Finance's exclusive end-date convention. Use `--help` to see all options:

```bash
python main.py --help
```

## Options

| Option | Default | Description |
|---|---|---|
| `--ticker` | `AAPL` | Stock ticker symbol |
| `--start` | `2020-01-01` | First date to download, in `YYYY-MM-DD` format |
| `--end` | Latest available | Optional exclusive end date, in `YYYY-MM-DD` format |
| `--seq-length` | `30` | Window length, including the target; 29 rows are model inputs |
| `--epochs` | `200` | Training epochs for each LSTM model |
| `--cv-splits` | `3` | Number of chronological cross-validation folds |

## How it works

1. Downloads daily OHLCV data for the selected ticker.
2. Builds rolling samples from 29 prior trading days of Open, High, Low, Close, and Volume, with the following day's Close as the target.
3. Fits feature scaling on each training portion only, then applies it to the following validation period.
4. Trains a two-layer LSTM and compares it against the naive “last Close” baseline.
5. Reports RMSE and MAE for each chronological validation fold and for the final 20% holdout.
6. Plots the holdout actual Close, both predictions, and the LSTM absolute error.

The chronological splits keep future observations out of each fold's training data. The final holdout remains separate from cross-validation and is used for the plot.
