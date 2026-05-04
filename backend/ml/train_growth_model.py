"""
LSTM Growth Prediction Model — Training Script
===============================================
Predicts future slum area (sq km) for the next N years given
a historical time series of annual slum area measurements.

Architecture : 2-layer LSTM → Linear output
Input        : Sequence of 10 annual area measurements (normalized)
Output       : Predicted area for next year (denormalized)
Loss         : MSE
Optimizer    : Adam with ReduceLROnPlateau

Using Real Data (recommended):
  Replace generate_growth_data() with real time-series from:
    - UN-Habitat Urban Indicators: https://data.unhabitat.org
    - World Bank Urban Development: https://data.worldbank.org
      → Indicators: "Urban population living in slums (%)"
    - Global Human Settlement Layer (GHSL): https://ghsl.jrc.ec.europa.eu
      → Multi-year settlement grids from 1975–2020
    - Landsat time series (manually extracted area per year)

Expected CSV format for real data:
  region_id, year, slum_area_sqkm
  DHARAVI, 2010, 2.1
  DHARAVI, 2011, 2.18
  ...

Usage:
    python train_growth_model.py
    python train_growth_model.py --data_csv path/to/timeseries.csv
    python train_growth_model.py --seq_len 15 --epochs 200 --hidden 128
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Model Architecture
# ─────────────────────────────────────────────

class LSTMGrowthPredictor(nn.Module):
    """
    2-layer LSTM for slum area time-series prediction.

    Input  : (batch, seq_len, 1) — sequence of normalized area values
    Output : (batch, 1) — predicted next area value (normalized)
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take only the last time step's output
        last_hidden = self.dropout(lstm_out[:, -1, :])
        return self.fc(last_hidden)  # (batch, 1)


# ─────────────────────────────────────────────
#  Data Generation / Loading
# ─────────────────────────────────────────────

def generate_growth_data(n_series: int = 3000, seq_len: int = 10) -> tuple:
    """
    Generate synthetic slum growth time series.

    Models different real-world growth patterns:
    1. Steady linear growth (formal settlement expansion)
    2. Exponential growth (rapid urbanization)
    3. Saturation growth (space constraints)
    4. Decline (demolition / resettlement)

    Returns: (X, y) where:
        X: (n_series, seq_len) — historical areas
        y: (n_series,) — next year area
    """
    np.random.seed(42)
    X_list, y_list = [], []

    patterns = {
        "linear":      int(n_series * 0.35),
        "exponential": int(n_series * 0.40),
        "saturation":  int(n_series * 0.15),
        "decline":     int(n_series * 0.10),
    }

    for pattern, count in patterns.items():
        for _ in range(count):
            base_area = np.random.uniform(0.2, 15.0)  # sq km

            if pattern == "linear":
                growth_per_year = np.random.uniform(0.01, 0.3)
                series = [base_area + growth_per_year * t for t in range(seq_len + 1)]

            elif pattern == "exponential":
                rate = np.random.uniform(0.02, 0.10)
                series = [base_area * ((1 + rate) ** t) for t in range(seq_len + 1)]

            elif pattern == "saturation":
                # Logistic growth — approaches carrying capacity
                capacity = np.random.uniform(base_area * 2, base_area * 5)
                rate = np.random.uniform(0.2, 0.5)
                series = [
                    capacity / (1 + ((capacity - base_area) / base_area) * np.exp(-rate * t))
                    for t in range(seq_len + 1)
                ]

            elif pattern == "decline":
                # Demolition / resettlement scenario
                decline_rate = np.random.uniform(0.02, 0.08)
                series = [base_area * ((1 - decline_rate) ** t) for t in range(seq_len + 1)]

            # Add realistic measurement noise (survey error, boundary changes)
            noise = np.random.normal(0, base_area * 0.02, seq_len + 1)
            series = [max(0.01, s + n) for s, n in zip(series, noise)]

            X_list.append(series[:seq_len])
            y_list.append(series[seq_len])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    logger.info(f"Generated {len(X)} time series (seq_len={seq_len})")
    return X, y


def load_real_timeseries(csv_path: str, seq_len: int = 10) -> tuple:
    """
    Load real time-series data from CSV.

    Expected CSV format:
        region_id, year, slum_area_sqkm
        DHARAVI, 2010, 2.1
        DHARAVI, 2011, 2.18

    Extracts overlapping sequences of length seq_len+1 from each region.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(["region_id", "year"])

    X_list, y_list = [], []
    for region_id, group in df.groupby("region_id"):
        areas = group["slum_area_sqkm"].values
        if len(areas) < seq_len + 1:
            logger.warning(f"Region {region_id} has only {len(areas)} years, need {seq_len+1}. Skipping.")
            continue
        # Extract all overlapping windows
        for i in range(len(areas) - seq_len):
            X_list.append(areas[i:i+seq_len])
            y_list.append(areas[i+seq_len])

    if not X_list:
        raise ValueError(f"No valid sequences found. Need at least {seq_len+1} years per region.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    logger.info(f"Loaded {len(X)} sequences from {df['region_id'].nunique()} regions")
    return X, y


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ── Data ──
    if args.data_csv:
        X, y = load_real_timeseries(args.data_csv, seq_len=args.seq_len)
    else:
        logger.info("No CSV provided — using synthetic data.")
        X, y = generate_growth_data(n_series=3000, seq_len=args.seq_len)

    # ── Normalize ──
    # Per-sequence normalization: divide by max in each sequence
    # This makes the model learn relative growth patterns, not absolute sizes
    X_max = X.max(axis=1, keepdims=True)
    X_max = np.where(X_max == 0, 1, X_max)  # avoid division by zero
    X_norm = X / X_max
    y_norm = y / X_max.squeeze()

    # Save normalization info for inference
    global_scale = float(np.percentile(X.max(axis=1), 95))  # 95th percentile scale
    logger.info(f"Global scale (95th percentile max): {global_scale:.2f} sq km")

    # ── Split ──
    n = len(X_norm)
    split = int(0.85 * n)
    X_train = torch.tensor(X_norm[:split]).unsqueeze(-1)  # (N, seq_len, 1)
    y_train = torch.tensor(y_norm[:split]).unsqueeze(-1)  # (N, 1)
    X_val = torch.tensor(X_norm[split:]).unsqueeze(-1)
    y_val = torch.tensor(y_norm[split:]).unsqueeze(-1)
    X_max_train = torch.tensor(X_max[:split])
    X_max_val = torch.tensor(X_max[split:])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train, X_max_train),
        batch_size=args.batch_size, shuffle=True
    )

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)}")

    # ── Model ──
    model = LSTMGrowthPredictor(
        hidden_size=args.hidden,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, verbose=True
    )
    criterion = nn.MSELoss()

    # ── Training Loop ──
    best_val_loss = float("inf")
    history = []
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred_norm = model(X_val.to(device)).cpu()
            # Denormalize: multiply by original scale
            val_pred = val_pred_norm * X_max_val
            y_val_denorm = y_val * X_max_val
            # MAE in original sq km units
            val_mae = torch.mean(torch.abs(val_pred - y_val_denorm)).item()
            val_loss = criterion(val_pred_norm, y_val).item()

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss/len(train_loader),
                        "val_loss": val_loss, "val_mae_sqkm": val_mae})

        if (epoch % 20 == 0) or epoch == 1:
            logger.info(
                f"Epoch {epoch:04d}/{args.epochs} | "
                f"Train Loss: {train_loss/len(train_loader):.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MAE: {val_mae:.4f} sq km"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "hidden_size": args.hidden,
                "seq_len": args.seq_len,
                "global_scale": global_scale,
                "best_val_loss": best_val_loss,
            }, args.output)

    # Save history
    history_path = Path(args.output).parent / "growth_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n✓ Best model saved to: {args.output}")
    logger.info(f"✓ Training history:    {history_path}")
    logger.info(f"  Best Val Loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM growth prediction model")
    parser.add_argument("--data_csv", default=None, help="CSV with columns: region_id,year,slum_area_sqkm")
    parser.add_argument("--output", default="models/lstm_growth.pt", help="Output model path")
    parser.add_argument("--seq_len", type=int, default=10, help="Historical sequence length (default: 10)")
    parser.add_argument("--hidden", type=int, default=64, help="LSTM hidden size (default: 64)")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs (default: 150)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    args = parser.parse_args()
    train(args)
