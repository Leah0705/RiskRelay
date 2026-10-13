"""
generate_synthetic_data.py

Generates a synthetic event stream. Each row is one event that needs a risk
decision (a payment, a login, an order, ...) and carries Time, V1-V28
(anonymized PCA-style features), Amount, and Class (1 = high-risk event), plus
account_id, counterparty_id, counterparty_category, device_id and a real
timestamp. Velocity, counterparty-risk and device-consistency features are
built from those identity fields.

The Time/V1-V28/Amount/Class layout is the same as the public Kaggle
credit-card dataset (https://www.kaggle.com/mlg-ulb/creditcardfraud), which is
a convenient source of real labels in this schema.

USE WITH REAL DATA:
src/data_loader.py detects a file that has only Time/V1-V28/Amount/Class (no
account_id/counterparty_id/device_id columns) and synthesizes the identity
fields on top of the real values, using the same logic below (see
`attach_identity_fields`). The label signal stays genuine (from the real
dataset) while the full feature set still builds.

Run:
    python data/generate_synthetic_data.py --n_rows 500000 --positive_rate 0.0017
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def attach_identity_fields(df: pd.DataFrame, n_accounts: int, n_counterparties: int,
                            n_devices: int, seed: int = 42) -> pd.DataFrame:
    """Adds account_id, counterparty_id, counterparty_category, device_id, timestamp
    to a dataframe that already has Time/Amount/Class (real or synthetic).
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # Accounts follow a power-law-ish activity pattern: a few accounts generate far more events than others.
    account_weights = rng.pareto(a=2.0, size=n_accounts) + 0.1
    account_weights /= account_weights.sum()
    df["account_id"] = rng.choice(
        [f"ACCT_{i:06d}" for i in range(n_accounts)], size=n, p=account_weights
    )

    # Ten anonymized counterparty categories; cat_3/cat_4/cat_9/cat_1 carry
    # structurally higher risk, cat_6/cat_0/cat_5 the lowest.
    counterparty_categories = [f"cat_{i}" for i in range(10)]
    category_risk_bias = {
        "cat_0": 0.2, "cat_1": 1.5, "cat_2": 1.2, "cat_3": 3.0, "cat_4": 3.5,
        "cat_5": 0.2, "cat_6": 0.1, "cat_7": 0.6, "cat_8": 0.3, "cat_9": 1.8,
    }
    df["counterparty_id"] = rng.integers(0, n_counterparties, size=n)
    counterparty_to_category = {
        m: counterparty_categories[m % len(counterparty_categories)] for m in range(n_counterparties)
    }
    df["counterparty_category"] = df["counterparty_id"].map(counterparty_to_category)
    df["counterparty_risk_bias"] = df["counterparty_category"].map(category_risk_bias)

    # Devices: most accounts use 1-2 devices consistently; a new/unrecognized
    # device is itself a risk signal we encode via device_id randomness.
    df["device_id"] = [
        f"DEV_{rng.integers(0, max(2, n_devices // n_accounts)):03d}_{acct}"
        if rng.random() > 0.03  # 3% chance of a brand-new device (higher risk)
        else f"DEV_NEW_{rng.integers(0, 999999):06d}"
        for acct in df["account_id"]
    ]

    base_time = datetime(2024, 1, 1)
    df["timestamp"] = [base_time + timedelta(seconds=int(t)) for t in df["Time"]]

    return df


def generate_base_events(n_rows: int, positive_rate: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_pos = int(n_rows * positive_rate)
    n_normal = n_rows - n_pos

    n_pca = 28
    # Normal events: PCA features centered near 0, tight spread
    normal_pca = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_pca))
    # High-risk events: shifted on a subset of components, so the positive
    # class separates on a few PCs
    risky_pca = rng.normal(loc=0.0, scale=1.0, size=(n_pos, n_pca))
    shifted_components = [1, 3, 4, 10, 12, 14, 17]
    for c in shifted_components:
        risky_pca[:, c] += rng.choice([-1, 1]) * rng.uniform(2.5, 5.0)

    normal_amount = np.round(np.abs(rng.lognormal(mean=3.0, sigma=1.1, size=n_normal)), 2)
    risky_amount = np.round(np.abs(rng.lognormal(mean=4.2, sigma=1.4, size=n_pos)), 2)

    normal_time = np.sort(rng.uniform(0, 60 * 60 * 24 * 30, size=n_normal))  # 30 days
    risky_time = np.sort(rng.uniform(0, 60 * 60 * 24 * 30, size=n_pos))

    df_normal = pd.DataFrame(normal_pca, columns=[f"V{i+1}" for i in range(n_pca)])
    df_normal["Time"] = normal_time
    df_normal["Amount"] = normal_amount
    df_normal["Class"] = 0

    df_risky = pd.DataFrame(risky_pca, columns=[f"V{i+1}" for i in range(n_pca)])
    df_risky["Time"] = risky_time
    df_risky["Amount"] = risky_amount
    df_risky["Class"] = 1

    df = pd.concat([df_normal, df_risky], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=300_000)
    parser.add_argument("--positive_rate", type=float, default=0.0017)  # seed rate; the realized rate is higher (README, Data)
    parser.add_argument("--n_accounts", type=int, default=20_000)
    parser.add_argument("--n_counterparties", type=int, default=3_000)
    parser.add_argument("--n_devices", type=int, default=25_000)
    parser.add_argument("--out", type=str, default="data/events.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.n_rows:,} synthetic events "
          f"(seed positive rate ~{args.positive_rate:.4%}) ...")
    df = generate_base_events(args.n_rows, args.positive_rate, args.seed)
    df = attach_identity_fields(df, args.n_accounts, args.n_counterparties, args.n_devices, args.seed)

    # High-risk events are more likely on risky categories and new devices —
    # bias a fraction of Class labels to correlate with those fields so the
    # engineered features actually carry signal (rather than being pure noise).
    rng = np.random.default_rng(args.seed)
    is_new_device = df["device_id"].str.startswith("DEV_NEW_")
    boost_mask = (rng.random(len(df)) < (0.15 * df["counterparty_risk_bias"] / df["counterparty_risk_bias"].max())) | \
                 (is_new_device & (rng.random(len(df)) < 0.05))
    flip_to_risky = boost_mask & (df["Class"] == 0) & (rng.random(len(df)) < 0.4)
    df.loc[flip_to_risky, "Class"] = 1

    df = df.drop(columns=["counterparty_risk_bias"])
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"Positive count: {df['Class'].sum():,} ({df['Class'].mean():.4%})")


if __name__ == "__main__":
    main()
