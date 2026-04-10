"""
Generate synthetic customer dataset for training and testing.
Run: python scripts/generate_sample_data.py
Outputs: data/sample_customers.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 2000

def generate_dataset(n: int = N) -> pd.DataFrame:
    tenure = np.random.randint(0, 72, n)
    monthly_charge = np.random.uniform(20, 200, n)
    support_tickets = np.random.poisson(1.5, n)
    login_freq = np.random.poisson(15, n)
    last_activity = np.random.randint(0, 120, n)
    nps = np.random.randint(0, 11, n)
    late_payments = np.random.poisson(0.5, n)

    # Churn probability influenced by features
    churn_score = (
        0.3 * (support_tickets / 10)
        + 0.25 * (last_activity / 120)
        + 0.2 * (1 - nps / 10)
        + 0.15 * (late_payments / 5)
        + 0.1 * (1 - login_freq / 30)
        + np.random.normal(0, 0.1, n)
    )
    churned = (churn_score > 0.5).astype(int)

    # Segment labels based on RFM heuristics
    def assign_segment(row):
        if row["login_frequency_30d"] > 20 and row["monthly_charge"] > 100:
            return "champion"
        elif row["tenure_months"] > 36 and row["nps_score"] >= 7:
            return "loyal"
        elif row["last_activity_days_ago"] > 60:
            return "hibernating"
        elif row["support_tickets_90d"] > 5 or row["nps_score"] < 5:
            return "at_risk"
        elif row["tenure_months"] < 6:
            return "new"
        else:
            return "potential"

    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(n)],
        "age": np.random.randint(18, 70, n),
        "gender": np.random.choice(["M", "F", "Other"], n, p=[0.48, 0.48, 0.04]),
        "location": np.random.choice(
            ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"], n
        ),
        "tenure_months": tenure,
        "subscription_plan": np.random.choice(
            ["Basic", "Standard", "Premium"], n, p=[0.3, 0.4, 0.3]
        ),
        "monthly_charge": monthly_charge.round(2),
        "total_charges": (monthly_charge * (tenure + 1)).round(2),
        "num_products": np.random.randint(1, 6, n),
        "contract_type": np.random.choice(
            ["Month-to-Month", "One Year", "Two Year"], n, p=[0.5, 0.3, 0.2]
        ),
        "payment_method": np.random.choice(
            ["Credit Card", "Debit Card", "Net Banking", "UPI"], n
        ),
        "login_frequency_30d": np.clip(login_freq, 0, 30),
        "support_tickets_90d": np.clip(support_tickets, 0, 20),
        "avg_session_duration_min": np.random.exponential(12, n).round(1),
        "last_activity_days_ago": last_activity,
        "nps_score": nps,
        "late_payments_count": np.clip(late_payments, 0, 10),
        "discount_used": np.random.choice([True, False], n, p=[0.3, 0.7]),
        "referrals_made": np.random.poisson(0.8, n),
        "last_feedback": np.random.choice([
            "Great service, very happy!",
            "App crashes sometimes, frustrating.",
            "Thinking about cancelling, too expensive.",
            "Support team was very helpful.",
            "Average experience, nothing special.",
            "Love the new features!",
            "Billing issue not resolved, very disappointed.",
            None,
        ], n),
        "churned": churned,
    })

    df["segment"] = df.apply(assign_segment, axis=1)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/sample_customers.csv", index=False)
    print(f"✅ Generated {len(df)} customer records → data/sample_customers.csv")
    print(f"   Churn rate: {df['churned'].mean():.1%}")
    print(f"   Segment distribution:\n{df['segment'].value_counts()}")
