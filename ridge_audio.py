"""can find me on social media with the ID 'sadraDiv'. """
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SR = 8000
N = SR // 2
N_TRAIN = 60
N_TEST = 240
CENTERS = np.linspace(360, 740, 24)
BANDWIDTH = 15.0


def simulated_clips(seed=17, n=N_TRAIN + N_TEST):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / SR
    share = rng.uniform(0.05, 0.95, n)
    low = 440 + rng.normal(0, 7, n)
    high = 660 + rng.normal(0, 7, n)
    phase = rng.uniform(0, 2 * np.pi, (2, n))
    a = np.sin(2 * np.pi * low[:, None] * t + phase[0, :, None])
    b = np.sin(2 * np.pi * high[:, None] * t + phase[1, :, None])
    a /= np.sqrt(2 * np.mean(a**2, axis=1, keepdims=True))
    b /= np.sqrt(2 * np.mean(b**2, axis=1, keepdims=True))

    signal = np.sqrt(1 - share[:, None]) * a + np.sqrt(share[:, None]) * b
    gain = rng.uniform(0.65, 1.35, (n, 1))
    noise = rng.uniform(0.16, 0.45, (n, 1)) * rng.normal(size=signal.shape)
    return gain * signal + noise, 100 * share


def spectral_features(clips):
    freq, psd = welch(clips, fs=SR, window="hann", nperseg=512)
    distances = (freq[None, :] - CENTERS[:, None]) / BANDWIDTH
    weights = np.exp(-0.5 * distances**2)
    energy = psd @ weights.T
    total = energy.sum(axis=-1, keepdims=True)
    return energy / np.maximum(total, 1e-20)


def fit_models(X_train, y_train):
    linear = make_pipeline(StandardScaler(), LinearRegression())
    pipeline = make_pipeline(StandardScaler(), Ridge())
    cv = KFold(5, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline, {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
        cv=cv, scoring="neg_mean_absolute_error",
    )
    return linear.fit(X_train, y_train), search.fit(X_train, y_train)


def run(sample_index=0):
    clips, y = simulated_clips()
    X = spectral_features(clips)
    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]
    linear, ridge = fit_models(X_train, y_train)
    mae = {
        "linear": float(mean_absolute_error(y_test, linear.predict(X_test))),
        "ridge": float(mean_absolute_error(y_test, ridge.predict(X_test))),
    }
    left_band, right_band = 15, 16
    corr = float(np.corrcoef(X_train[:, left_band], X_train[:, right_band])[0, 1])
    results = {
        "synthetic_clips": True,
        "training_clips": len(y_train),
        "test_clips": len(y_test),
        "features": X.shape[1],
        "neighboring_band_centers_hz": CENTERS[[left_band, right_band]].tolist(),
        "neighboring_band_pearson_r": corr,
        "selected_alpha": ridge.best_params_["ridge__alpha"],
        "mae_percentage_points": mae,
        "relative_mae_reduction_percent": (1 - mae["ridge"] / mae["linear"]) * 100,
    }
    estimate = float(ridge.predict(X_test[sample_index:sample_index + 1])[0])
    actual = float(y_test[sample_index])
    results["sample"] = {
        "test_index": sample_index,
        "estimated_share_percent": estimate,
        "actual_share_percent": actual,
        "absolute_error_percentage_points": abs(estimate - actual),
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0, help="Held-out clip index (0–239)")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if not 0 <= args.sample < N_TEST:
        parser.error(f"--sample must be between 0 and {N_TEST - 1}")
    report = json.dumps(run(args.sample), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
