"""Mushroom classification — bashorat funksiyasi.

Ishlatish:
    python predict.py
"""

import joblib
import pandas as pd

BUNDLE = joblib.load("mushroom_dt.joblib")
MODEL = BUNDLE["model"]
FEATURES = BUNDLE["features"]

# Har bir ustunda qanday qiymatlar bo'lishi mumkin (train paytida ko'rilganlar)
VALID = {
    col: set(cats)
    for col, cats in zip(FEATURES, MODEL.named_steps["ohe"].categories_)
}


def predict_mushroom(features: dict, threshold: float = 0.5) -> dict:
    """Bitta qo'ziqorin uchun bashorat.

    Args:
        features: 22 ta xom qiymat, {ustun_nomi: harf} ko'rinishida.
        threshold: zaharli deb hisoblash chegarasi. Xavfsizroq ishlashi uchun
            pasaytiring (masalan 0.4) — shunda model shubhalansa zaharli deydi.

    Returns:
        {"javob": "ZAHARLI"|"zaharsiz", "zaharli_ehtimoli": float}

    Raises:
        ValueError: ustun yetishmasa yoki notanish qiymat berilsa.
    """
    missing = [c for c in FEATURES if c not in features]
    if missing:
        raise ValueError(f"Yetishmayotgan ustunlar: {missing}")

    unknown = {c: v for c, v in features.items() if c in VALID and v not in VALID[c]}
    if unknown:
        raise ValueError(f"Notanish qiymatlar: {unknown}")

    row = pd.DataFrame([features], columns=FEATURES)
    proba = float(MODEL.predict_proba(row)[0, 1])

    return {
        "javob": "ZAHARLI" if proba >= threshold else "zaharsiz",
        "zaharli_ehtimoli": round(proba, 4),
    }


if __name__ == "__main__":
    namuna = {
        "cap-shape": "x", "cap-surface": "s", "cap-color": "n", "bruises": "t",
        "odor": "p", "gill-attachment": "f", "gill-spacing": "c", "gill-size": "n",
        "gill-color": "k", "stalk-shape": "e", "stalk-root": "e",
        "stalk-surface-above-ring": "s", "stalk-surface-below-ring": "s",
        "stalk-color-above-ring": "w", "stalk-color-below-ring": "w",
        "veil-type": "p", "veil-color": "w", "ring-number": "o", "ring-type": "p",
        "spore-print-color": "k", "population": "s", "habitat": "u",
    }
    print(predict_mushroom(namuna))
