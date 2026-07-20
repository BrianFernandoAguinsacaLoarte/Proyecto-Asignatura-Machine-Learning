"""
=========================================================
Predicción utilizando el modelo XGBoost optimizado
=========================================================
"""

import joblib
from pathlib import Path


# =========================================================
# Rutas
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgb_model_opt.joblib"


# =========================================================
# Cargar modelo
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# Función principal
# =========================================================

def predict_student(data):

    """
    Realiza la predicción del riesgo académico.

    Parámetros
    ----------
    data : pandas.DataFrame
        DataFrame preparado por preprocess.py

    Retorna
    -------
    dict
    """

    prediction = model.predict(data)[0]

    probabilities = model.predict_proba(data)[0]


    # Según la codificación del modelo:
    # 0 = No Riesgo
    # 1 = Riesgo Académico

    prob_no_risk = float(probabilities[0])

    prob_risk = float(probabilities[1])


    label = (
        "Riesgo Académico"
        if prediction == 1
        else "No Riesgo"
    )


    # Probabilidad asociada a la clase predicha
    confidence = (
        prob_risk
        if prediction == 1
        else prob_no_risk
    )


    return {

        "prediction": int(prediction),

        "label": label,

        "prob_no_risk": prob_no_risk,

        "prob_risk": prob_risk,

        "confidence": confidence

    }