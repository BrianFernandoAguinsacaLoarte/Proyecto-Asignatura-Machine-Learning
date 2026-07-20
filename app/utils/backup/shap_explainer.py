"""
=========================================================
Explicabilidad mediante SHAP
=========================================================
"""

import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from utils.explanation_generator import generate_explanation


# ==========================================================
# Ruta del modelo
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgb_model_opt.joblib"


# ==========================================================
# Cargar modelo una sola vez
# ==========================================================

model = joblib.load(MODEL_PATH)

explainer = shap.TreeExplainer(model)


# ==========================================================
# Explicación local
# ==========================================================

def explain_prediction(student_df):

    """
    Genera la explicación SHAP para un único estudiante.

    Parameters
    ----------
    student_df : pandas.DataFrame

    Returns
    -------
    fig : matplotlib.figure.Figure

    importance_df : pandas.DataFrame

    human_explanations : list
    Explicaciones interpretables para usuario final.
    """

    # ============================================
    # Calcular valores SHAP
    # ============================================

    shap_values = explainer.shap_values(student_df)

    # ============================================
    # Crear gráfico Waterfall
    # ============================================

    explanation = shap.Explanation(

        values=shap_values[0],

        base_values=explainer.expected_value,

        data=student_df.iloc[0].values,

        feature_names=student_df.columns

    )

    plt.close("all")

    shap.plots.waterfall(
        explanation,
        max_display=5,
        show=False
    )

    fig = plt.gcf()

    # ============================================
    # Tabla de importancia
    # ============================================

    importance_df = pd.DataFrame({

        "Variable": student_df.columns,

        "Valor": student_df.iloc[0].values,

        "Contribución SHAP": shap_values[0]

    })

    importance_df["Impacto"] = importance_df[
        "Contribución SHAP"
    ].abs()

    importance_df = importance_df.sort_values(

        "Impacto",

        ascending=False

    )

    importance_df = importance_df.drop(
        columns="Impacto"
    )

    # ============================================
    # Generar explicación humana
    # ============================================

    human_explanations = []


    for _, row in importance_df.head(5).iterrows():

        explanation = generate_explanation(

            feature=row["Variable"],

            value=row["Valor"],

            shap_value=row["Contribución SHAP"]

        )


        human_explanations.append(
            explanation
        )

    return fig, importance_df, human_explanations
