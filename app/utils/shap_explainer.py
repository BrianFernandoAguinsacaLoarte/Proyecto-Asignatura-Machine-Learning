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
# Helpers de compatibilidad SHAP (clasificación binaria)
# ==========================================================
# Según la versión de SHAP y el tipo de modelo, TreeExplainer
# puede devolver los valores SHAP y el valor base de dos formas
# distintas para un clasificador binario. Estos helpers
# normalizan ambos casos para que el resto del código siempre
# trabaje con un vector 1D de longitud n_features y un escalar.

def _select_positive_class_shap_values(raw_shap_values):

    """
    Normaliza la salida de ``explainer.shap_values()`` para una
    única muestra, devolviendo siempre un vector 1D (n_features,)
    correspondiente a la clase positiva (riesgo académico).

    Soporta:

    - Caso 1: ``raw_shap_values`` es un ndarray de forma
      (n_samples, n_features). Se toma la primera muestra.
    - Caso 2: ``raw_shap_values`` es una lista
      ``[array_clase_0, array_clase_1]``. Se toma la clase
      positiva (índice 1) y, dentro de ella, la primera muestra.

    Parameters
    ----------
    raw_shap_values : list or numpy.ndarray
        Salida cruda de ``explainer.shap_values(student_df)``.

    Returns
    -------
    numpy.ndarray
        Vector 1D con la contribución SHAP de cada variable
        para la muestra analizada.
    """

    if isinstance(raw_shap_values, list):

        # Lista por clase: [array_clase_0, array_clase_1]
        positive_class_values = (
            raw_shap_values[1]
            if len(raw_shap_values) > 1
            else raw_shap_values[0]
        )

        return positive_class_values[0]

    # ndarray: puede venir como (n_samples, n_features) o,
    # en versiones más recientes de shap con salida por clase,
    # como (n_samples, n_features, n_classes).
    if raw_shap_values.ndim == 3:
        return raw_shap_values[0, :, -1]

    return raw_shap_values[0]


def _select_positive_class_base_value(raw_expected_value):

    """
    Normaliza ``explainer.expected_value`` para obtener el valor
    base correspondiente a la clase positiva (riesgo académico).

    Soporta tanto un valor escalar único como una lista/array
    con un valor base por clase.

    Parameters
    ----------
    raw_expected_value : float or list or numpy.ndarray
        Valor(es) base entregado(s) por ``TreeExplainer``.

    Returns
    -------
    float
        Valor base escalar correspondiente a la clase positiva.
    """

    if isinstance(raw_expected_value, (list, tuple)):

        return (
            raw_expected_value[1]
            if len(raw_expected_value) > 1
            else raw_expected_value[0]
        )

    if hasattr(raw_expected_value, "__len__"):

        return (
            raw_expected_value[1]
            if len(raw_expected_value) > 1
            else raw_expected_value[0]
        )

    return raw_expected_value


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

    raw_shap_values = explainer.shap_values(student_df)

    shap_values_row = _select_positive_class_shap_values(
        raw_shap_values
    )

    base_value = _select_positive_class_base_value(
        explainer.expected_value
    )

    # ============================================
    # Crear gráfico Waterfall
    # ============================================

    explanation = shap.Explanation(

        values=shap_values_row,

        base_values=base_value,

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

        "Contribución SHAP": shap_values_row

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
