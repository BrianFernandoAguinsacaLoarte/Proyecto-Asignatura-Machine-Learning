import pandas as pd
import joblib


# ============================
# Cargar nombres de variables
# ============================

FEATURE_NAMES = joblib.load("models/feature_names.joblib")


# ======================================================
# Convierte el formulario de Streamlit al formato XGBoost
# ======================================================

def preprocess_input(data: dict):

    # ----------------------------
    # Variables numéricas
    # ----------------------------

    row = {

        "num_of_prev_attempts": data["previous_attempts"],

        "studied_credits": 60,

        "score_mean": data["score_mean"],

        "score_max": data["score_max"],

        "score_min": data["score_min"],

        "score_std": data["score_std"],

        "num_assessments": data["num_assessments"],

        "weighted_grade": data["weighted_grade"],

        "score_trend": data["score_trend"],

        "late_submission_ratio": data["late_ratio"],

        "unfinished_tasks": data["unfinished_tasks"],

        "total_clicks": data["total_clicks"],

        "active_days": data["active_days"],

        "max_clicks_day": 0,

        "median_clicks_day": 0,

        "avg_clicks_per_day": 0

    }

    # -------------------------------------------------
    # Variables dummy (todas inicialmente en cero)
    # -------------------------------------------------

    for feature in FEATURE_NAMES:

        if feature not in row:

            row[feature] = 0

    # ==================================================
    # MODULE
    # ==================================================

    module = data["module"]

    if module != "AAA":

        feature = f"code_module_{module}"

        if feature in row:

            row[feature] = 1

    # ==================================================
    # HIGHEST EDUCATION
    # ==================================================

    education = data["education"]

    education_map = {

        "Universitario":
            "highest_education_HE Qualification",

        "Bachiller":
            "highest_education_Lower Than A Level",

        "Sin estudios formales":
            "highest_education_No Formal quals",

        "Posgrado":
            "highest_education_Post Graduate Qualification"

    }

    if education in education_map:

        feature = education_map[education]

        if feature in row:

            row[feature] = 1

    # ==================================================
    # AGE BAND
    # ==================================================

    age = data["age"]

    if age == "35 a 55 años":
        row["age_band_35-55"] = 1

    elif age == "Mayor de 55 años":
        row["age_band_55="] = 1

    # ==================================================
    # DISABILITY
    # ==================================================

    if data["disability"] == "Sí":

        row["disability_Y"] = 1

    # ==================================================
    # REGION
    # ==================================================

    #region = data["region"]

    #region_map = {

    #    "Costa":
    #        "region_South Region",

    #    "Sierra":
    #        "region_Wales",

    #    "Amazonía":
    #        "region_Scotland"

    #}

    #if region in region_map:

    #    feature = region_map[region]

    #    if feature in row:

    #        row[feature] = 1

    # ==================================================
    # IMD
    # ==================================================
    # No existe en Ecuador.
    # Se deja en la categoría base (todo cero)

    # ==================================================
    # PRESENTATION
    # ==================================================
    # Tampoco aplica.
    # Queda en la categoría base.

    # ==================================================
    # GENDER
    # ==================================================

    #if data["gender"] == "Masculino":

    #    row["gender_M"] = 1

    # ==================================================
    # Crear DataFrame
    # ==================================================

    df = pd.DataFrame([row])

    # ==================================================
    # Reordenar columnas exactamente igual
    # ==================================================

    df = df[FEATURE_NAMES]

    return df