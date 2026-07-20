"""
=========================================================
Diccionario de conocimiento de variables

Este módulo almacena únicamente la información semántica
de cada variable utilizada por el modelo. La generación
de explicaciones dinámicas se realiza posteriormente en
explanation_generator.py.
=========================================================
"""

FEATURE_INFO = {

    # =====================================================
    # Rendimiento académico
    # =====================================================

    "weighted_grade": {
        "label": "Promedio ponderado",
        "description": "Promedio académico general del estudiante.",
        "type": "grade",
        "unit": "/10",
        "higher_is_better": True
    },

    "score_mean": {
        "label": "Promedio de calificaciones",
        "description": "Promedio obtenido en las actividades evaluadas.",
        "type": "grade",
        "unit": "/10",
        "higher_is_better": True
    },

    "score_max": {
        "label": "Calificación máxima",
        "description": "Mejor calificación obtenida.",
        "type": "grade",
        "unit": "/10",
        "higher_is_better": True
    },

    "score_min": {
        "label": "Calificación mínima",
        "description": "Calificación más baja obtenida.",
        "type": "grade",
        "unit": "/10",
        "higher_is_better": True
    },

    "score_std": {
        "label": "Variabilidad de las calificaciones",
        "description": "Variación entre las calificaciones obtenidas.",
        "type": "dispersion",
        "unit": "",
        "higher_is_better": False
    },

    "score_trend": {
        "label": "Tendencia del rendimiento",
        "description": "Evolución del rendimiento académico.",
        "type": "trend",
        "unit": "",
        "higher_is_better": True
    },

    # =====================================================
    # Participación
    # =====================================================

    "active_days": {
        "label": "Días de actividad",
        "description": "Cantidad de días con actividad en el aula virtual.",
        "type": "count",
        "unit": "días",
        "higher_is_better": True
    },

    "total_clicks": {
        "label": "Interacciones en el aula virtual",
        "description": "Número total de accesos registrados.",
        "type": "count",
        "unit": "interacciones",
        "higher_is_better": True
    },

    "avg_clicks_per_day": {
        "label": "Promedio diario de interacciones",
        "description": "Promedio de accesos por día.",
        "type": "count",
        "unit": "interacciones/día",
        "higher_is_better": True
    },

    "max_clicks_day": {
        "label": "Máxima interacción diaria",
        "description": "Mayor número de accesos registrados en un día.",
        "type": "count",
        "unit": "interacciones",
        "higher_is_better": True
    },

    "unfinished_tasks": {
        "label": "Actividades pendientes",
        "description": "Número de actividades no entregadas.",
        "type": "count",
        "unit": "actividades",
        "higher_is_better": False
    },

    "late_ratio": {
        "label": "Entregas tardías",
        "description": "Proporción de actividades entregadas fuera de plazo.",
        "type": "ratio",
        "unit": "%",
        "higher_is_better": False
    },

    "num_assessments": {
        "label": "Actividades evaluadas",
        "description": "Cantidad de actividades evaluadas realizadas.",
        "type": "count",
        "unit": "actividades",
        "higher_is_better": True
    },

    "median_clicks_day": {
        "label": "Mediana de interacciones diarias",
        "description": "Valor central de los accesos diarios del estudiante al aula virtual.",
        "type": "count",
        "unit": "interacciones",
        "higher_is_better": True
    },

    # =====================================================
    # Información académica
    # =====================================================

    "studied_credits": {
        "label": "Créditos matriculados",
        "description": "Número de créditos académicos matriculados por el estudiante.",
        "type": "count",
        "unit": "créditos",
        "higher_is_better": None
    },

    "previous_attempts": {
        "label": "Intentos previos",
        "description": "Número de veces que el estudiante ha cursado la asignatura.",
        "type": "count",
        "unit": "intentos",
        "higher_is_better": False
    },

    "academic_load": {
        "label": "Carga académica",
        "description": "Nivel de carga académica del estudiante.",
        "type": "category",
        "higher_is_better": None
    },

    "education": {
        "label": "Nivel de estudios previo",
        "description": "Máximo nivel educativo alcanzado.",
        "type": "category",
        "higher_is_better": None
    },

    "module": {
        "label": "Asignatura",
        "description": "Asignatura analizada.",
        "type": "category",
        "higher_is_better": None
    },

    # =====================================================
    # Información personal
    # =====================================================

    "age": {
        "label": "Grupo de edad",
        "description": "Grupo de edad del estudiante.",
        "type": "category",
        "higher_is_better": None
    },

    "disability": {
        "label": "Discapacidad registrada",
        "description": "Indica si el estudiante registra una discapacidad.",
        "type": "binary",
        "higher_is_better": None
    }

}


def get_feature_info(feature: str) -> dict:
    """
    Devuelve la información semántica de una variable.

    Parameters
    ----------
    feature : str
        Nombre técnico de la variable.

    Returns
    -------
    dict
        Información asociada a la variable.
    """

    return FEATURE_INFO.get(
        feature,
        {
            "label": feature,
            "description": "Variable considerada por el modelo.",
            "type": "unknown",
            "unit": "",
            "higher_is_better": None,
        },
    )