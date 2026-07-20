"""
=========================================================
Generador de explicaciones en lenguaje natural (XAI)
=========================================================

Este módulo constituye la capa de Inteligencia Artificial
Explicable (XAI) orientada a usuarios NO técnicos: docentes,
tutores y estudiantes.

Su responsabilidad es traducir el resultado numérico de SHAP
(valor de la variable + contribución SHAP) en una narrativa
comprensible, evitando por completo terminología técnica de
Machine Learning.

Este módulo NO calcula valores SHAP ni interactúa con el
modelo. Esa responsabilidad pertenece exclusivamente a
``shap_explainer.py``. Aquí únicamente se construye el
lenguaje natural a partir de la información ya calculada.

La información semántica (etiqueta, descripción, unidad, etc.)
se obtiene desde ``utils/feature_dictionary.py`` mediante
``get_feature_info()``. Este módulo no debe modificar ni
duplicar esa información, solo consumirla.

Arquitectura
------------
Cada variable relevante del modelo cuenta con su propia función
``explain_<variable>()``, especializada en construir una
narrativa propia y contextual para esa variable. Estas funciones
son registradas en ``GENERATOR_MAP`` y se seleccionan
dinámicamente desde la función pública ``generate_explanation()``.

Esta organización permite que, en el futuro, funciones de más
alto nivel (por ejemplo ``generate_prediction_summary()`` o
``generate_recommendations()``) reutilicen las explicaciones
individuales ya construidas por ``generate_explanation()`` sin
necesidad de reescribir la lógica narrativa de cada variable.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from utils.feature_dictionary import get_feature_info


# =========================================================
# Constantes de impacto
# =========================================================
# Estas cadenas se utilizan literalmente en app.py para decidir
# el color/estilo del mensaje mostrado al usuario. No deben
# modificarse sin actualizar también app.py.

INCREASE_IMPACT: str = "Incrementa la probabilidad de riesgo"
DECREASE_IMPACT: str = "Reduce la probabilidad de riesgo"


# =========================================================
# Variables cuyo valor fue escalado x10 durante el
# preprocesamiento (ver utils/preprocess.py), y que por lo
# tanto deben reconvertirse a la escala original (0-10)
# únicamente para fines de presentación en la narrativa.
# =========================================================

_SCALED_BY_TEN = {
    "weighted_grade",
    "score_mean",
    "score_max",
    "score_min",
    "score_std",
}


# =========================================================
# Helpers genéricos de formato e interpretación
# =========================================================

def _is_risk_factor(shap_value: float) -> bool:
    """Determina si la contribución SHAP incrementa el riesgo.

    Parameters
    ----------
    shap_value : float
        Contribución SHAP de la variable para el estudiante.

    Returns
    -------
    bool
        ``True`` si la variable incrementó la probabilidad de
        riesgo, ``False`` si actuó como factor protector.
    """
    return shap_value > 0


def _impact_label(shap_value: float) -> str:
    """Devuelve la etiqueta de impacto estandarizada para la UI."""
    return INCREASE_IMPACT if _is_risk_factor(shap_value) else DECREASE_IMPACT


def _display_value(feature: str, value: float) -> float:
    """Reconvierte a escala original (0-10) las variables que
    fueron escaladas x10 durante el preprocesamiento.

    Parameters
    ----------
    feature : str
        Nombre técnico de la variable.
    value : float
        Valor tal como fue recibido por el modelo.

    Returns
    -------
    float
        Valor listo para mostrarse al usuario final.
    """
    if feature in _SCALED_BY_TEN:
        return value / 10
    return value


def _fmt(value: float, decimals: int = 1) -> str:
    """Formatea un número evitando decimales artificiales.

    Ejemplo: ``_fmt(7.0)`` -> "7", ``_fmt(7.35)`` -> "7.4"
    """
    rounded = round(float(value), decimals)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.{decimals}f}"


def _percent(value: float) -> str:
    """Formatea una proporción (0-1) como porcentaje legible."""
    return f"{round(float(value) * 100)}%"


# =========================================================
# Explicaciones especializadas — Rendimiento académico
# =========================================================

def explain_weighted_grade(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para el promedio ponderado del estudiante."""
    grade = _display_value("weighted_grade", value)
    texto_valor = f"El estudiante obtuvo un promedio ponderado de {_fmt(grade)}/10"

    if _is_risk_factor(shap_value):
        if grade < 6:
            return (
                f"{texto_valor}, un resultado por debajo de lo esperado para un "
                "desempeño académico estable. Este promedio limitado fue uno de "
                "los factores que más incrementó la probabilidad de riesgo "
                "académico."
            )
        return (
            f"{texto_valor}, valor inferior al esperado para estudiantes sin "
            "riesgo, lo que incrementó la probabilidad de riesgo académico."
        )

    if grade >= 8:
        return (
            f"{texto_valor}, un resultado sólido que actuó como un factor "
            "protector y contribuyó a reducir el riesgo académico estimado."
        )
    return (
        f"{texto_valor}, lo que actuó como un factor protector y redujo el "
        "riesgo académico estimado."
    )


def explain_score_mean(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para el promedio de calificaciones en actividades evaluadas."""
    mean = _display_value("score_mean", value)
    base = f"En promedio, el estudiante alcanzó {_fmt(mean)}/10 en sus actividades evaluadas"

    if _is_risk_factor(shap_value):
        return (
            f"{base}, un rendimiento sostenido que no alcanza el nivel esperado "
            "y que incrementó la probabilidad de riesgo académico."
        )
    return (
        f"{base}, un rendimiento constante que ayudó a disminuir el riesgo "
        "académico estimado."
    )


def explain_score_max(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para la mejor calificación obtenida."""
    best = _display_value("score_max", value)
    base = f"La calificación más alta registrada por el estudiante fue {_fmt(best)}/10"

    if _is_risk_factor(shap_value):
        return (
            f"{base}. Incluso en su mejor desempeño, el resultado no fue "
            "suficiente para compensar otras dificultades, lo que contribuyó "
            "a incrementar el riesgo académico."
        )
    return (
        f"{base}, lo que demuestra que el estudiante es capaz de alcanzar un "
        "buen desempeño y esto disminuyó el riesgo académico estimado."
    )


def explain_score_min(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para la calificación más baja obtenida."""
    worst = _display_value("score_min", value)
    base = f"La calificación más baja obtenida por el estudiante fue {_fmt(worst)}/10"

    if _is_risk_factor(shap_value):
        return (
            f"{base}. Este resultado puntual señala una dificultad importante "
            "en al menos una actividad, lo que incrementó la probabilidad de "
            "riesgo académico."
        )
    return (
        f"{base}, un piso de rendimiento razonable que evitó caídas fuertes en "
        "el desempeño y contribuyó a reducir el riesgo académico."
    )


def explain_score_std(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para la variabilidad (dispersión) de las calificaciones."""
    dispersion = _display_value("score_std", value)
    base = f"Las calificaciones del estudiante muestran una variabilidad de {_fmt(dispersion)} puntos"

    if _is_risk_factor(shap_value):
        return (
            f"{base}, lo que indica un desempeño irregular a lo largo del "
            "curso. Esta inconsistencia incrementó la probabilidad de riesgo "
            "académico."
        )
    return (
        f"{base}, un nivel de dispersión bajo que refleja un desempeño "
        "consistente y ayudó a reducir el riesgo académico estimado."
    )


def explain_score_trend(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para la tendencia del rendimiento académico a lo largo del curso."""
    if value > 0.05:
        tendencia = "una tendencia de mejora en su rendimiento académico"
    elif value < -0.05:
        tendencia = "una tendencia de deterioro en su rendimiento académico"
    else:
        tendencia = "un rendimiento estable, sin cambios significativos"

    base = f"El estudiante muestra {tendencia} a lo largo del curso"

    if _is_risk_factor(shap_value):
        return (
            f"{base}. Esta evolución del desempeño incrementó la probabilidad "
            "de riesgo académico."
        )
    return (
        f"{base}. Esta evolución del desempeño contribuyó a reducir el riesgo "
        "académico estimado."
    )


# =========================================================
# Explicaciones especializadas — Participación
# =========================================================

def explain_active_days(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para los días de actividad en el aula virtual."""
    days = int(round(value))

    if _is_risk_factor(shap_value):
        return (
            f"El estudiante registró actividad únicamente durante {days} días "
            "en el aula virtual. Esta baja participación incrementó el riesgo "
            "académico."
        )
    return (
        f"El estudiante registró actividad durante {days} días en el aula "
        "virtual. Esta participación constante contribuyó a disminuir el "
        "riesgo académico."
    )


def explain_total_clicks(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para el número total de accesos al aula virtual."""
    clicks = int(round(value))

    if _is_risk_factor(shap_value):
        return (
            f"Se registraron {clicks} interacciones totales del estudiante en "
            "el aula virtual, un nivel de uso escaso que incrementó la "
            "probabilidad de riesgo académico."
        )
    return (
        f"Se registraron {clicks} interacciones totales del estudiante en el "
        "aula virtual, lo que refleja un compromiso activo con el curso y "
        "redujo el riesgo académico estimado."
    )


def explain_avg_clicks_per_day(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para el promedio diario de interacciones en el aula virtual."""
    avg = _fmt(value)

    if _is_risk_factor(shap_value):
        return (
            f"En promedio, el estudiante interactuó {avg} veces por día en el "
            "aula virtual, una frecuencia baja que incrementó el riesgo "
            "académico estimado."
        )
    return (
        f"En promedio, el estudiante interactuó {avg} veces por día en el "
        "aula virtual, una frecuencia que evidencia hábitos de estudio "
        "regulares y contribuyó a reducir el riesgo académico."
    )


def explain_max_clicks_day(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para el pico máximo de interacción diaria registrado."""
    peak = int(round(value))

    if _is_risk_factor(shap_value):
        return (
            f"El día de mayor actividad del estudiante alcanzó apenas {peak} "
            "interacciones, lo que sugiere picos de participación débiles y "
            "aumentó la probabilidad de riesgo académico."
        )
    return (
        f"El día de mayor actividad del estudiante alcanzó {peak} "
        "interacciones, un pico de participación notorio que ayudó a "
        "disminuir el riesgo académico estimado."
    )


def explain_unfinished_tasks(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para el número de actividades no entregadas."""
    pending = int(round(value))

    if pending == 0:
        base = "El estudiante no registra actividades pendientes de entrega"
    elif pending == 1:
        base = "El estudiante mantiene 1 actividad pendiente de entrega"
    else:
        base = f"El estudiante mantiene {pending} actividades pendientes de entrega"

    if _is_risk_factor(shap_value):
        return f"{base}, situación que incrementó el riesgo académico."
    return f"{base}, lo que contribuyó a mantener un riesgo académico bajo."


def explain_late_ratio(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para la proporción de entregas tardías."""
    ratio = _percent(value)

    if _is_risk_factor(shap_value):
        return (
            f"El {ratio} de las entregas del estudiante se realizaron fuera "
            "de plazo, un patrón de incumplimiento que incrementó la "
            "probabilidad de riesgo académico."
        )
    return (
        f"Solo el {ratio} de las entregas del estudiante se realizaron fuera "
        "de plazo, lo que refleja puntualidad en el cumplimiento de "
        "actividades y redujo el riesgo académico estimado."
    )


def explain_num_assessments(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para la cantidad de actividades evaluadas realizadas."""
    count = int(round(value))

    if _is_risk_factor(shap_value):
        return (
            f"El estudiante ha completado {count} actividades evaluadas, un "
            "número reducido que limita la evidencia de su desempeño y "
            "incrementó la probabilidad de riesgo académico."
        )
    return (
        f"El estudiante ha completado {count} actividades evaluadas, lo que "
        "brinda una base sólida de evidencia sobre su desempeño y ayudó a "
        "reducir el riesgo académico estimado."
    )


# =========================================================
# Explicaciones especializadas — Información académica
# =========================================================

def explain_previous_attempts(value: float, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para el número de intentos previos en la asignatura."""
    attempts = int(round(value))

    if attempts == 0:
        base = "Esta es la primera vez que el estudiante cursa la asignatura"
    elif attempts == 1:
        base = "El estudiante ya ha cursado esta asignatura en una ocasión anterior"
    else:
        base = f"El estudiante ha cursado esta asignatura en {attempts} ocasiones anteriores"

    if _is_risk_factor(shap_value):
        return (
            f"{base}, lo que sugiere dificultades previas y incrementó la "
            "probabilidad de riesgo académico."
        )
    return (
        f"{base}, situación que no representó un factor adverso relevante y "
        "contribuyó a reducir el riesgo académico estimado."
    )


def explain_academic_load(value: Any, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para el nivel de carga académica del estudiante."""
    carga = str(value)
    base = f"El estudiante presenta una carga académica clasificada como '{carga}'"

    if _is_risk_factor(shap_value):
        return (
            f"{base}, lo que exige una mayor dedicación y disponibilidad de "
            "tiempo, incrementando la probabilidad de riesgo académico."
        )
    return (
        f"{base}, un nivel que resulta manejable en relación con sus demás "
        "actividades y contribuyó a reducir el riesgo académico estimado."
    )


def explain_education(value: Any, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para el nivel de estudios previo del estudiante."""
    nivel = str(value)
    base = f"El estudiante registra como nivel de estudios previo '{nivel}'"

    if _is_risk_factor(shap_value):
        return (
            f"{base}, una base formativa que podría implicar mayores retos de "
            "adaptación y que incrementó la probabilidad de riesgo académico."
        )
    return (
        f"{base}, una base formativa que favorece su desempeño actual y "
        "contribuyó a reducir el riesgo académico estimado."
    )


def explain_module(value: Any, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para la asignatura analizada."""
    asignatura = str(value)
    base = f"El análisis corresponde a la asignatura '{asignatura}'"

    if _is_risk_factor(shap_value):
        return (
            f"{base}, una materia que históricamente presenta mayores tasas "
            "de riesgo académico, lo que incrementó la probabilidad estimada."
        )
    return (
        f"{base}, una materia en la que el estudiante muestra condiciones "
        "favorables, lo que contribuyó a reducir el riesgo académico "
        "estimado."
    )


# =========================================================
# Explicaciones especializadas — Información personal
# =========================================================

def explain_age(value: Any, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para el grupo de edad del estudiante."""
    grupo = str(value)
    base = f"El estudiante pertenece al grupo de edad '{grupo}'"

    if _is_risk_factor(shap_value):
        return (
            f"{base}, un rango en el que suelen presentarse mayores "
            "exigencias de conciliación entre el estudio y otras "
            "responsabilidades, lo que incrementó la probabilidad de riesgo "
            "académico."
        )
    return (
        f"{base}, un rango asociado en este caso a condiciones favorables "
        "para el desempeño académico, lo que contribuyó a reducir el riesgo "
        "estimado."
    )


def explain_disability(value: Any, shap_value: float, info: Dict[str, Any]) -> str:
    """Narrativa para el registro de discapacidad del estudiante."""
    registra = str(value).strip().lower() in {"sí", "si", "1", "true", "yes"}

    if registra:
        base = "El estudiante registra una discapacidad"
    else:
        base = "El estudiante no registra ninguna discapacidad"

    if _is_risk_factor(shap_value):
        return (
            f"{base}, condición que el modelo asoció a un mayor "
            "acompañamiento necesario, incrementando la probabilidad de "
            "riesgo académico."
        )
    return (
        f"{base}, lo cual contribuyó a reducir el riesgo académico estimado."
    )


# =========================================================
# Normalización: nombres técnicos del modelo -> conceptuales
# =========================================================
# El DataFrame que llega desde shap_explainer.py utiliza los
# nombres de columna reales del modelo (definidos en
# models/feature_names.joblib), que no siempre coinciden con
# las claves semánticas de FEATURE_INFO / GENERATOR_MAP. Este
# bloque traduce esos nombres técnicos a los nombres
# conceptuales que el resto de este módulo ya sabe explicar,
# sin tocar GENERATOR_MAP ni las funciones de explicación.

# Renombres directos (mismo significado, distinto nombre técnico).
_TECHNICAL_TO_CONCEPTUAL: Dict[str, str] = {
    "num_of_prev_attempts": "previous_attempts",
    "late_submission_ratio": "late_ratio",
}

# Prefijos de columnas dummy (one-hot) generadas en preprocess.py.
_MODULE_PREFIX = "code_module_"
_EDUCATION_PREFIX = "highest_education_"
_AGE_PREFIX = "age_band_"
_ACADEMIC_LOAD_PREFIX = "academic_load_"
_DISABILITY_DUMMY = "disability_Y"

# Traducciones exactas para los sufijos conocidos de educación y
# edad, para conservar el mismo texto que el estudiante vio en
# el formulario (utils/app.py).
_EDUCATION_HUMAN: Dict[str, str] = {
    "HE Qualification": "Universitario",
    "Lower Than A Level": "Bachiller",
    "No Formal quals": "Sin estudios formales",
    "Post Graduate Qualification": "Posgrado",
}

_AGE_HUMAN: Dict[str, str] = {
    "35-55": "35 a 55 años",
    "55=": "Mayor de 55 años",
}


def _to_native(value: Any) -> Any:
    """Convierte escalares de numpy a tipos nativos de Python.

    Evita exponer ``numpy.float64`` / ``numpy.int64`` en el
    diccionario de retorno de ``generate_explanation``.
    """
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _is_active(raw_value: Any) -> bool:
    """Determina si una columna dummy (one-hot) está activa."""
    try:
        return float(raw_value) == 1.0
    except (TypeError, ValueError):
        return bool(raw_value)


def _resolve_feature(feature: str, value: Any) -> tuple[str, Any]:
    """Traduce un nombre técnico de columna a su forma conceptual.

    Maneja dos casos:

    1. Renombres directos (misma variable, distinto nombre de
       columna), por ejemplo ``num_of_prev_attempts`` ->
       ``previous_attempts``.
    2. Variables categóricas codificadas como dummy (one-hot),
       por ejemplo ``code_module_BBB`` -> variable ``module``
       con valor humano ``"BBB"``.

    Parameters
    ----------
    feature : str
        Nombre técnico de la columna tal como llega desde
        ``student_df.columns``.
    value : Any
        Valor de esa columna para el estudiante analizado.

    Returns
    -------
    tuple[str, Any]
        Nombre conceptual de la variable y su valor humano,
        listos para ``get_feature_info`` y ``GENERATOR_MAP``.
    """

    # ------------------------------------------------
    # 1) Renombres directos
    # ------------------------------------------------
    if feature in _TECHNICAL_TO_CONCEPTUAL:
        return _TECHNICAL_TO_CONCEPTUAL[feature], _to_native(value)

    # ------------------------------------------------
    # 2) Asignatura (module)
    # ------------------------------------------------
    if feature.startswith(_MODULE_PREFIX):
        categoria = feature[len(_MODULE_PREFIX):]
        human_value = categoria if _is_active(value) else f"distinta a {categoria}"
        return "module", human_value

    # ------------------------------------------------
    # 3) Nivel de estudios previo (education)
    # ------------------------------------------------
    if feature.startswith(_EDUCATION_PREFIX):
        sufijo = feature[len(_EDUCATION_PREFIX):]
        categoria = _EDUCATION_HUMAN.get(sufijo, sufijo)
        human_value = categoria if _is_active(value) else f"distinto a {categoria}"
        return "education", human_value

    # ------------------------------------------------
    # 4) Grupo de edad (age)
    # ------------------------------------------------
    if feature.startswith(_AGE_PREFIX):
        sufijo = feature[len(_AGE_PREFIX):]
        categoria = _AGE_HUMAN.get(sufijo, sufijo)
        human_value = categoria if _is_active(value) else f"distinto a {categoria}"
        return "age", human_value

    # ------------------------------------------------
    # 5) Discapacidad (disability)
    # ------------------------------------------------
    if feature == _DISABILITY_DUMMY:
        human_value = "Sí" if _is_active(value) else "No"
        return "disability", human_value

    # ------------------------------------------------
    # 6) Carga académica (academic_load), solo si el modelo
    #    llegara a incluirla como variable dummy (ver
    #    preprocess.py). Se resuelve de forma defensiva por si
    #    en el futuro el modelo se reentrena con esta variable.
    # ------------------------------------------------
    if feature.startswith(_ACADEMIC_LOAD_PREFIX):
        categoria = feature[len(_ACADEMIC_LOAD_PREFIX):]
        human_value = categoria if _is_active(value) else f"distinta a {categoria}"
        return "academic_load", human_value

    # ------------------------------------------------
    # 7) Sin transformación: variable ya conceptual o
    #    variable numérica sin codificación especial.
    # ------------------------------------------------
    return feature, _to_native(value)


# =========================================================
# Explicación genérica de respaldo
# =========================================================

def _explain_generic(value: Any, shap_value: float, info: Dict[str, Any]) -> str:
    """Explicación de respaldo para variables sin narrativa especializada.

    Se utiliza únicamente cuando una variable no cuenta con una función
    dedicada en ``GENERATOR_MAP``. Aun así, evita el lenguaje técnico y
    construye una frase legible a partir de la información disponible en
    ``feature_dictionary.py``.
    """
    etiqueta = info.get("label", "esta variable")
    unidad = info.get("unit", "")
    valor_texto = f"{_fmt(value)} {unidad}".strip() if isinstance(value, (int, float)) else str(value)

    base = f"La variable '{etiqueta}' presentó un valor de {valor_texto} para el estudiante"

    if _is_risk_factor(shap_value):
        return f"{base}, lo que incrementó la probabilidad de riesgo académico."
    return f"{base}, lo que contribuyó a reducir el riesgo académico estimado."


# =========================================================
# Mapa de generadores: variable -> función especializada
# =========================================================

GENERATOR_MAP: Dict[str, Callable[[Any, float, Dict[str, Any]], str]] = {
    "weighted_grade": explain_weighted_grade,
    "score_mean": explain_score_mean,
    "score_max": explain_score_max,
    "score_min": explain_score_min,
    "score_std": explain_score_std,
    "score_trend": explain_score_trend,
    "active_days": explain_active_days,
    "total_clicks": explain_total_clicks,
    "avg_clicks_per_day": explain_avg_clicks_per_day,
    "max_clicks_day": explain_max_clicks_day,
    "unfinished_tasks": explain_unfinished_tasks,
    "late_ratio": explain_late_ratio,
    "num_assessments": explain_num_assessments,
    "previous_attempts": explain_previous_attempts,
    "academic_load": explain_academic_load,
    "education": explain_education,
    "module": explain_module,
    "age": explain_age,
    "disability": explain_disability,
}


# =========================================================
# Función pública principal
# =========================================================

def generate_explanation(feature: str, value: Any, shap_value: float) -> Dict[str, Any]:
    """Genera una explicación en lenguaje natural para una variable.

    Obtiene la información semántica de la variable desde
    ``feature_dictionary.get_feature_info()``, selecciona la función
    narrativa especializada correspondiente en ``GENERATOR_MAP`` (o una
    explicación genérica de respaldo si no existe) y construye el
    resultado final listo para ser consumido por la interfaz de usuario.

    Parameters
    ----------
    feature : str
        Nombre técnico de la variable, tal como aparece en el
        DataFrame utilizado por el modelo.
    value : Any
        Valor de la variable para el estudiante analizado.
    shap_value : float
        Contribución SHAP de la variable en la predicción.

    Returns
    -------
    dict
        Diccionario con las claves ``feature``, ``name``,
        ``description``, ``value``, ``impact``, ``message`` y
        ``shap_value``, listo para ser mostrado al usuario final.
    """
    # Traduce el nombre técnico de columna (y, de ser necesario,
    # el valor dummy 0/1) a la variable conceptual y su valor
    # humano correspondiente antes de buscar la narrativa.
    resolved_feature, resolved_value = _resolve_feature(feature, value)

    info = get_feature_info(resolved_feature)

    generator = GENERATOR_MAP.get(resolved_feature, _explain_generic)
    message = generator(resolved_value, shap_value, info)

    return {
        "feature": resolved_feature,
        "name": info.get("label", resolved_feature),
        "description": info.get("description", ""),
        "value": resolved_value,
        "impact": _impact_label(shap_value),
        "message": message,
        "shap_value": float(shap_value),
    }


# =========================================================
# Alias público
# =========================================================
# ``explain_feature`` se mantiene como alias de ``generate_explanation``
# por compatibilidad con el nombre esperado en shap_explainer.py.
# Ambos nombres son intercambiables y apuntan a la misma implementación,
# de modo que futuras funciones (generate_prediction_summary,
# generate_recommendations) puedan reutilizar cualquiera de los dos sin
# necesidad de refactorizar este módulo.

explain_feature = generate_explanation
