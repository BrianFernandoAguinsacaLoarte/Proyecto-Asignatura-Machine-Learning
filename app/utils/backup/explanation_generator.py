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
    info = get_feature_info(feature)

    generator = GENERATOR_MAP.get(feature, _explain_generic)
    message = generator(value, shap_value, info)

    return {
        "feature": feature,
        "name": info.get("label", feature),
        "description": info.get("description", ""),
        "value": value,
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
