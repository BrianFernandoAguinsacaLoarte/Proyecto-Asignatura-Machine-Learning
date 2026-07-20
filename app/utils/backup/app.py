import streamlit as st
from utils.preprocess import preprocess_input
from utils.predict import predict_student
from utils.shap_explainer import explain_prediction

#from utils.shap_explainer import plot_shap_waterfall

st.set_page_config(
    page_title="Predicción de Riesgo Académico",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# Título
# --------------------------------------------------

st.title("🎓 Sistema de Predicción de Riesgo Académico")

st.markdown("""
Esta aplicación permite estimar el riesgo académico de un estudiante
utilizando un modelo de Machine Learning basado en XGBoost.

Ingrese la información académica del estudiante para estimar la probabilidad
de riesgo académico mediante un modelo XGBoost Optimizado.

La predicción incluye una explicación del resultado,
identificando los principales factores académicos que influyeron en la estimación.
""")

st.divider()

# ============================================================
# FORMULARIO
# ============================================================

with st.form("prediction_form"):

    # ========================================================
    # INFORMACIÓN ACADÉMICA
    # ========================================================

    st.subheader("📘 Información académica")

    col1, col2 = st.columns(2)

    with col1:

        module = st.selectbox(
            "Asignatura",
            (
                "BBB",
                "CCC",
                "DDD",
                "EEE",
                "FFF",
                "GGG"
            )
        )

        previous_attempts = st.number_input(
            "Número de veces que ha cursado la asignatura",
            min_value=0,
            max_value=10,
            value=0
        )

    with col2:

        education = st.selectbox(
            "Nivel de estudios previo",
            (
                "Sin estudios formales",
                "Bachiller",
                "Tecnólogo",
                "Universitario",
                "Posgrado"
            )
        )

        academic_load = st.selectbox(
            "Carga académica",
            (
                "Baja",
                "Media",
                "Alta"
            )
        )

    st.divider()

    # ========================================================
    # RENDIMIENTO ACADÉMICO
    # ========================================================

    st.subheader("📊 Rendimiento académico")

    col1, col2 = st.columns(2)

    with col1:

        weighted_grade = st.number_input(
            "Promedio ponderado (0 - 10)",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1,
            help="Ingrese el promedio académico en escala de 0 a 10."
        )

        score_mean = st.number_input(
            "Promedio de calificaciones (0 - 10)",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

        score_max = st.number_input(
            "Calificación máxima (0 - 10)",
            min_value=0.0,
            max_value=10.0,
            value=8.0,
            step=0.1
        )

    with col2:

        score_min = st.number_input(
            "Calificación mínima (0 - 10)",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1
        )

        score_std = st.number_input(
            "Desviación estándar de las calificaciones",
            min_value=0.0,
            max_value=5.0,
            value=0.8,
            step=0.1
        )

        score_trend = st.slider(
            "Tendencia del rendimiento",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.01
        )

    st.divider()

    # ========================================================
    # PARTICIPACIÓN
    # ========================================================

    st.subheader("💻 Participación en el aula virtual")

    col1, col2 = st.columns(2)

    with col1:

        num_assessments = st.number_input(
            "Actividades evaluadas realizadas",
            min_value=0,
            max_value=50,
            value=5
        )

        active_days = st.number_input(
            "Días con actividad en el aula virtual",
            min_value=0,
            max_value=365,
            value=30
        )

        total_clicks = st.number_input(
            "Número de accesos al aula virtual",
            min_value=0,
            value=300
        )

    with col2:

        late_ratio = st.slider(
            "Proporción de tareas entregadas fuera de plazo",
            0.0,
            1.0,
            0.10,
            help="0 representa ninguna entrega tardía y 1 representa todas las entregas fuera de plazo."
        )

        unfinished_tasks = st.number_input(
            "Número de tareas no entregadas",
            min_value=0,
            value=0
        )

    st.divider()

    # ========================================================
    # INFORMACIÓN PERSONAL
    # ========================================================

    st.subheader("👤 Información del estudiante")

    col1, col2 = st.columns(2)

    with col1:

        age = st.selectbox(
            "Grupo de edad",
            (
                "Menor de 35 años",
                "35 a 55 años",
                "Mayor de 55 años"
            )
        )

    with col2:

        disability = st.selectbox(
            "Discapacidad registrada",
            (
                "No",
                "Sí"
            )
        )

    st.divider()

    predict = st.form_submit_button(
        "🔍 Analizar estudiante",
        use_container_width=True
    )

# ============================================================
# VALIDACIÓN
# ============================================================

if predict:

    errores = []

    if score_max < score_min:
        errores.append(
            "La calificación máxima no puede ser menor que la mínima."
        )

    if weighted_grade > score_max:
        errores.append(
            "El promedio ponderado no puede superar la calificación máxima."
        )

    if unfinished_tasks > num_assessments:
        errores.append(
            "Las tareas no entregadas no pueden superar las actividades realizadas."
        )

    if late_ratio == 1 and unfinished_tasks == 0:
        errores.append(
            "Si todas las tareas fueron entregadas tarde, debería existir al menos una entrega registrada."
        )

    if score_mean > score_max:
        errores.append(
            "El promedio de calificaciones no puede superar la nota máxima."
        )

    if score_mean < score_min:
        errores.append(
            "El promedio de calificaciones no puede ser menor que la nota mínima."
        )

    if active_days == 0 and total_clicks > 0:
        errores.append(
            "Existen accesos registrados pero cero días de actividad."
        )

    if errores:

        st.error("Se encontraron los siguientes problemas:")

        for e in errores:
            st.write("•", e)

    else:

        st.success("Datos validados correctamente.")

        st.info(
            "Procesando información del estudiante mediante el modelo XGBoost."
        )

        # ---------------------------------------------
        # Construcción del registro del estudiante
        # ---------------------------------------------

        student_data = {

            "module": module,

            "previous_attempts": previous_attempts,

            "education": education,

            "academic_load": academic_load,

            "weighted_grade": weighted_grade * 10,

            "score_mean": score_mean * 10,

            "score_max": score_max * 10,

            "score_min": score_min * 10,

            "score_std": score_std * 10,

            "score_trend": score_trend,

            "num_assessments": num_assessments,

            "active_days": active_days,

            "total_clicks": total_clicks,

            "late_ratio": late_ratio,

            "unfinished_tasks": unfinished_tasks,

            "age": age,

            "disability": disability

        }

        # ---------------------------------------------
        # Preprocesamiento
        # ---------------------------------------------

        X = preprocess_input(student_data)

        # =====================================================
        # RESULTADO DEL MODELO
        # =====================================================

        result = predict_student(X)


        st.divider()

        st.subheader("📋 Resultado del análisis académico")


        # -----------------------------------------------------
        # Resultado principal
        # -----------------------------------------------------

        if result["prediction"] == 1:

            st.error(
                "⚠️ Riesgo Académico detectado"
            )

        else:

            st.success(
                "✅ No Riesgo Académico"
            )

        fig, importance, explanations = explain_prediction(X)

        with st.expander("🔎 Ver explicación técnica SHAP"):

            st.pyplot(fig)

            st.dataframe(
                importance.head(10)
            )

        st.subheader("💡 Explicación del resultado")


        for item in explanations:

            if item.get("impact") == "Incrementa la probabilidad de riesgo":

                st.warning(
                    f"""
                    🔴 **{item['name']}**

                    {item['message']}
                    """
                )

            else:

                st.success(
                    f"""
                    🟢 **{item['name']}**

                    {item['message']}
                    """
                )

        # -----------------------------------------------------
        # Métricas
        # -----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Confianza del modelo",
                f"{result['confidence']*100:.2f}%"
            )


        with col2:

            st.metric(
                "Probabilidad de riesgo",
                f"{result['prob_risk']*100:.2f}%"
            )


        with col3:

            st.metric(
                "Probabilidad de no riesgo",
                f"{result['prob_no_risk']*100:.2f}%"
            )

        
        
        # Barra visual
        st.subheader("📊 Nivel de riesgo")


        st.progress(
            result["prob_risk"]
        )


        st.caption(
            f"Probabilidad estimada de riesgo académico: "
            f"{result['prob_risk']*100:.2f}%"
        )

        
        #Recomendacion 
        st.subheader("💡 Recomendación")


        if result["prediction"] == 1:

            st.warning(
                """
                El estudiante presenta indicadores asociados
                a riesgo académico.

                Se recomienda realizar seguimiento sobre:
                - Rendimiento académico.
                - Cumplimiento de actividades.
                - Participación en el aula virtual.
                """
            )

        else:

            st.info(
                """
                El estudiante presenta indicadores compatibles
                con un desempeño académico adecuado.

                Se recomienda mantener el seguimiento regular.
                """
            )