"""
Ingeniería de features para el análisis del caso "Cancelado" (Kai Duarte).

Cada función calcula una variable derivada y documenta su fórmula.
Todas las salidas están normalizadas a [0, 1] salvo Score_Confiabilidad (0–100).
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Mapas de codificación (fuente: enunciado del reto)
# ---------------------------------------------------------------------------

PERFIL_SOSPECHA = {
    "Bot_Sospechoso": 1.0,
    "Cuenta_Nueva":   0.8,
    "Anónimo":        0.6,
    "Fan_Antiguo":    0.2,
    "Verificado":     0.0,
}

TIPO_PESO_BASE = {
    "Post original":               0.80,
    "Comentario":                  0.50,
    "Repost/Cita":                 0.40,
    "Descripción de captura":      0.30,
    "Transcripción de audio/video": 0.20,
}

RECICLADO_PENALIZACION = {
    "No":                                                       0.00,
    "Posible (similar a una publicación anterior)":             0.20,
    "Sí (idéntico a publicación de 2023 sobre otro creador)":   0.45,
}

# ---------------------------------------------------------------------------
# Thresholds para Score_Confiabilidad → decisión
# ---------------------------------------------------------------------------
UMBRAL_CANCELAR = 40    # score < 40  → Cancelar
UMBRAL_BAJAR    = 65    # 40 ≤ score ≤ 65 → Bajar el video
#                         score > 65 → Mantener


def calcular_indice_sospecha(df: pd.DataFrame) -> pd.Series:
    """
    Indice_Sospecha  (0 = nada sospechoso, 1 = máxima sospecha)

    Fórmula:
        IS = 0.30 * perfil_score
           + 0.25 * antiguedad_inv
           + 0.25 * velocidad_pct
           + 0.20 * reciclado_score

    - perfil_score    : puntaje de sospecha del tipo de perfil (ver PERFIL_SOSPECHA).
    - antiguedad_inv  : inversamente proporcional a la antigüedad normalizada.
                        cuentas < 30 días → ~1.0; cuentas > 365 días → ~0.
                        = max(0, 1 - Antiguedad_Cuenta_Dias / 365) recortado a [0,1].
    - velocidad_pct   : rango percentil de Velocidad_Viralizacion en el dataset.
                        Publicaciones que viralizan más rápido que el 90 % del resto → ~0.9.
    - reciclado_score : penalización por contenido reciclado (ver RECICLADO_PENALIZACION).
    """
    perfil_score   = df["Perfil_Usuario"].map(PERFIL_SOSPECHA).fillna(0.6)

    antiguedad_inv = (1 - (df["Antiguedad_Cuenta_Dias"].fillna(0) / 365)).clip(0, 1)

    velocidad_pct  = df["Velocidad_Viralizacion"].rank(pct=True)

    reciclado_score = df["Contenido_Reciclado"].map(RECICLADO_PENALIZACION).fillna(0.0)

    indice = (
        0.30 * perfil_score
        + 0.25 * antiguedad_inv
        + 0.25 * velocidad_pct
        + 0.20 * reciclado_score
    ).clip(0, 1)

    return indice


def calcular_tasa_coordinacion(df: pd.DataFrame, ventana_min: int = 15) -> pd.Series:
    """
    Tasa_Coordinacion  (0 = no coordinado, 1 = altamente coordinado)

    Fórmula:
        Para cada publicación P en ventana de 'ventana_min' minutos:
            cuentas_sospechosas_ventana = # publicaciones de perfiles Bot/Nueva/Anónimo
                                          en la misma ventana temporal
            total_ventana               = # total de publicaciones en esa ventana
            TC_ventana = cuentas_sospechosas_ventana / total_ventana

        TC_P = proporción de publicaciones sospechosas en su propia ventana temporal.
               Normalizada al [0,1] con el percentil rank global.

    Intuición: si en el mismo bloque de 15 min el 80 % de las publicaciones vienen
    de cuentas nuevas o bots, todas las publicaciones de esa ventana reciben una
    Tasa_Coordinacion alta.
    """
    fechas_ns  = pd.to_datetime(df["Fecha_Hora_Publicacion"]).values.astype("int64")
    sospechoso = df["Perfil_Usuario"].isin(["Bot_Sospechoso", "Cuenta_Nueva", "Anónimo"]).values
    ventana_ns = ventana_min * 60 * int(1e9)

    # Vectorizado con NumPy broadcasting (evita loop Python puro)
    diff    = np.abs(fechas_ns[:, None] - fechas_ns[None, :])   # (n, n)
    en_vnt  = diff <= ventana_ns                                 # máscara de ventana
    total   = en_vnt.sum(axis=1)
    sosp    = (en_vnt & sospechoso[None, :]).sum(axis=1)
    tasas   = np.where(total > 0, sosp / total, 0.0)

    return pd.Series(tasas, index=df.index).clip(0, 1)


def calcular_peso_prueba(df: pd.DataFrame) -> pd.Series:
    """
    Peso_Prueba  (0 = evidencia no verificable, 1 = evidencia sólida)

    Fórmula:
        PP = tipo_base
           + min(0.20, Corroboraciones_Independientes * 0.04)
           - reciclado_penalizacion

        Recortado a [0, 1].

    - tipo_base   : credibilidad base según tipo de publicación (ver TIPO_PESO_BASE).
    - corrobora   : cada corroboración independiente suma +0.04, máximo +0.20.
    - reciclado   : penalización por contenido reciclado (ver RECICLADO_PENALIZACION).
    """
    tipo_base    = df["Tipo_Publicacion"].map(TIPO_PESO_BASE).fillna(0.40)
    corrobora    = (df["Corroboraciones_Independientes"].fillna(0) * 0.04).clip(upper=0.20)
    penalizacion = df["Contenido_Reciclado"].map(RECICLADO_PENALIZACION).fillna(0.0)

    peso = (tipo_base + corrobora - penalizacion).clip(0, 1)
    return peso


def calcular_score_confiabilidad(df: pd.DataFrame,
                                  indice_sospecha: pd.Series,
                                  tasa_coordinacion: pd.Series,
                                  peso_prueba: pd.Series) -> pd.Series:
    """
    Score_Confiabilidad  (0 = no confiable, 100 = totalmente confiable)

    Fórmula:
        SC = 40 * (1 - Indice_Sospecha)
           + 35 *  Peso_Prueba
           + 15 * (1 - Tasa_Coordinacion)
           + 10 *  min(Corroboraciones_Independientes / 7, 1)

    Ponderación:
        40 % → credibilidad del perfil / comportamiento (inverso de sospecha)
        35 % → calidad de la evidencia en sí (peso_prueba)
        15 % → ausencia de coordinación sospechosa
        10 % → respaldo de fuentes independientes

    Resultado redondeado a 2 decimales, en escala 0–100.
    """
    corrobora_norm = (df["Corroboraciones_Independientes"].fillna(0) / 7).clip(0, 1)

    score = (
        40 * (1 - indice_sospecha)
        + 35 * peso_prueba
        + 15 * (1 - tasa_coordinacion)
        + 10 * corrobora_norm
    ).round(2)

    return score.clip(0, 100)


def asignar_decision(score: pd.Series) -> pd.Series:
    """
    Asigna la decisión semáforo a cada publicación según su Score_Confiabilidad.

        score < 40  → 'Cancelar'      (rojo)
        40 ≤ score ≤ 65 → 'Bajar el video'  (amarillo)
        score > 65  → 'Mantener'      (verde)
    """
    condiciones = [
        score < UMBRAL_CANCELAR,
        (score >= UMBRAL_CANCELAR) & (score <= UMBRAL_BAJAR),
        score > UMBRAL_BAJAR,
    ]
    etiquetas = ["Cancelar", "Bajar el video", "Mantener"]
    return pd.Series(np.select(condiciones, etiquetas, default="Bajar el video"), index=score.index, dtype=str)


def calcular_porcentajes_decision(decisiones: pd.Series) -> dict:
    """
    Calcula los porcentajes de las 3 decisiones y la recomendación final.

    Retorna:
        {
            'Cancelar':      float (0–100),
            'Bajar el video': float (0–100),
            'Mantener':      float (0–100),
            'decision_final': str,
            'confianza_pct':  float,
        }
    """
    conteos = decisiones.value_counts(normalize=True) * 100
    resultado = {
        "Cancelar":       round(conteos.get("Cancelar", 0), 1),
        "Bajar el video": round(conteos.get("Bajar el video", 0), 1),
        "Mantener":       round(conteos.get("Mantener", 0), 1),
    }
    decision_final  = max(resultado, key=resultado.get)
    confianza       = resultado[decision_final]
    resultado["decision_final"] = decision_final
    resultado["confianza_pct"]  = confianza
    return resultado


def aplicar_pipeline_completo(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todo el pipeline de features sobre el DataFrame crudo.

    Pasos:
        1. Renombra columna con tilde para compatibilidad uniforme.
        2. Calcula Indice_Sospecha, Tasa_Coordinacion y Peso_Prueba.
        3. Calcula Score_Confiabilidad.
        4. Asigna Decision_Semaforo.

    Retorna un DataFrame enriquecido con 4 columnas nuevas.
    """
    df = df_raw.copy()

    # Normalizar nombre de columna con tilde
    df.rename(columns={"Antigüedad_Cuenta_Dias": "Antiguedad_Cuenta_Dias"}, inplace=True)

    df["Indice_Sospecha"]    = calcular_indice_sospecha(df)
    df["Tasa_Coordinacion"]  = calcular_tasa_coordinacion(df)
    df["Peso_Prueba"]        = calcular_peso_prueba(df)
    df["Score_Confiabilidad"] = calcular_score_confiabilidad(
        df,
        df["Indice_Sospecha"],
        df["Tasa_Coordinacion"],
        df["Peso_Prueba"],
    )
    df["Decision_Semaforo"]  = asignar_decision(df["Score_Confiabilidad"])

    return df
