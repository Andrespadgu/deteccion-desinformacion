"""
Pipeline completo de reproducibilidad.

Ejecutar desde la raíz del proyecto:
    python generar_todo.py

Genera:
    - outputs/resultados.json
    - outputs/figuras/*.png  (4 figuras PNG)
    - dashboard.html actualizado con datos embebidos
"""

import sys, json, os
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

from features import aplicar_pipeline_completo, calcular_porcentajes_decision

# Paleta
AZUL = "#1e6cbf"; AZUL_CLARO = "#4a9edd"
ROJO = "#d9534f"; AMARILLO = "#f0a500"; VERDE = "#3aaa5c"
GRIS_FONDO = "#f7f9fc"; GRIS_GRID = "#e0e6ed"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.facecolor": GRIS_FONDO,
    "figure.facecolor": "white", "axes.grid": True, "grid.color": GRIS_GRID,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

os.makedirs("outputs/figuras", exist_ok=True)

print("Cargando datos...")
df = pd.read_csv("data/BASE.csv")
df_e = aplicar_pipeline_completo(df)
df_e["Fecha"] = pd.to_datetime(df_e["Fecha_Hora_Publicacion"])
df_e["minuto"] = df_e["Fecha"].dt.floor("5min")
pct = calcular_porcentajes_decision(df_e["Decision_Semaforo"])
print(f"Pipeline OK — {len(df_e):,} registros")

# ------- FIGURAS -------

def fig1_score():
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = range(0, 105, 5)
    n, _, patches = ax.hist(df_e["Score_Confiabilidad"], bins=bins, edgecolor="white", linewidth=0.5)
    for patch in patches:
        c = patch.get_x()
        patch.set_facecolor(ROJO if c < 40 else (AMARILLO if c < 65 else VERDE))
        patch.set_alpha(0.85)
    ax.axvline(40, color=ROJO,  linestyle="--", linewidth=1.5)
    ax.axvline(65, color=VERDE, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Score de Confiabilidad"); ax.set_ylabel("Publicaciones")
    ax.set_title("Distribución del Score de Confiabilidad\nCaso Kai Duarte — 3.210 publicaciones", fontsize=13, fontweight="bold")
    c_n = (df_e["Score_Confiabilidad"]<40).sum()
    b_n = ((df_e["Score_Confiabilidad"]>=40)&(df_e["Score_Confiabilidad"]<=65)).sum()
    m_n = (df_e["Score_Confiabilidad"]>65).sum()
    ax.legend(handles=[
        mpatches.Patch(color=ROJO,    alpha=0.85, label=f"Cancelar (<40) — {c_n} pub."),
        mpatches.Patch(color=AMARILLO, alpha=0.85, label=f"Bajar video (40–65) — {b_n} pub."),
        mpatches.Patch(color=VERDE,   alpha=0.85, label=f"Mantener (>65) — {m_n} pub."),
    ], loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/figuras/fig1_score_confiabilidad.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig1 OK")


def fig2_timeline():
    tl = df_e.groupby("minuto").agg(
        total=("ID_Publicacion","count"),
        cancelar=("Decision_Semaforo", lambda x:(x=="Cancelar").sum()),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(tl["minuto"], tl["total"], alpha=0.2, color=AZUL)
    ax.plot(tl["minuto"], tl["total"], color=AZUL, linewidth=2, label="Total publicaciones")
    ax.fill_between(tl["minuto"], tl["cancelar"], alpha=0.35, color=ROJO)
    ax.plot(tl["minuto"], tl["cancelar"], color=ROJO, linewidth=1.5, linestyle="--", label="Zona Cancelar")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=30)
    ax.set_xlabel("Hora (2026-03-15)"); ax.set_ylabel("Publicaciones / 5 min")
    ax.set_title("Línea de tiempo del escándalo", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/figuras/fig2_linea_tiempo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig2 OK")


def fig3_plataforma():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plat = df_e.groupby("Plataforma")["Indice_Sospecha"].mean().sort_values()
    cp = [ROJO if v>0.45 else AMARILLO if v>0.35 else VERDE for v in plat.values]
    bars = axes[0].barh(plat.index, plat.values, color=cp, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("IS prom."); axes[0].set_title("Por Plataforma", fontweight="bold"); axes[0].set_xlim(0, 0.75)
    for bar, val in zip(bars, plat.values): axes[0].text(val+0.01,bar.get_y()+bar.get_height()/2,f"{val:.2f}",va="center",fontsize=9)
    perfil = df_e.groupby("Perfil_Usuario")["Indice_Sospecha"].mean().sort_values()
    cp2 = [ROJO if v>0.5 else AMARILLO if v>0.35 else VERDE for v in perfil.values]
    bars2 = axes[1].barh(perfil.index, perfil.values, color=cp2, alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("IS prom."); axes[1].set_title("Por Perfil", fontweight="bold"); axes[1].set_xlim(0, 0.95)
    for bar, val in zip(bars2, perfil.values): axes[1].text(val+0.01,bar.get_y()+bar.get_height()/2,f"{val:.2f}",va="center",fontsize=9)
    fig.suptitle("Índice de Sospecha por Plataforma y Perfil", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/figuras/fig3_sospecha_plataforma_perfil.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig3 OK")


def fig4_top_cuentas():
    top = (df_e.groupby("Usuario_Handle").agg(is_p=("Indice_Sospecha","mean"),n=("ID_Publicacion","count"))
           .query("n>=3").sort_values("is_p",ascending=False).head(20))
    fig, ax = plt.subplots(figsize=(9, 7))
    ct = [ROJO if v>0.65 else AMARILLO if v>0.5 else AZUL_CLARO for v in top["is_p"]]
    bars = ax.barh(top.index[::-1], top["is_p"][::-1], color=ct[::-1], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Índice de Sospecha prom."); ax.set_title("Top 20 cuentas más sospechosas", fontweight="bold"); ax.set_xlim(0, 1.0)
    for bar, val in zip(bars, top["is_p"][::-1]): ax.text(val+0.01,bar.get_y()+bar.get_height()/2,f"{val:.2f}",va="center",fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/figuras/fig4_top_cuentas_sospechosas.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig4 OK")


print("\nGenerando figuras...")
fig1_score(); fig2_timeline(); fig3_plataforma(); fig4_top_cuentas()

# ------- JSON -------
print("\nGenerando outputs/resultados.json...")

def serial(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    raise TypeError

def df_to_list(d):
    r = []
    for _, row in d.reset_index().iterrows():
        rr = {}
        for k,v in row.items():
            rr[k] = int(v) if isinstance(v,(np.integer,)) else float(v) if isinstance(v,(np.floating,)) else v
        r.append(rr)
    return r

hist_c, hist_e = np.histogram(df_e["Score_Confiabilidad"], bins=range(0,105,5))
tl = df_e.groupby("minuto").agg(total=("ID_Publicacion","count"),cancelar=("Decision_Semaforo",lambda x:(x=="Cancelar").sum()),bajar=("Decision_Semaforo",lambda x:(x=="Bajar el video").sum()),mantener=("Decision_Semaforo",lambda x:(x=="Mantener").sum()),score_prom=("Score_Confiabilidad","mean")).round(2).reset_index()
top_j = (df_e.groupby("Usuario_Handle").agg(is2=("Indice_Sospecha","mean"),sc2=("Score_Confiabilidad","mean"),n2=("ID_Publicacion","count"),p2=("Perfil_Usuario",lambda x:x.mode()[0]),pl2=("Plataforma",lambda x:x.mode()[0])).query("n2>=2").sort_values("is2",ascending=False).head(30).round(2).reset_index())
muestras = {}
for zona in ["Cancelar","Bajar el video","Mantener"]:
    sub = df_e[df_e["Decision_Semaforo"]==zona].sort_values("Score_Confiabilidad").head(5)
    muestras[zona] = [{"id":int(r.ID_Publicacion),"usuario":r.Usuario_Handle,"plataforma":r.Plataforma,"tipo":r.Tipo_Publicacion,"texto":str(r.Texto_Publicacion)[:200],"score":float(r.Score_Confiabilidad),"indice_sospecha":float(r.Indice_Sospecha),"peso_prueba":float(r.Peso_Prueba),"tasa_coordinacion":float(r.Tasa_Coordinacion)} for _,r in sub.iterrows()]

plat_s = df_e.groupby("Plataforma").agg(total=("ID_Publicacion","count"),score_promedio=("Score_Confiabilidad","mean"),indice_sospecha=("Indice_Sospecha","mean")).round(2)
tipo_s = df_e.groupby("Tipo_Publicacion").agg(total=("ID_Publicacion","count"),score_promedio=("Score_Confiabilidad","mean"),peso_prueba=("Peso_Prueba","mean")).round(2)
perf_s = df_e.groupby("Perfil_Usuario").agg(total=("ID_Publicacion","count"),score_promedio=("Score_Confiabilidad","mean"),indice_sospecha=("Indice_Sospecha","mean")).round(2)

resultado = {
    "resumen":{"total_publicaciones":int(len(df_e)),"periodo_inicio":str(df_e["Fecha_Hora_Publicacion"].min()),"periodo_fin":str(df_e["Fecha_Hora_Publicacion"].max()),"score_promedio":round(float(df_e["Score_Confiabilidad"].mean()),2),"score_mediana":round(float(df_e["Score_Confiabilidad"].median()),2),"indice_sospecha_promedio":round(float(df_e["Indice_Sospecha"].mean()),3),"decision":pct},
    "por_plataforma":df_to_list(plat_s), "por_tipo_publicacion":df_to_list(tipo_s), "por_perfil_usuario":df_to_list(perf_s),
    "linea_de_tiempo":[{"hora":row["minuto"].strftime("%H:%M"),"total":int(row["total"]),"cancelar":int(row["cancelar"]),"bajar":int(row["bajar"]),"mantener":int(row["mantener"]),"score_prom":float(row["score_prom"])} for _,row in tl.iterrows()],
    "top_cuentas_sospechosas":[{"usuario":r.Usuario_Handle,"indice_sospecha":float(r.is2),"score_confiabilidad":float(r.sc2),"num_publicaciones":int(r.n2),"perfil":r.p2,"plataforma":r.pl2} for _,r in top_j.iterrows()],
    "distribucion_score":[{"bin_inicio":int(hist_e[i]),"bin_fin":int(hist_e[i+1]),"conteo":int(hist_c[i])} for i in range(len(hist_c))],
    "muestras_por_zona":muestras,
    "metadata":{"generado_el":"2026-10-13","version":"1.0","fuente":"data/BASE.csv","nota":"Datos sinteticos — fines academicos"},
}

with open("outputs/resultados.json","w",encoding="utf-8") as f:
    json.dump(resultado,f,ensure_ascii=False,indent=2,default=serial)
print("  resultados.json OK")

# ------- ACTUALIZAR DASHBOARD -------
print("\nActualizando dashboard.html...")
json_str = json.dumps(resultado, ensure_ascii=False, indent=2, default=serial)

with open("dashboard.html", encoding="utf-8") as f:
    lineas = f.readlines()

inicio = None
for i, l in enumerate(lineas):
    if "const DATOS_EMBEBIDOS" in l:
        inicio = i
        break

if inicio is None:
    print("  WARN: no se encontró DATOS_EMBEBIDOS en dashboard.html")
else:
    profundidad = 0
    fin = None
    for i in range(inicio, len(lineas)):
        profundidad += lineas[i].count("{") - lineas[i].count("}")
        if i > inicio and profundidad <= 0 and ";" in lineas[i]:
            fin = i; break

    if fin:
        lineas_nuevas = lineas[:inicio] + ["const DATOS_EMBEBIDOS = " + json_str + ";\n"] + lineas[fin+1:]
        with open("dashboard.html","w",encoding="utf-8") as f:
            f.writelines(lineas_nuevas)
        print("  dashboard.html OK")

print(f"\nOK Todo generado. Decision final: {pct['decision_final']} ({pct['confianza_pct']:.1f}%)")
