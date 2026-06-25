# ARXIU 5: Gràfics

import os
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

directori_coords         = "coordenades_dades"
directori_coords_jugades = "coordenades_jugades_dades" 

fitxer_html_animacio     = "mapa_per_jugada.html"
fitxer_html_trajectories = "mapa.html"
fitxer_kde_sortida       = "kde.png"
fitxer_kde_resultat      = "kde_resultat.png"
fitxer_html_analitica = "dimensionalitat_vs_supervivencia.html"

_COLOR_RESULT = {
    "1-0":     "#1f77b4", 
    "0-1":     "#d62728",   
    "1/2-1/2": "#2ca02c",   
}

PARTIDES_RESSALTADES = [19899, 23752]

print(f"Llegint Parquets de {directori_coords_jugades}...")
llista_dfs_animacio = []
for nom_fitxer in sorted(os.listdir(directori_coords_jugades)):
    if not nom_fitxer.endswith(".parquet"):
        continue
    num_jug = int(nom_fitxer.replace("jugada_", "").replace(".parquet", ""))
    df = (
        pl.read_parquet(os.path.join(directori_coords_jugades, nom_fitxer))
        .with_columns(pl.lit(f"Jugada {num_jug}").alias("jugada"))
    )
    llista_dfs_animacio.append(df)
 
df_animacio_total = pl.concat(llista_dfs_animacio)
 
x_min, x_max = df_animacio_total["x"].min() * 1.1, df_animacio_total["x"].max() * 1.1
y_min, y_max = df_animacio_total["y"].min() * 1.1, df_animacio_total["y"].max() * 1.1
 
jugades_uniques = sorted(df_animacio_total["jugada"].unique(), key=lambda x: int(x.split()[1]))
 
 
def construir_traces_jugada(df_jugada):
    traces = []
    for result, color in _COLOR_RESULT.items():
        df_res = df_jugada.filter(pl.col("result") == result)
        if df_res.height > 0:
            customdata = np.column_stack([
                df_res["id_partida"].to_numpy(), df_res["white_elo"].to_numpy(),
                df_res["black_elo"].to_numpy(), df_res["elo_mitja"].to_numpy(),
                df_res["fen"].to_list()
            ])
            x_vals = df_res["x"].to_numpy()
            y_vals = df_res["y"].to_numpy()
        else:
            customdata = np.empty((0, 5))
            x_vals = np.array([])
            y_vals = np.array([])
 
        traces.append(go.Scattergl(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name=f"Resultat {result}",
            marker=dict(size=2, color=color, opacity=0.8),
            customdata=customdata,
            showlegend=True,
            legendgroup=result,
            hovertemplate=(
                "Partida: %{customdata[0]}<br>"
                "ELO Blanc: %{customdata[1]}<br>"
                "ELO Negre: %{customdata[2]}<br>"
                "ELO Mitjà: %{customdata[3]:.0f}<br>"
                "FEN: %{customdata[4]}<br>"
                "<extra></extra>"
            )
        ))
    return traces
 
 
df_inicial = df_animacio_total.filter(pl.col("jugada") == jugades_uniques[0])
fig = go.Figure(data=construir_traces_jugada(df_inicial))
 
llista_frames_plotly = []
diccionari_partides_vives = {}
 
for jug in jugades_uniques:
    df_frame = df_animacio_total.filter(pl.col("jugada") == jug)
    num_partides = df_frame.height
    diccionari_partides_vives[jug] = num_partides

    frame = go.Frame(data=construir_traces_jugada(df_frame), name=jug)
    llista_frames_plotly.append(frame)
 
fig.frames = llista_frames_plotly

args_play = [None, {"frame": {"duration": 400, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0}}]
args_pause = [[None], {"frame": {"duration": 0, "redraw": False},
                       "mode": "immediate",
                       "transition": {"duration": 0}}]
 
def args_step(jug):
    return [[jug], {"frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}}]
 
fig.update_layout(
    title="Mapa de projecció per jugades",
    xaxis=dict(range=[x_min, x_max], title="Dimensió 1", showgrid=True, zeroline=True),
    yaxis=dict(range=[y_min, y_max], title="Dimensió 2", showgrid=True, zeroline=True),
    template="plotly_white",
    title_font_size=18,
    showlegend=True,
    legend=dict(title_text="Resultat final", itemsizing="constant"),
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {"label": "Play", "method": "animate", "args": args_play},
            {"label": "Pause", "method": "animate", "args": args_pause}
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 80},
        "showactive": False,
        "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16}, "prefix": "", "visible": True},
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.9, "x": 0.1, "y": 0,
        "steps": [
            {"args": args_step(jug), "label": jug, "method": "animate"}
            for jug in jugades_uniques
        ]
    }]
)

for frame in fig.frames:
    jug_name = frame.name
    n_partides = diccionari_partides_vives[jug_name]
    frame.layout = {
        "annotations": [
            {
                "text": f"{jug_name} ({n_partides} partides actives)",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.98,
                "showarrow": False,
                "font": {"size": 16, "color": "black"},
                "xanchor": "center", "yanchor": "top"
            }
        ]
    }
 
 
 
fig.write_html(fitxer_html_animacio)
print(f"[OK] Bloc 1: animació desada a: '{fitxer_html_animacio}'")

del llista_dfs_animacio, df_animacio_total, fig

print(f"\nLlegint Parquets de {directori_coords}...")
df_total = pl.read_parquet(os.path.join(directori_coords, "*.parquet"))
print(f"Posicions totals carregades: {df_total.height:,}")

print("\n[BLOC 2] Generant el mapa interactiu de trajectòries...")
coords_pts    = df_total.select(["x", "y"]).to_numpy()
ids_all       = df_total["id_partida"].to_numpy()
jugades_all   = df_total["num_jugada"].to_numpy()
resultats_all = df_total["result"].to_numpy()

n_partides_fons = 5000
partides_unicas = np.unique(ids_all)
n_part_fons = min(n_partides_fons, len(partides_unicas))
ids_partides_fons = np.random.choice(partides_unicas, size=n_part_fons, replace=False)
mask_partides_fons = np.isin(ids_all, ids_partides_fons)

coords_mostra    = coords_pts[mask_partides_fons]
ids_mostra       = ids_all[mask_partides_fons]
jugades_mostra   = jugades_all[mask_partides_fons]
resultats_mostra = resultats_all[mask_partides_fons]
print(f"Partides de fons seleccionades: {n_part_fons:,} "
      f"(~{len(coords_mostra):,} posicions)")

ids_ressaltar = np.array([p for p in PARTIDES_RESSALTADES if p in partides_unicas])

fig = go.Figure()

# Núvol de fons: totes les posicions de la mostra, en gris i petites
fig.add_trace(go.Scattergl(
    x=coords_mostra[:, 0],
    y=coords_mostra[:, 1],
    mode="markers",
    name="Posicions",
    marker=dict(size=1.5, color="gray", opacity=0.6),
    hovertemplate=(
        "Partida %{customdata[0]}<br>"
        "Jugada %{customdata[1]}<br>"
        "Resultat %{customdata[2]}<extra></extra>"
    ),
    customdata=np.column_stack([ids_mostra, jugades_mostra, resultats_mostra]),
))

for id_p in ids_ressaltar:
    mask_p = (ids_all == id_p)
    ordre  = np.argsort(jugades_all[mask_p])

    x_traj = coords_pts[mask_p, 0][ordre]
    y_traj = coords_pts[mask_p, 1][ordre]
    resultat_p = resultats_all[mask_p][0]
    color_resultat = _COLOR_RESULT[resultat_p]

    fig.add_trace(go.Scattergl(
        x=x_traj, y=y_traj, mode="lines",
        name=f"Partida {id_p} ({resultat_p})",
        line=dict(width=1.8, color=color_resultat),
        hoverinfo="skip",
        visible=False,
    ))

num_traces_base = len(fig.data) - len(ids_ressaltar)

opacitats_normals   = [0.6]
opacitats_atenuades = [0.30]

buttons = [
    dict(
        label="Cap partida ressaltada",
        method="update",
        args=[{"visible": [True] * num_traces_base + [False] * len(ids_ressaltar),
               "opacity": opacitats_normals + [1.0] * len(ids_ressaltar)},
              {"title": "Projecció de l'espai de totes les partides"}]
    )
]

for idx_part, id_p in enumerate(ids_ressaltar):
    mask_p = (ids_all == id_p)
    resultat_p = resultats_all[mask_p][0]

    visible = [True] * num_traces_base
    for i in range(len(ids_ressaltar)):
        visible.append(i == idx_part)

    opacitats = opacitats_atenuades + [1.0] * len(ids_ressaltar)

    buttons.append(
        dict(
            label=f"Partida {id_p}",
            method="update",
            args=[{"visible": visible,
                   "opacity": opacitats},
                  {"title": f"Projecció de l'espai de totes les partides | Partida {id_p} ({resultat_p})"}]
        )
    )

fig.update_layout(
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.02,
            xanchor="left",
            y=0.45,
            yanchor="top"
        ),
    ]
)

fig.update_layout(
    title="Mapa de projecció de partides",
    xaxis_title="Component 1",
    yaxis_title="Component 2",
    template="plotly_white",
    legend=dict(itemsizing="constant", title_text="Resultat Final"),
    height=700,
)
fig.update_xaxes(showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor="lightgray")
fig.update_yaxes(showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor="lightgray")

fig.write_html(fitxer_html_trajectories)
print(f"[OK] Bloc 2: gràfic desat a: {fitxer_html_trajectories}")

print("\n[BLOC 3] Generant els tres panells de contorns de densitat...")

prob_content_contour_level = 0.75
mask_kde = mask_partides_fons

x_all = df_total["x"].to_numpy()
y_all = df_total["y"].to_numpy()
fase_all = df_total["fase"].to_numpy()
categoria_dames_all = df_total["categoria_dames"].to_numpy()
result_all = df_total["result"].to_numpy()

x_mostra = x_all[mask_kde]
y_mostra = y_all[mask_kde]
fase_mostra = fase_all[mask_kde]
categoria_dames_mostra = categoria_dames_all[mask_kde]
result_mostra = result_all[mask_kde]
print(f"Partides KDE (reutilitzades del Bloc 2): {n_part_fons:,} "
      f"(~{len(x_mostra):,} posicions)")

xpad = (x_all.max() - x_all.min()) * 0.05
ypad = (y_all.max() - y_all.min()) * 0.05
xmin, xmax = x_all.min() - xpad, x_all.max() + xpad
ymin, ymax = y_all.min() - ypad, y_all.max() + ypad
xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, 200),
    np.linspace(ymin, ymax, 200)
)


def panell_kde(ax, mask_categoria_mostra, colors_dict, etiquetes_dict, titol):
    categories = list(mask_categoria_mostra.keys())

    # Núvol de punts de fons en gris (sota les regions)
    ax.scatter(x_mostra, y_mostra, s=1, c="lightgray",
               alpha=0.3, linewidths=0, zorder=0)

    densities = {}
    for cat in categories:
        mask = mask_categoria_mostra[cat]
        if mask.sum() < 10:
            continue
        data_cat = np.vstack([x_mostra[mask], y_mostra[mask]])
        densities[cat] = gaussian_kde(data_cat)

    evaluated_densities = {}
    for cat in densities.keys():
        mask = mask_categoria_mostra[cat]
        data = np.vstack([x_mostra[mask], y_mostra[mask]])
        evaluated_densities[cat] = densities[cat].evaluate(data)

    zz = {}
    for cat in densities.keys():
        zz[cat] = densities[cat](np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    for cat in densities.keys():
        contour_level = np.quantile(evaluated_densities[cat], 1 - prob_content_contour_level)
        z_max = zz[cat].max()
        if z_max > contour_level:
            ax.contourf(xx, yy, zz[cat], levels=[contour_level, z_max],
                        colors=[colors_dict[cat]], alpha=0.25, zorder=1)
        ax.contour(xx, yy, zz[cat], levels=[contour_level],
                   colors=[colors_dict[cat]], linewidths=2, zorder=2)
        ax.plot([], [], color=colors_dict[cat], label=etiquetes_dict[cat], linewidth=2)

    ax.set_title(titol, fontsize=12, fontweight="bold")


fig_kde, axos = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

colors_fase = {
    "Obertura": "#1f77b4",
    "Mig joc": "#2ca02c",
    "Final": "#d62728"
}
etiquetes_fase = {
    "Obertura": "Obertura (jugades 1-15)",
    "Mig joc":  "Mig joc (jugades 16-40)",
    "Final":    "Final (jugades 40+)"
}
mask_fase = {fase: (fase_mostra == fase) for fase in colors_fase.keys()}
panell_kde(axos[0], mask_fase, colors_fase, etiquetes_fase, "Per fase de la partida")

colors_dames = {
    "0 dames":  "#808080",
    "1 dama":   "#FFD700",
    "2+ dames": "#9370DB"
}
etiquetes_dames = {
    "0 dames":  "0 dames a la posició",
    "1 dama":   "1 dama a la posició",
    "2+ dames": "≥2 dames a la posició"
}
mask_dames_global = {cat: (categoria_dames_mostra == cat) for cat in colors_dames.keys()}
panell_kde(axos[1], mask_dames_global, colors_dames, etiquetes_dames,
           "Per presència de dames (global)")

for ax in axos:
    ax.set_xlabel("Component 1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
axos[0].set_ylabel("Component 2")

fig_kde.suptitle(
    f"Corbes de nivell de les funcions de densitat estimades que contenen el {int(prob_content_contour_level*100)}% de les dades",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()
plt.savefig(fitxer_kde_sortida, dpi=200, bbox_inches="tight")
plt.close()
print(f"[OK] Bloc 3: gràfic KDE (2 panells) desat a: {fitxer_kde_sortida}")

colors_resultat = {
    "1-0":     "#1f77b4",
    "0-1":     "#d62728",
    "1/2-1/2": "#2ca02c"
}
etiquetes_resultat = {
    "1-0":     "Victòria de les blanques (1-0)",
    "0-1":     "Victòria de les negres (0-1)",
    "1/2-1/2": "Taules (1/2-1/2)"
}

fig_kde_res, ax_res = plt.subplots(1, 1, figsize=(8, 7))
mask_resultat = {res: (result_mostra == res) for res in colors_resultat.keys()}
panell_kde(ax_res, mask_resultat, colors_resultat, etiquetes_resultat,
           "Per resultat de la partida")

ax_res.set_xlabel("Component 1")
ax_res.set_ylabel("Component 2")
ax_res.grid(True, alpha=0.3)
ax_res.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax_res.set_xlim(xmin, xmax)
ax_res.set_ylim(ymin, ymax)

fig_kde_res.suptitle(
    f"Corbes de nivell de les funcions de densitat estimades que contenen el {int(prob_content_contour_level*100)}% de les dades",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()
plt.savefig(fitxer_kde_resultat, dpi=200, bbox_inches="tight")
plt.close()
print(f"[OK] Bloc 3: gràfic KDE per resultat desat a: {fitxer_kde_resultat}")

print("\n[BLOC 4] Generant panell analític (dimensionalitat vs supervivència)...")

llista_num_jugada = []
llista_dimensions_80 = []
llista_partides_vives = []

for nom_fitxer in sorted(os.listdir(directori_coords_jugades)):
    if not nom_fitxer.endswith(".parquet"):
        continue
    
    num_jug = int(nom_fitxer.replace("jugada_", "").replace(".parquet", ""))
    df_jugada = pl.read_parquet(os.path.join(directori_coords_jugades, nom_fitxer))
    
    llista_num_jugada.append(num_jug)
    llista_dimensions_80.append(df_jugada["dimensions_80"][0])
    llista_partides_vives.append(df_jugada["partides_vives"][0])

# Crear el gràfic dual-axis amb Plotly
fig_analitica = make_subplots(specs=[[{"secondary_y": True}]])

fig_analitica.add_trace(go.Scatter(
    x=llista_num_jugada, y=llista_dimensions_80,
    mode="lines+markers",
    name="80%",
    line=dict(color="#636EFA", width=2.5),
    marker=dict(size=5),
    hovertemplate="Jugada %{x}<br>Espai de ℝ: %{y}<extra></extra>"
), secondary_y=False)

fig_analitica.add_trace(go.Scatter(
    x=llista_num_jugada, y=llista_partides_vives,
    mode="lines+markers",
    name="Partides en joc",
    line=dict(color="#EF553B", width=2.5),
    marker=dict(size=5),
    hovertemplate="Jugada %{x}<br>Partides en joc: %{y}<extra></extra>"
), secondary_y=True)

fig_analitica.update_layout(
    title="Relació entre dimensionalitat i supervivència de les partides",
    template="plotly_white",
    title_font_size=18,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig_analitica.update_xaxes(title_text="Número de jugada", showgrid=True)
fig_analitica.update_yaxes(title_text="Espai de ℝ", secondary_y=False, color="#636EFA")
fig_analitica.update_yaxes(title_text="Partides en joc", secondary_y=True,
                           color="#EF553B", showgrid=False)

fig_analitica.write_html(fitxer_html_analitica)
print(f"[OK] Bloc 4: panell analític desat a: '{fitxer_html_analitica}'")

print("\n[FET] Els quatre gràfics s'han generat correctament.")