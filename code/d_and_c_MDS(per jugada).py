# ARXIU 4: Divide & Conquer + classical MDS (per jugada) + càlcul de dimensionalitat

import os
import warnings
import numpy as np
import polars as pl
from sklearn.manifold import MDS
from private_d_and_c import get_partitions_for_divide_conquer, perform_procrustes
from distancia import fen_a_arrays, calcular_matriu_distancies

warnings.filterwarnings("ignore", category=FutureWarning)

directori_entrada     = "jugades_dades"
directori_sortida     = "coordenades_jugades_dades"
l = 1000              
c = 100                
c_inter = 100          
mida_mostra_dim = 2000 

_COLOR_RESULT = {
    "1-0":     "#1f77b4",
    "0-1":     "#d62728",
    "1/2-1/2": "#2ca02c",
}

os.makedirs(directori_sortida, exist_ok=True)

mds_classic = MDS(
    n_components=2,
    dissimilarity="precomputed",
    init="classical_mds",
    n_init=1,
    max_iter=100,
    random_state=42,
)

fitxers_parquet = sorted(f for f in os.listdir(directori_entrada) if f.endswith(".parquet"))
print(f"S'han trobat {len(fitxers_parquet)} jugades per processar.")

llista_dfs_animacio   = []
llista_num_jugada     = []
llista_dimensions_80  = []
llista_partides_vives = []
coordenades_anterior  = {}  

for i, nom_fitxer in enumerate(fitxers_parquet):
    num_jugada = i + 1
    df = pl.read_parquet(os.path.join(directori_entrada, nom_fitxer))

    if df.height <= 1:
        print(f"\n[STOP] Procés aturat a la jugada {num_jugada}: només {df.height} partida(es) viva(es).")
        break

    print(f"\n[JUGADA {num_jugada}] {nom_fitxer} ({df.height} partides vives)")

    df = df.with_columns([
        ((pl.col("white_elo") + pl.col("black_elo")) / 2).alias("elo_mitja"),
        pl.col("fen").str.split(" ").list.first().alias("board_fen"),
    ])

    board_fens_totals = df["board_fen"].to_list()
    board_fens_unics  = list(set(board_fens_totals))
    n_unics = len(board_fens_unics)

    coords_all = np.zeros((n_unics, 14, 8, 2), dtype=np.float64)
    lens_all   = np.zeros((n_unics, 14),         dtype=np.int32)
    for j, bf in enumerate(board_fens_unics):
        coords_all[j], lens_all[j] = fen_a_arrays(bf + " w - - 0 1")

    coords_uniques = np.zeros((n_unics, 2))
    matriu_per_dim = None   

    if n_unics <= l:
        print(f" -> {n_unics} posicions úniques.")
        matriu_per_dim = calcular_matriu_distancies(coords_all, lens_all)
        coords_uniques = mds_classic.fit_transform(matriu_per_dim)
    else:
        idx_list = get_partitions_for_divide_conquer(n=n_unics, l=l, c_points=c, r=2)
        num_partitions = len(idx_list)
        print(f" -> {n_unics} posicions úniques. D&C amb {num_partitions} particions.")

        idx_1 = idx_list[0]
        d_1   = calcular_matriu_distancies(coords_all[idx_1], lens_all[idx_1])
        coords_uniques[idx_1] = mds_classic.fit_transform(d_1)

        sample_1_idx     = np.random.choice(len(idx_1), size=c, replace=False)
        idx_anchor       = idx_1[sample_1_idx]
        proj_anchor_base = coords_uniques[idx_anchor]

        for p, idx_part in enumerate(idx_list[1:], start=2):
            print(f"    Partició {p}/{num_partitions}...")
            idx_comb  = np.concatenate([idx_anchor, idx_part])
            d_comb    = calcular_matriu_distancies(coords_all[idx_comb], lens_all[idx_comb])
            proj_comb = mds_classic.fit_transform(d_comb)

            coords_uniques[idx_part] = perform_procrustes(
                x=proj_comb[:c, :],
                target=proj_anchor_base,
                matrix_to_transform=proj_comb[c:, :],
                translation=False,
            )

    if n_unics <= 2:
        dimensions_necessaries = 1
    else:
        mida = min(mida_mostra_dim, df.height)
        idx_partides = np.random.choice(df.height, size=mida, replace=False)

        bf_a_idx = {bf: j for j, bf in enumerate(board_fens_unics)}
        idx_unics_mostra = np.array(
            [bf_a_idx[board_fens_totals[i]] for i in idx_partides]
        )

        idx_unics_presents, inverse = np.unique(idx_unics_mostra, return_inverse=True)

        if matriu_per_dim is not None:
            matriu_unics = matriu_per_dim[idx_unics_presents][:, idx_unics_presents]
        else:
            matriu_unics = calcular_matriu_distancies(
                coords_all[idx_unics_presents], lens_all[idx_unics_presents]
            )
            
        matriu_dim = matriu_unics[inverse][:, inverse]

        n_calc      = matriu_dim.shape[0]
        d_quadrada  = matriu_dim ** 2
        i_mat       = np.eye(n_calc)
        j_mat       = np.ones((n_calc, n_calc))
        h_mat       = i_mat - (j_mat / n_calc)
        b_mat       = -0.5 * (h_mat @ d_quadrada @ h_mat)

        valors_propis   = np.linalg.eigvalsh(b_mat)[::-1]
        valors_positius = valors_propis[valors_propis > 1e-10]

        if len(valors_positius) > 0:
            variabilitat_acumulada = np.cumsum(valors_positius) / np.sum(valors_positius)
            dimensions_necessaries = int(np.argmax(variabilitat_acumulada >= 0.80) + 1)
        else:
            dimensions_necessaries = 1

    llista_num_jugada.append(num_jugada)
    llista_dimensions_80.append(dimensions_necessaries)
    llista_partides_vives.append(df.height)
    print(f" -> Dimensions per al 80% variància: {dimensions_necessaries}")

    bf_a_coords = {bf: coords_uniques[j] for j, bf in enumerate(board_fens_unics)}
    x_vals = np.array([bf_a_coords[bf][0] for bf in board_fens_totals])
    y_vals = np.array([bf_a_coords[bf][1] for bf in board_fens_totals])

    df_enriquit = (
        df.with_columns([pl.Series("x", x_vals), pl.Series("y", y_vals)])
          .drop("board_fen")
    )

    ids_actuals    = df_enriquit["id_partida"].to_numpy()
    coords_actuals = df_enriquit.select(["x", "y"]).to_numpy()

    if num_jugada > 1 and len(coordenades_anterior) > 0:
        ids_comuns = np.array([id_p for id_p in ids_actuals if id_p in coordenades_anterior])
        n_comuns = len(ids_comuns)

        if n_comuns > 0:
            n_mostra   = min(c_inter, n_comuns)
            ids_mostra = np.random.choice(ids_comuns, size=n_mostra, replace=False)

            pos_actuals       = {id_p: idx for idx, id_p in enumerate(ids_actuals)}
            idx_mostra_actual = np.array([pos_actuals[id_p] for id_p in ids_mostra])

            source = coords_actuals[idx_mostra_actual]
            target = np.array([coordenades_anterior[id_p] for id_p in ids_mostra])

            coords_aliniades = perform_procrustes(
                x=source,
                target=target,
                matrix_to_transform=coords_actuals,
                translation=False,
            )

            df_enriquit = df_enriquit.with_columns([
                pl.Series("x", coords_aliniades[:, 0]),
                pl.Series("y", coords_aliniades[:, 1]),
            ])
            coords_actuals = coords_aliniades

    df_enriquit = df_enriquit.with_columns([
        pl.lit(dimensions_necessaries).alias("dimensions_80"),
        pl.lit(df.height).alias("partides_vives"),
    ])

    df_enriquit.write_parquet(os.path.join(directori_sortida, nom_fitxer))

    coordenades_anterior = {
        id_p: coords_actuals[idx] for idx, id_p in enumerate(ids_actuals)
    }

    df_enriquit = df_enriquit.with_columns(
        pl.Series("jugada", [f"Jugada {num_jugada}"] * df_enriquit.height)
    )
    llista_dfs_animacio.append(df_enriquit)
