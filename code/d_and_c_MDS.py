# ARXIU 3: Divide & Conquer + classical MDS

import os
import warnings
import numpy as np
import polars as pl
from sklearn.manifold import MDS
from private_d_and_c import get_partitions_for_divide_conquer, perform_procrustes
from distancia import fen_a_arrays, calcular_matriu_distancies

warnings.filterwarnings("ignore", category=FutureWarning)

directori_entrada   = "partides_dades"
directori_sortida   = "coordenades_dades"
l = 1000     
c = 100   

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

llista_dataframes  = []
board_fens_globals = []

for nom_fitxer in fitxers_parquet:
    df = pl.read_parquet(os.path.join(directori_entrada, nom_fitxer))
    df = df.with_columns([
        ((pl.col("white_elo") + pl.col("black_elo")) / 2).alias("elo_mitja"),
        pl.col("fen").str.split(" ").list.first().alias("board_fen"),
    ])
    llista_dataframes.append((nom_fitxer, df))
    board_fens_globals.extend(df["board_fen"].to_list())

board_fens_unics = list(set(board_fens_globals))
n_unics = len(board_fens_unics)
print(f"Posicions totals: {len(board_fens_globals)} | úniques (board_fen): {n_unics}")

coords_all = np.zeros((n_unics, 14, 8, 2), dtype=np.float64)
lens_all   = np.zeros((n_unics, 14),        dtype=np.int32)

for i, bf in enumerate(board_fens_unics):
    coords_all[i], lens_all[i] = fen_a_arrays(bf + " w - - 0 1")

coords_uniques = np.zeros((n_unics, 2))

if n_unics <= l:
    print("Dataset petit: MDS directe sense particions.")
    matriu = calcular_matriu_distancies(coords_all, lens_all)
    coords_uniques = mds_classic.fit_transform(matriu)
else:
    idx_list = get_partitions_for_divide_conquer(n=n_unics, l=l, c_points=c, r=2)
    num_partitions = len(idx_list)

    print(f"Partició 1/{num_partitions}...")
    idx_1 = idx_list[0]
    d_1 = calcular_matriu_distancies(coords_all[idx_1], lens_all[idx_1])
    coords_uniques[idx_1] = mds_classic.fit_transform(d_1)

    sample_1_idx     = np.random.choice(len(idx_1), size=c, replace=False)
    idx_anchor       = idx_1[sample_1_idx]
    proj_anchor_base = coords_uniques[idx_anchor]

    for i, idx_part in enumerate(idx_list[1:], start=2):
        print(f"Partició {i}/{num_partitions}...")

        idx_comb  = np.concatenate([idx_anchor, idx_part])
        d_comb    = calcular_matriu_distancies(coords_all[idx_comb], lens_all[idx_comb])
        proj_comb = mds_classic.fit_transform(d_comb)

        coords_uniques[idx_part] = perform_procrustes(
            x=proj_comb[:c, :],
            target=proj_anchor_base,
            matrix_to_transform=proj_comb[c:, :],
            translation=False,
        )

bf_a_coords = {bf: coords_uniques[i] for i, bf in enumerate(board_fens_unics)}

llista_dfs_enriquits = []
for nom_fitxer, df in llista_dataframes:
    bfs = df["board_fen"].to_list()
    x = np.array([bf_a_coords[bf][0] for bf in bfs])
    y = np.array([bf_a_coords[bf][1] for bf in bfs])

    df_enriquit = (
        df.with_columns([pl.Series("x", x), pl.Series("y", y)])
          .drop("board_fen")
    )
    df_enriquit.write_parquet(os.path.join(directori_sortida, nom_fitxer))
    llista_dfs_enriquits.append(df_enriquit)
    print(f" -> {nom_fitxer} desat.")