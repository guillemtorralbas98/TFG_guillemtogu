# ARXIU 2: Definició i càlcul de la distància

import numpy as np
import chess
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
from numba import njit

OMEGA = np.array([1., 3., 5., 9., 10., 3., 3.,  1., 3., 5., 9., 10., 3., 3.])
PENALTY = np.array([3, 6.0, 7.5, 15.0, 0.0, 6.0, 6.0, 3, 6.0, 7.5, 15.0, 0.0, 6.0, 6.0])

_MAP_PECES = {
    'W': {'P':0, 'N':1, 'R':2, 'Q':3, 'K':4, 'B_light':5, 'B_dark':6},
    'B': {'p':7, 'n':8, 'r':9, 'q':10, 'k':11, 'b_light':12, 'b_dark':13}
}


def fen_a_arrays(fen):
    """FEN → (coords[14, 8, 2], counts[14])."""
    tauler = chess.Board(fen)
    peces_tmp = {i: [] for i in range(14)}
    for sq, peca in tauler.piece_map().items():
        fila, col = sq >> 3, sq & 7
        simbol = peca.symbol()
        color = 'W' if peca.color == chess.WHITE else 'B'
        if simbol.upper() == 'B':
            tipus = simbol + ('_light' if (fila + col) % 2 != 0 else '_dark')
        else:
            tipus = simbol
        idx = _MAP_PECES[color][tipus]
        peces_tmp[idx].append((fila, col))

    coords = np.zeros((14, 8, 2))
    counts = np.zeros(14, dtype=np.int32)
    for idx in range(14):
        llista = sorted(peces_tmp[idx])[:8]
        counts[idx] = len(llista)
        for i, (f, c) in enumerate(llista):
            coords[idx, i, 0] = f
            coords[idx, i, 1] = c
    return coords, counts


@njit(inline='always', cache=True)
def dist_peca(t_idx, f1, c1, f2, c2):
    omega = OMEGA[t_idx]
    df, dc = abs(f1 - f2), abs(c1 - c2)
    if t_idx == 0 or t_idx == 7:        # Peons
        return omega * (dc * 0.19 + df * 0.27)
    elif t_idx in (5, 6, 12, 13):       # Alfils
        return omega * max(df, dc) * 0.28
    else:                                # Cavalls, Torres, Reines, Reis
        return omega * max(df, dc) * 0.14


def dist_grup(t_idx, coords1, len1, coords2, len2):
    pen = PENALTY[t_idx]
    if len1 == 0 and len2 == 0: return 0.0
    if len1 == 0: return len2 * pen
    if len2 == 0: return len1 * pen
    if len1 == 1 and len2 == 1:
        return dist_peca(t_idx, coords1[0,0], coords1[0,1], coords2[0,0], coords2[0,1])

    cost = np.zeros((len1, len2))
    for i in range(len1):
        for j in range(len2):
            cost[i, j] = dist_peca(t_idx, coords1[i,0], coords1[i,1], coords2[j,0], coords2[j,1])
    row, col = linear_sum_assignment(cost)
    return cost[row, col].sum() + abs(len1 - len2) * pen


def dist_posicions(c1, l1, c2, l2):
    d = 0.0
    for t_idx in range(14):
        if l1[t_idx] > 0 or l2[t_idx] > 0:
            d += dist_grup(t_idx, c1[t_idx], l1[t_idx], c2[t_idx], l2[t_idx])
    return d


def _fila_matriu(i, coords_all, lens_all):
    n = len(coords_all)
    fila = np.zeros(n)
    for j in range(i + 1, n):
        fila[j] = dist_posicions(coords_all[i], lens_all[i], coords_all[j], lens_all[j])
    return i, fila


def calcular_matriu_distancies(coords_all, lens_all):
    n = len(coords_all)
    resultats = Parallel(n_jobs=-1)(
        delayed(_fila_matriu)(i, coords_all, lens_all) for i in range(n)
    )
    D = np.zeros((n, n))
    for i, fila in resultats:
        D[i, i+1:] = fila[i+1:]
        D[i+1:, i] = fila[i+1:]
    return D


def calcular_distancies(llista_fens):
    n = len(llista_fens)
    coords_all = np.zeros((n, 14, 8, 2))
    lens_all = np.zeros((n, 14), dtype=np.int32)
    for i, fen in enumerate(llista_fens):
        coords_all[i], lens_all[i] = fen_a_arrays(fen)
    return calcular_matriu_distancies(coords_all, lens_all)