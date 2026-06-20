# ARXIU 1: PGN -> FEN -> .parquet + agrupament per jugada

import os
import chess
import chess.pgn
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

fitxer_pgn = "dades.pgn"
directori_partides = "partides_dades"   
directori_jugades = "jugades_dades"    
partides_per_bloc = 5000


def extreure_posicio(partida):
    tauler = partida.board()
    dades = []
    plies = 0

    for moviment in partida.mainline_moves():
        tauler.push(moviment)
        plies += 1

        if plies % 2 == 0:
            rb = tauler.king(chess.WHITE)
            rn = tauler.king(chess.BLACK)
            fb = (rb is not None) and (chess.square_file(rb) >= 4)
            fn = (rn is not None) and (chess.square_file(rn) >= 4)
            
            dames_blanques = len([p for p in tauler.piece_map().values() if p.symbol() == 'Q'])
            dames_negres   = len([p for p in tauler.piece_map().values() if p.symbol() == 'q'])
            total_dames = dames_blanques + dames_negres
            
            if total_dames == 0:
                categoria_dames = "0 dames"
            elif total_dames == 1:
                categoria_dames = "1 dama"
            else:
                categoria_dames = "2+ dames"
            
            dades.append((tauler.fen(), dames_blanques, dames_negres, categoria_dames, bool(fb == fn)))

    if plies % 2 == 1:
        rb = tauler.king(chess.WHITE)
        rn = tauler.king(chess.BLACK)
        fb = (rb is not None) and (chess.square_file(rb) >= 4)
        fn = (rn is not None) and (chess.square_file(rn) >= 4)
        
        dames_blanques = len([p for p in tauler.piece_map().values() if p.symbol() == 'Q'])
        dames_negres   = len([p for p in tauler.piece_map().values() if p.symbol() == 'q'])
        total_dames = dames_blanques + dames_negres
        
        if total_dames == 0:
            categoria_dames = "0 dames"
        elif total_dames == 1:
            categoria_dames = "1 dama"
        else:
            categoria_dames = "2+ dames"
        
        dades.append((tauler.fen(), dames_blanques, dames_negres, categoria_dames, bool(fb == fn)))

    return dades


def escriure_bloc(buffer, directori, num_bloc):
    taula = pa.table({
        "id_partida": pa.array(buffer["id_partida"], type=pa.int32()),
        "num_jugada": pa.array(buffer["num_jugada"], type=pa.int16()),
        "fen":        pa.array(buffer["fen"],        type=pa.string()),
        "result":     pa.array(buffer["result"],     type=pa.string()),
        "white_elo":  pa.array(buffer["white_elo"],  type=pa.int16()),
        "black_elo":  pa.array(buffer["black_elo"],  type=pa.int16()),
        "year":       pa.array(buffer["year"],       type=pa.int16()),
        "fase":       pa.array(buffer["fase"],       type=pa.string()),
        "dames_blanques":   pa.array(buffer["dames_blanques"],   type=pa.int8()),
        "dames_negres":     pa.array(buffer["dames_negres"],     type=pa.int8()),
        "categoria_dames":  pa.array(buffer["categoria_dames"],  type=pa.string()),
        "reis_mateix_flanc": pa.array(buffer["reis_mateix_flanc"], type=pa.bool_()),
    })
    ruta = os.path.join(directori, f"bloc_{num_bloc:05d}.parquet")
    pq.write_table(taula, ruta, compression="zstd", use_dictionary=True)


def buffer_buit():
    return {
        "id_partida": [], "num_jugada": [], "fen": [], "result": [],
        "white_elo": [], "black_elo": [], "year": [], "fase": [],
        "dames_blanques": [], "dames_negres": [], "categoria_dames": [],
        "reis_mateix_flanc": []
    }


def processar_pgn(fitxer_pgn=fitxer_pgn,
                  directori_sortida=directori_partides,
                  partides_per_bloc=partides_per_bloc):

    os.makedirs(directori_sortida, exist_ok=True)

    buffer        = buffer_buit()
    n_partides    = 0
    partides_bloc = 0
    num_bloc      = 0

    with open(fitxer_pgn, encoding="utf-8", errors="replace") as f_in:
        for _ in tqdm(iter(int, 1), desc="Processant partides", unit=" partides"):
            try:
                partida = chess.pgn.read_game(f_in)
            except Exception:
                continue

            if partida is None:
                break

            try:
                dades = extreure_posicio(partida)
            except Exception:
                continue

            if not dades:
                continue

            result = partida.headers.get("Result", "*")
            white_elo = int(partida.headers.get("WhiteElo", 0))
            black_elo = int(partida.headers.get("BlackElo", 0))
            date_str = partida.headers.get("Date", "2000.01.01")
            year = int(date_str.split('.')[0])

            for num_jugada, (fen, dames_blanques, dames_negres, categoria_dames, reis_mateix_flanc) in enumerate(dades, start=1):
                if num_jugada <= 15:
                    fase = "Obertura"
                elif num_jugada <= 40:
                    fase = "Mig joc"
                else:
                    fase = "Final"

                buffer["id_partida"].append(n_partides)
                buffer["num_jugada"].append(num_jugada)
                buffer["fen"].append(fen)
                buffer["result"].append(result)
                buffer["white_elo"].append(white_elo)
                buffer["black_elo"].append(black_elo)
                buffer["year"].append(year)
                buffer["fase"].append(fase)
                buffer["dames_blanques"].append(dames_blanques)
                buffer["dames_negres"].append(dames_negres)
                buffer["categoria_dames"].append(categoria_dames)
                buffer["reis_mateix_flanc"].append(reis_mateix_flanc)

            n_partides    += 1
            partides_bloc += 1

            if partides_bloc >= partides_per_bloc:
                escriure_bloc(buffer, directori_sortida, num_bloc)
                buffer        = buffer_buit()
                partides_bloc = 0
                num_bloc     += 1

    if partides_bloc > 0:
        escriure_bloc(buffer, directori_sortida, num_bloc)


def agrupar_per_jugada(directori_entrada=directori_partides,
                       directori_sortida=directori_jugades):
    os.makedirs(directori_sortida, exist_ok=True)

    df = pl.read_parquet(os.path.join(directori_entrada, "*.parquet"))

    for (jugada,), df_jugada in df.group_by("num_jugada"):
        ruta = os.path.join(directori_sortida, f"jugada_{jugada:05d}.parquet")
        df_jugada.drop("num_jugada").write_parquet(ruta, compression="zstd")
        print(f"  Jugada {jugada:5d}: {df_jugada.height} posicions")


if __name__ == "__main__":
    processar_pgn()
    agrupar_per_jugada()