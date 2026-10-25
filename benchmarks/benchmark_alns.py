"""Benchmark sintético: A Etapa 2, B mejora local, C ALNS directo, D local + ALNS, E local + ALNS + intensificación.

Uso: python benchmarks/benchmark_alns.py [tiempo_limite_s]
Sale con código 1 si alguna corrida devuelve algo infactible, con precedencias rotas, sin conservar
movimientos, fuera de tiempo, sin explorar o peor que un incumbente hallado en la misma ejecución.
"""
import sys

import pandas as pd

import _comun
from _comun import SEMILLAS, VARIANTES, ejecutar
from tests.test_interferencias import config, config_gmph, cronograma, fila, intervalo_fn, pontón


def _multigrua(n, tareas_por_grua=5):
    filas = []
    for k in range(n):
        t = 0
        for j in range(tareas_por_grua):
            bay = 2 + 12 * k + 4 * (j % 3)
            filas.append(fila(f"G{k + 1}", bay, 30 + 10 * ((j + k) % 4), t, tipo="Descarga" if j % 2 == 0 else "Embarque"))
            t += 130 + 10 * j
    return cronograma(*filas)


def casos():
    conflicto = cronograma(
        fila("G1", 10, 50, 0), fila("G1", 14, 40, 100), fila("G1", 14, 20, 180, tipo="Embarque"),
        pontón("G1", 18, 250), fila("G2", 14, 60, 0), fila("G2", 18, 40, 120),
        fila("G2", 22, 30, 200, tipo="Embarque"),
    )
    factible = cronograma(fila("G1", 2, 60, 0), fila("G1", 6, 30, 120), fila("G2", 30, 60, 0), fila("G2", 34, 30, 120))
    desbalance = cronograma(fila("G1", 2, 200, 0), fila("G1", 10, 200, 400), fila("G2", 30, 20, 0))
    cadenas = cronograma(
        fila("G1", 2, 200, 0), fila("G1", 2, 200, 400), fila("G1", 14, 60, 800), fila("G1", 14, 60, 1000),
        fila("G2", 30, 20, 0), fila("G2", 30, 60, 100), fila("G2", 14, 60, 300), fila("G2", 34, 40, 500),
    )
    return [
        ("conflictivo (primera factible 355)", conflicto, config("G1", "G2"), {}),
        ("ya factible", factible, config("G1", "G2"), {}),
        ("reasignación 800→320", desbalance, config_gmph(30, 60), {"permitir_division": False}),
        ("división (420 movimientos)", desbalance, config_gmph(30, 60), {}),
        ("cadenas con división y reasignación", cadenas, config_gmph(30, 60), {}),
        ("3 grúas", _multigrua(3), config("G1", "G2", "G3"), {}),
        ("4 grúas", _multigrua(4), config("G1", "G2", "G3", "G4"), {}),
    ]


def main():
    tiempo = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    filas, errores = [], []
    for nombre, df, cfg, extra in casos():
        for variante, kw in VARIANTES:
            for semilla in SEMILLAS:
                fila_, errs = ejecutar(df, cfg, intervalo_fn, 15, variante, kw, semilla, tiempo, extra)
                fila_["caso"] = nombre
                filas.append(fila_)
                errores += [f"{nombre} | {variante} | seed {semilla}: {e}" for e in errs]
    tabla = pd.DataFrame(filas)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 800)
    columnas = ["caso", "variante", "seed", "factible", "ms_semilla", "ms_inicial_candidatos", "ms_antes_alns",
                "ms_mejor_alns", "ms_tras_alns", "ms_tras_intensificacion", "ms_final", "precedencias_ok",
                "interf", "cruces", "movs_ok", "t_total_s", "iter_alns", "gruas_usadas", "fragmentos",
                "cand_distintas", "pct_identicas", "op_destruccion_ganador"]
    print(tabla[columnas].to_string(index=False))
    print("\nMakespan final medio por caso y variante:")
    print(tabla.pivot_table(index="caso", columns="variante", values="ms_final", aggfunc="mean").round(1).to_string())
    print("\nCandidatas idénticas (%) medias por caso (variantes con ALNS):")
    print(tabla[tabla["iter_alns"] > 0].groupby("caso")["pct_identicas"].mean().round(1).to_string())
    if errores:
        print("\nERRORES:\n" + "\n".join(errores))
        return 1
    print("\nOK: factibilidad, precedencias, movimientos, tiempo, no regresión y exploración verificados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
