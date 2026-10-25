"""Benchmark con un archivo real de secuencia de trabajo (Etapa 2 contra híbrido).

Uso:
    python benchmarks/benchmark_real.py ruta_archivo [--config gruas.benchmark.local.json] [--tiempo 20] [--holgura 15]

Datos necesarios (nada de esto se incluye en el repositorio):
  * Archivo .xlsx/.csv con el formato que acepta la app (columnas Grúa/Crane, Bay, Action Name/Operacion, Mov/Duration, Seq).
  * Archivo JSON local con la configuración de cada grúa, sin inventar valores operativos:
      {"G1": {"inicio": "2025-01-10T08:00", "gmph": 25},
       "G2": {"inicio": "2025-01-10T08:00", "gmph_schedule": [["2025-01-10T08:00", "2025-01-10T15:30", 28]]}}
    Cada grúa del archivo debe figurar en el JSON. Use un nombre *.benchmark.local.json (ignorado por Git).

El archivo original solo se lee. Los resultados van a benchmarks/results/ (ignorado por Git).
"""
import argparse
import ast
import io
import json
import pathlib
import re
import sys
from datetime import datetime, time, timedelta

import pandas as pd

import _comun
from _comun import SEMILLAS, VARIANTES, ejecutar

RAIZ = pathlib.Path(__file__).resolve().parents[1]


class _Streamlit:
    """Sustituto mínimo para extraer funciones de app.py sin ejecutar su interfaz."""

    @staticmethod
    def error(mensaje):
        raise ValueError(mensaje)


def cargar_funciones_app():
    """Ejecuta solo las funciones y constantes de nivel superior de app.py (sin decoradores ni UI)."""
    arbol = ast.parse((RAIZ / "app.py").read_text(encoding="utf-8"))
    nodos = []
    for n in arbol.body:
        if isinstance(n, ast.FunctionDef) and not n.decorator_list:
            nodos.append(n)
        elif isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets):
            nodos.append(n)
    ns = {"pd": pd, "re": re, "io": io, "st": _Streamlit, "datetime": datetime, "time": time, "timedelta": timedelta}
    exec(compile(ast.Module(body=nodos, type_ignores=[]), "app.py", "exec"), ns)
    return ns


class _Archivo(io.BytesIO):
    def __init__(self, ruta):
        super().__init__(pathlib.Path(ruta).read_bytes())
        self.name = pathlib.Path(ruta).name


def cargar_config(ruta, gruas):
    datos = json.loads(pathlib.Path(ruta).read_text(encoding="utf-8"))
    faltan = [g for g in gruas if g not in datos]
    if faltan:
        raise SystemExit(f"Faltan grúas en la configuración: {faltan}")
    config = {}
    for g in gruas:
        c = datos[g]
        params = {"inicio": datetime.fromisoformat(c["inicio"]), "gmph": float(c.get("gmph", 20))}
        if c.get("gmph_schedule"):
            params["gmph_schedule"] = [
                (datetime.fromisoformat(a), datetime.fromisoformat(b), float(v)) for a, b, v in c["gmph_schedule"]
            ]
        config[g] = params
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archivo")
    ap.add_argument("--config", help="JSON local con la configuración de grúas")
    ap.add_argument("--tiempo", type=float, default=20.0)
    ap.add_argument("--holgura", type=float, default=15.0)
    ap.add_argument("--salida", default=str(RAIZ / "benchmarks" / "results" / "benchmark_real.csv"))
    args = ap.parse_args()

    if not pathlib.Path(args.archivo).is_file():
        raise SystemExit(f"No existe el archivo: {args.archivo}")
    if not args.config:
        raise SystemExit("Falta --config: indique el JSON local de grúas (ver la ayuda); no se inventa configuración.")

    app = cargar_funciones_app()
    df, _ = app["procesar_excel"](_Archivo(args.archivo))
    if df is None or "Grúa" not in df.columns:
        raise SystemExit("No se pudo interpretar el archivo con las funciones de la app.")
    config = cargar_config(args.config, sorted(df["Grúa"].astype(str).unique()))
    df["Grúa"] = df["Grúa"].astype(str)
    semilla = app["calcular_cronograma"](df, config)
    if semilla.empty:
        raise SystemExit("El cronograma semilla quedó vacío.")

    filas, errores = [], []
    for variante, kw in (VARIANTES[1], VARIANTES[4]):
        for s in SEMILLAS:
            fila_, errs = ejecutar(semilla, config, app["calcular_intervalo_tarea"], args.holgura, variante, kw, s, args.tiempo)
            filas.append(fila_)
            errores += [f"{variante} seed {s}: {e}" for e in errs]
    salida = pathlib.Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(salida, index=False, encoding="utf-8-sig")
    print(tabla[["variante", "seed", "ms_semilla", "ms_final", "factible", "precedencias_ok", "interf", "cruces",
                 "t_total_s", "cand_distintas"]].to_string(index=False))
    print(f"\nResultados: {salida}")
    if errores:
        print("ERRORES:\n" + "\n".join(errores))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
