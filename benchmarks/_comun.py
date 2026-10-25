"""Utilidades compartidas por los benchmarks: variantes A–E y fila de resultados por corrida."""
import pathlib
import sys
import time

RAIZ = pathlib.Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import interferencias as ifc  # noqa: E402

SEMILLAS = (42, 43, 44)

VARIANTES = [
    ("A Etapa 2 sin mejora local", {"activar_alns": False, "activar_mejora_local": False}),
    ("B Mejora local inicial", {"activar_alns": False, "activar_mejora_local": True}),
    ("C ALNS directo", {
        "activar_alns": True, "activar_mejora_local_inicial": False, "activar_intensificacion_final": False,
    }),
    ("D Mejora local + ALNS", {
        "activar_alns": True, "activar_mejora_local_inicial": True, "activar_intensificacion_final": False,
    }),
    ("E Mejora local + ALNS + intensificación", {
        "activar_alns": True, "activar_mejora_local_inicial": True, "activar_intensificacion_final": True,
    }),
]


def ejecutar(df, config, intervalo_fn, holgura, variante, kw, semilla, tiempo, extra=None, ref_movimientos=None):
    """Corre proponer_secuencia y devuelve (fila, errores_de_invariantes)."""
    extra = extra or {}
    t0 = time.monotonic()
    p = ifc.proponer_secuencia(
        df, config, intervalo_fn, holgura, tiempo_limite=tiempo, random_seed=semilla, **kw, **extra
    )
    dur = time.monotonic() - t0
    a = p.attrs
    rotas = ifc.verificar_precedencias(p, df)
    total = df["Movimientos"].sum() if ref_movimientos is None else ref_movimientos
    movs_ok = bool(a["movimientos_conservados"] and p["Movimientos"].sum() == total)
    fases = a.get("tiempos_fases_seg", {})
    fila = {
        "variante": variante, "seed": semilla, "factible": a["factible"],
        "ms_semilla": a["makespan_semilla_min"], "ms_primera_factible": a["makespan_primera_factible_min"],
        "ms_inicial_candidatos": a["makespan_inicial_candidatos_min"],
        "ms_antes_alns": a["makespan_antes_alns_min"], "ms_mejor_alns": a["makespan_mejor_candidata_alns_min"],
        "ms_tras_alns": a["makespan_tras_alns_min"], "ms_tras_intensificacion": a["makespan_tras_intensificacion_min"],
        "ms_final": a["makespan_final_min"], "precedencias_ok": not rotas, "interf": a["interferencias_finales"],
        "cruces": a["cruces_finales"], "movs_ok": movs_ok, "t_total_s": round(dur, 1),
        "t_fases": ";".join(f"{k}={v}" for k, v in fases.items()),
        "tareas_originales": a["tareas_originales"], "fragmentos": a["fragmentos_finales"],
        "gruas_disp": a["gruas_disponibles"], "gruas_usadas": a["gruas_utilizadas"],
        "reasignadas": a["tareas_reasignadas"], "iter_alns": a["iteraciones_alns"],
        "cand_distintas": a["candidatas_distintas"],
        "pct_identicas": round(a["porcentaje_candidatas_identicas"], 1),
        "op_destruccion_ganador": a["operador_destruccion_ganador"],
        "op_reparacion_ganador": a["operador_reparacion_ganador"],
    }
    errores = []
    if not a["factible"] or a["interferencias_finales"] or a["cruces_finales"]:
        errores.append("solución no factible")
    if rotas:
        errores.append(f"precedencias rotas: {rotas[:3]}")
    if not movs_ok:
        errores.append("movimientos no conservados")
    if dur > tiempo + 3.0:
        errores.append(f"tiempo excedido ({dur:.1f}s)")
    if kw.get("activar_alns") and a["factible"] and a["iteraciones_alns"] == 0:
        errores.append("ALNS no exploró")
    if kw.get("activar_alns"):
        previos = [v for v in (a["makespan_inicial_candidatos_min"], a["makespan_antes_alns_min"]) if v is not None]
        if previos and a["makespan_final_min"] > min(previos) + ifc.TOLERANCIA_MAKESPAN_MIN:
            errores.append("peor que un incumbente encontrado dentro de la misma ejecución")
    return fila, errores
