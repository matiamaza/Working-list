"""Detección de interferencias entre grúas y propuesta de re-secuenciación.

Dos grúas interfieren cuando trabajan en el mismo bay (visual) o en bays visuales
contiguos con intervalos de tiempo que coinciden o quedan separados por menos que
la holgura de seguridad. Cubierta y bodega del mismo bay cuentan como el mismo bay.
"""
import contextlib
import itertools
import math
import random
import re
import time
from collections import defaultdict, deque
from datetime import timedelta

import pandas as pd

# Los bays visuales avanzan de 4 en 4 (02, 06, 10...), por lo que 4 = mismo bay o contiguo.
DISTANCIA_MAX_BAYS = 4
# Separación mínima entre grúas simultáneas: más de un bay de distancia (8 en bay visual).
SEPARACION_MIN = 2 * DISTANCIA_MAX_BAYS
MIN_PARTE = 6
MIN_PARTE_CPSAT = 20
TIEMPO_MAX_CPSAT = 20
MAX_REINTENTOS = 200
MAX_HUECOS_POR_GRUA = 8
# Dos makespan que difieren en esta cantidad de minutos o menos se consideran empatados.
TOLERANCIA_MAKESPAN_MIN = 1.0
FRACCION_TIEMPO_HEURISTICAS = 0.5
FRACCION_TIEMPO_CPSAT = 0.4
FRACCION_ALNS_HEURISTICAS = 0.35
FRACCION_ALNS_CPSAT = 0.25
# Presupuesto máximo por fase como fracción del tiempo total; lo que una fase no usa pasa a las siguientes
# y el ALNS recibe todo el tiempo restante (menos la reserva de la intensificación final).
PRESUPUESTO_FASES = {"candidatos": 0.30, "reparacion": 0.25, "mejora_inicial": 0.25, "alns": 0.30, "final": 0.05}
MAX_ITERACIONES_REPARACION = 30
PARAMETROS_PREDETERMINADOS = {
    "tiempo_limite": 20, "random_seed": 42, "workers": 4, "activar_alns": True, "porcentaje_vecindad": 0.10,
    "permitir_reasignacion": True, "permitir_division": True, "usar_simulated_annealing": True,
    "usar_cpsat_local": True, "orden_fisico_gruas": None, "activar_mejora_local_inicial": True,
    "activar_intensificacion_final": True,
}
MAX_BARRIDOS_MEJORA = 6
COLUMNAS_IDENTIDAD = ["Task_ID", "Parent_Task_ID", "Fragment_ID"]

COLUMNAS_INTERFERENCIAS = [
    "Grúa A", "Bay A", "Grúa B", "Bay B", "Desde", "Hasta", "Tipo", "Separación (min)", "Ubicación",
]


def _bay_numerico(valor):
    texto = str(valor).strip()
    return int(texto) if texto.isdigit() else None


def detectar_interferencias(df_cronograma, holgura_min=15):
    """Lista los pares de tareas de grúas distintas cercanas en bay y en tiempo."""
    if df_cronograma is None or df_cronograma.empty:
        return pd.DataFrame(columns=COLUMNAS_INTERFERENCIAS)

    margen = pd.Timedelta(minutes=holgura_min)
    tareas = df_cronograma.assign(
        _bay=df_cronograma["Bay Visual"].map(_bay_numerico),
        _ini=pd.to_datetime(df_cronograma["Inicio"]),
        _fin=pd.to_datetime(df_cronograma["Fin"]),
    )
    tareas = tareas[tareas["_bay"].notna()].sort_values("_ini")
    registros = tareas.to_dict("records")

    filas = []
    for i, a in enumerate(registros):
        for b in registros[i + 1:]:
            if b["_ini"] >= a["_fin"] + margen:
                break
            if a["Grúa"] == b["Grúa"]:
                continue
            distancia = abs(a["_bay"] - b["_bay"])
            if distancia > DISTANCIA_MAX_BAYS:
                continue

            inicio_comun = max(a["_ini"], b["_ini"])
            fin_comun = min(a["_fin"], b["_fin"])
            simultaneo = inicio_comun < fin_comun
            separacion = 0.0 if simultaneo else (inicio_comun - fin_comun).total_seconds() / 60
            filas.append({
                "Grúa A": a["Grúa"], "Bay A": a["Bay Real"],
                "Grúa B": b["Grúa"], "Bay B": b["Bay Real"],
                "Desde": min(inicio_comun, fin_comun), "Hasta": max(inicio_comun, fin_comun),
                "Tipo": "Simultáneas" if simultaneo else "Holgura insuficiente",
                "Separación (min)": round(separacion, 1),
                "Ubicación": "Mismo bay" if distancia == 0 else "Bay contiguo",
            })

    resultado = pd.DataFrame(filas, columns=COLUMNAS_INTERFERENCIAS)
    return resultado.sort_values("Desde").reset_index(drop=True)


def _tipo_tarea(categoria):
    categoria = str(categoria)
    if categoria == "Pontón":
        return "Pontón"
    if categoria.endswith(" - Descarga"):
        return "Descarga"
    if categoria.endswith(" - Embarque"):
        return "Embarque"
    return "Otro"


def _asegurar_identidad(df_cronograma):
    """Copia del cronograma con Task_ID, Parent_Task_ID y Fragment_ID; no modifica el original.

    La identidad de una tarea de la semilla es su posición en el orden estable por hora de inicio,
    de modo que no depende del índice del DataFrame. En la semilla Task_ID == Parent_Task_ID.
    """
    copia = df_cronograma.reset_index(drop=True).copy()
    copia.attrs = dict(df_cronograma.attrs)
    if all(c in copia.columns for c in COLUMNAS_IDENTIDAD):
        return copia
    orden = pd.to_datetime(copia["Inicio"]).sort_values(kind="stable").index
    ids = {indice: f"T{k:04d}" for k, indice in enumerate(orden)}
    copia["Task_ID"] = [ids[i] for i in copia.index]
    copia["Parent_Task_ID"] = copia["Task_ID"]
    copia["Fragment_ID"] = copia["Task_ID"] + ".0"
    return copia


def calcular_makespan_min(df_cronograma):
    """Makespan real: minutos entre el primer Inicio y el último Fin del cronograma."""
    if df_cronograma is None or df_cronograma.empty:
        return 0.0
    inicio = pd.to_datetime(df_cronograma["Inicio"]).min()
    fin = pd.to_datetime(df_cronograma["Fin"]).max()
    return float((fin - inicio).total_seconds() / 60)


def _regla_cadena_heredada(tareas):
    """Precedencias heredadas de la semilla: en cada bay, el orden de inicio del plan original.

    Es la única regla vigente. No incorpora descarga antes que embarque ni cubierta antes que
    bodega: esas reglas dependen de la operación real y hoy solo se respetan si la semilla las tenía.
    """
    por_cadena = defaultdict(list)
    for t in tareas:
        por_cadena[t["clave"]].append(t["id"])
    return [(a, b) for ids in por_cadena.values() for a, b in zip(ids, ids[1:])]


# Para agregar reglas explícitas en el futuro basta con añadir aquí una función
# `regla(tareas) -> [(id_predecesora, id_sucesora), ...]`; el despacho, CP-SAT, la
# reparación y la validación consumen estas aristas y no necesitan cambios.
REGLAS_PRECEDENCIA = [_regla_cadena_heredada]


def construir_precedencias(tareas, reglas=None):
    """Aristas (predecesora, sucesora) únicas: la sucesora parte cuando termina toda la predecesora."""
    aristas = []
    for regla in REGLAS_PRECEDENCIA if reglas is None else reglas:
        for par in regla(tareas):
            if par not in aristas:
                aristas.append(par)
    return aristas


def verificar_precedencias(df_resultado, df_semilla):
    """Aristas heredadas de la semilla que el resultado rompe (lista vacía = todas se cumplen).

    Verificación independiente del contexto: la predecesora debe terminar, con todos sus fragmentos,
    antes de que comience el primer fragmento de su sucesora.
    """
    semilla = _asegurar_identidad(df_semilla)
    aristas = construir_precedencias(_tareas_desde_cronograma(semilla))
    inicio = pd.to_datetime(df_resultado["Inicio"]).groupby(df_resultado["Parent_Task_ID"]).min()
    fin = pd.to_datetime(df_resultado["Fin"]).groupby(df_resultado["Parent_Task_ID"]).max()
    tolerancia = pd.Timedelta(seconds=1)
    return [(a, b) for a, b in aristas if a in fin.index and b in inicio.index and fin[a] > inicio[b] + tolerancia]


def _predecesoras(aristas):
    resultado = defaultdict(list)
    for pred, suc in aristas:
        resultado[suc].append(pred)
    return resultado


def _tareas_desde_cronograma(df_cronograma):
    df = df_cronograma.assign(_ini=pd.to_datetime(df_cronograma["Inicio"]))
    df = df.sort_values("_ini", kind="stable")
    tareas = []
    for orden, fila in enumerate(df.to_dict("records")):
        tipo = _tipo_tarea(fila["Categoría Visual"])
        bay = _bay_numerico(fila["Bay Visual"])
        grua = fila["Grúa"]
        # Un pontón sin bay numérico queda ligado a su grúa en una cadena propia.
        clave = str(bay) if bay is not None else f"General|{grua}"
        tareas.append({
            "id": f"T{orden:04d}",
            "orden": orden,
            "grua": grua,
            "tipo": tipo,
            "bay": bay,
            "clave": clave,
            "bay_real": fila["Bay Real"],
            "bay_visual": fila["Bay Visual"],
            "zona": fila["Zona"],
            "base": re.sub(r"\s*\(\d+\)\s*$", "", str(fila["Acción"])),
            "movs": int(fila["Movimientos"]),
            "dur_fija": float(fila["Duración Neta"]) if tipo == "Pontón" else None,
            "fijo_grua": grua if tipo == "Pontón" else None,
            "orig_ini": fila["_ini"],
        })
    return tareas


def orden_original_gruas(df_cronograma):
    """Posición de cada grúa en el eje de bays al inicio (0 = bay menor), según su primera tarea."""
    df = df_cronograma.assign(
        _ini=pd.to_datetime(df_cronograma["Inicio"]),
        _bay=df_cronograma["Bay Visual"].map(_bay_numerico),
    )
    df = df[df["_bay"].notna()].sort_values("_ini", kind="stable")
    primera = df.groupby("Grúa")["_bay"].first()
    gruas = sorted(df_cronograma["Grúa"].unique())
    ordenadas = sorted(gruas, key=lambda g: (primera.get(g, float("inf")), g))
    return {g: i for i, g in enumerate(ordenadas)}


def _viola_orden(lo, hi, rango_g, lo_o, hi_o, rango_o):
    """True si dos grúas simultáneas no cumplen bay(delantera) + SEPARACION_MIN <= bay(trasera)."""
    if rango_g < rango_o:
        return hi + SEPARACION_MIN > lo_o
    return lo < hi_o + SEPARACION_MIN


def _segmentos_tarea(grua, bay, inicio, fin, previo_bay=None, previo_fin=None):
    """Tramo en que una grúa restringe a las demás: solo mientras ejecuta la tarea.

    Una grúa que espera, se traslada o ya terminó se asume fuera del camino y no restringe.
    """
    return [{"grua": grua, "lo": bay, "hi": bay, "ini": inicio, "fin": fin}]


def _tramos_cronograma(df_cronograma):
    df = df_cronograma.assign(
        _bay=df_cronograma["Bay Visual"].map(_bay_numerico),
        _ini=pd.to_datetime(df_cronograma["Inicio"]),
        _fin=pd.to_datetime(df_cronograma["Fin"]),
    )
    tramos = []
    for grua, grupo in df[df["_bay"].notna()].sort_values("_ini", kind="stable").groupby("Grúa"):
        previo_bay, previo_fin = None, None
        for fila in grupo.to_dict("records"):
            tramos.extend(_segmentos_tarea(grua, fila["_bay"], fila["_ini"], fila["_fin"], previo_bay, previo_fin))
            previo_bay, previo_fin = fila["_bay"], fila["_fin"]
    return tramos


def detectar_cruces(df_cronograma, rango=None, inicios=None):
    """Pares de grúas que, en algún instante, incumplen el orden y la separación de bays original."""
    rango = rango or orden_original_gruas(df_cronograma)
    tramos = _tramos_cronograma(df_cronograma)
    cruces = set()
    for i, a in enumerate(tramos):
        for b in tramos[i + 1:]:
            if a["grua"] == b["grua"] or not (a["ini"] < b["fin"] and b["ini"] < a["fin"]):
                continue
            if _viola_orden(a["lo"], a["hi"], rango[a["grua"]], b["lo"], b["hi"], rango[b["grua"]]):
                cruces.add(tuple(sorted((a["grua"], b["grua"]), key=lambda g: rango[g])))
    return sorted(cruces)


def _conflictos(grua, bay, inicio, fin, ctx):
    if bay is None:
        return []
    margen, rango = ctx["margen"], ctx["rango"]
    conflictos = [
        o for o in ctx["ocupaciones"]
        if o["grua"] != grua
        and abs(o["bay"] - bay) <= DISTANCIA_MAX_BAYS
        and inicio < o["fin"] + margen
        and o["ini"] < fin + margen
    ]
    previo_bay, previo_fin = ctx["posicion"][grua]
    for seg in _segmentos_tarea(grua, bay, inicio, fin, previo_bay, previo_fin):
        for t in ctx["tramos"]:
            if (
                t["grua"] != grua and seg["ini"] < t["fin"] and t["ini"] < seg["fin"]
                and _viola_orden(seg["lo"], seg["hi"], rango[grua], t["lo"], t["hi"], rango[t["grua"]])
            ):
                conflictos.append(t)
    return conflictos


def _max_movs_hasta(params, inicio, limite, movs, intervalo_fn):
    """Mayor cantidad de movimientos que termina antes del límite."""
    bajo, alto, mejor = 1, movs, 0
    while bajo <= alto:
        medio = (bajo + alto) // 2
        try:
            _, fin, _ = intervalo_fn(params, inicio, medio, None)
            cabe = fin <= limite
        except ValueError:
            cabe = False
        if cabe:
            mejor, bajo = medio, medio + 1
        else:
            alto = medio - 1
    return mejor


def _evaluar(tarea, grua, params, desde, ctx, intervalo_fn):
    """Primer inicio sin interferencia; parte la tarea si hay un hueco antes del conflicto."""
    movs = tarea["movs"]
    margen = ctx["margen"]
    inicio_busqueda = desde
    for _ in range(MAX_REINTENTOS):
        inicio, fin, duracion = intervalo_fn(params, inicio_busqueda, movs, tarea["dur_fija"])
        conflictos = _conflictos(grua, tarea["bay"], inicio, fin, ctx)
        if not conflictos:
            return inicio, fin, duracion, movs

        ya_activos = [o for o in conflictos if o["ini"] - margen <= inicio]
        if ctx.get("dividir", True) and tarea["id"] not in ctx.get("no_dividir", ()) \
                and not ya_activos and tarea["dur_fija"] is None and movs >= 2 * MIN_PARTE:
            limite = min(o["ini"] for o in conflictos) - margen
            k = min(_max_movs_hasta(params, inicio_busqueda, limite, movs, intervalo_fn), movs - MIN_PARTE)
            if k >= MIN_PARTE:
                inicio_k, fin_k, duracion_k = intervalo_fn(params, inicio_busqueda, k, None)
                if not _conflictos(grua, tarea["bay"], inicio_k, fin_k, ctx):
                    return inicio_k, fin_k, duracion_k, k

        nuevo = max(o["fin"] for o in conflictos) + margen
        if nuevo <= inicio_busqueda:
            return None
        inicio_busqueda = nuevo
    return None


def validar_solucion(
    solucion, solucion_semilla, holgura, orden_gruas=None, precedencias=None,
    permitir_reasignacion=True, permitir_division=True, inicios_minimos=None,
):
    """Validación global de un cronograma frente a la semilla.

    Usa `detectar_interferencias` y `detectar_cruces` como únicas fuentes de las reglas espaciales
    (DISTANCIA_MAX_BAYS y SEPARACION_MIN), las aristas de precedencia heredadas de la semilla, y
    la semilla para los movimientos y los pontones. `factible` exige todo a la vez.
    Sin `permitir_reasignacion` cada tarea conserva su grúa; sin `permitir_division` no se parte.
    `inicios_minimos` (grúa → hora) es la hora antes de la cual una grúa no puede trabajar.
    """
    semilla = _asegurar_identidad(solucion_semilla)
    sol = _asegurar_identidad(solucion)
    errores = []
    tolerancia = pd.Timedelta(seconds=1)

    if not permitir_reasignacion:
        grua_semilla = semilla.groupby("Parent_Task_ID")["Grúa"].first()
        if (sol["Grúa"] != sol["Parent_Task_ID"].map(grua_semilla)).any():
            errores.append("Hay tareas reasignadas con la reasignación deshabilitada")
    if not permitir_division and sol["Parent_Task_ID"].duplicated().any():
        errores.append("Hay tareas divididas con la división deshabilitada")
    reglas_validas = not errores

    interferencias = len(detectar_interferencias(sol, holgura))
    try:
        pares_cruce = detectar_cruces(sol, orden_gruas or orden_original_gruas(semilla))
    except KeyError as e:
        pares_cruce = []
        errores.append(f"Grúa sin orden físico definido: {e}")
    if interferencias:
        errores.append(f"{interferencias} interferencia(s) entre grúas")
    if pares_cruce:
        errores.append(f"Cruces de grúas: {pares_cruce}")

    originales = semilla.groupby("Parent_Task_ID")["Movimientos"].sum()
    finales = sol.groupby("Parent_Task_ID")["Movimientos"].sum()
    ids = originales.index.union(finales.index)
    delta = finales.reindex(ids, fill_value=0) - originales.reindex(ids, fill_value=0)
    diferencia = int(delta.abs().sum())
    if diferencia:
        errores.append(f"Movimientos distintos a la semilla en {int((delta != 0).sum())} tarea(s)")
    if (sol["Movimientos"] < 0).any():
        errores.append("Hay movimientos negativos")
    movimientos_conservados = diferencia == 0 and not (sol["Movimientos"] < 0).any()

    if precedencias is None:
        precedencias = construir_precedencias(_tareas_desde_cronograma(semilla))
    inicio_tarea = pd.to_datetime(sol["Inicio"]).groupby(sol["Parent_Task_ID"]).min()
    fin_tarea = pd.to_datetime(sol["Fin"]).groupby(sol["Parent_Task_ID"]).max()
    rotas = [
        (a, b) for a, b in precedencias
        if a in fin_tarea.index and b in inicio_tarea.index and fin_tarea[a] > inicio_tarea[b] + tolerancia
    ]
    precedencias_validas = not rotas
    if rotas:
        errores.append(f"Precedencias heredadas rotas: {rotas[:5]}")

    pontones_validos = True
    for _, ponton in semilla[semilla["Categoría Visual"] == "Pontón"].iterrows():
        filas = sol[sol["Parent_Task_ID"] == ponton["Parent_Task_ID"]]
        if len(filas) != 1 or filas.iloc[0]["Grúa"] != ponton["Grúa"] \
                or int(filas.iloc[0]["Movimientos"]) != int(ponton["Movimientos"]):
            pontones_validos = False
    if not pontones_validos:
        errores.append("Un pontón cambió de grúa, se dividió o cambió sus movimientos")

    inicios = pd.to_datetime(sol["Inicio"])
    fines = pd.to_datetime(sol["Fin"])
    fechas_validas = not (inicios.isna().any() or fines.isna().any() or (fines < inicios).any())
    if fechas_validas and inicios_minimos:
        minimo = sol["Grúa"].map(lambda g: pd.Timestamp(inicios_minimos[g]) if g in inicios_minimos else pd.NaT)
        if (inicios < minimo - tolerancia).any():
            fechas_validas = False
            errores.append("Una grúa comienza antes de su hora de inicio configurada")
    if fechas_validas:
        ordenado = sol.assign(_i=inicios, _f=fines).sort_values("_i", kind="stable")
        for _, grupo in ordenado.groupby("Grúa"):
            if (grupo["_i"].shift(-1) < grupo["_f"] - tolerancia).any():
                fechas_validas = False
                break
    if not fechas_validas:
        errores.append("Fechas inválidas o tareas solapadas en una misma grúa")

    factible = (
        not errores and interferencias == 0 and not pares_cruce and movimientos_conservados
        and precedencias_validas and pontones_validos and fechas_validas
    )
    return {
        "factible": bool(factible),
        "interferencias": interferencias,
        "cruces": len(pares_cruce),
        "pares_cruce": pares_cruce,
        "movimientos_originales": int(originales.sum()),
        "movimientos_finales": int(finales.sum()),
        "diferencia_movimientos": diferencia,
        "movimientos_conservados": bool(movimientos_conservados),
        "precedencias_validas": precedencias_validas,
        "pontones_validos": pontones_validos,
        "fechas_validas": bool(fechas_validas),
        "reglas_validas": reglas_validas,
        "makespan_min": calcular_makespan_min(sol),
        "errores": errores,
    }


def _evaluar_plan(df, cx, metodo):
    """Valida y puntúa un cronograma. `utilizable` = conserva movimientos, precedencias, pontones y fechas."""
    val = validar_solucion(
        df, cx["semilla"], cx["holgura"], cx["rango"], cx["aristas"],
        cx["reasignar"], cx["dividir"], cx["inicios"],
    )
    termino = pd.to_datetime(df["Fin"]).groupby(df["Grúa"]).max()
    minutos = (termino - cx["t_ref"]).dt.total_seconds() / 60
    inicio_tarea = pd.to_datetime(df["Inicio"]).groupby(df["Parent_Task_ID"]).min()
    desviacion = float(
        ((inicio_tarea - cx["ini_semilla"].reindex(inicio_tarea.index)).abs().dt.total_seconds() / 60).sum()
    )
    reasignadas = int((df["Grúa"] != df["Grúa original"]).sum()) if "Grúa original" in df.columns else 0
    fragmentos = int(len(df) - df["Parent_Task_ID"].nunique())
    suma, desbalance = float(minutos.sum()), float(minutos.max() - minutos.min())
    return {
        "df": df, "metodo": metodo, "val": val, "factible": val["factible"],
        "utilizable": bool(
            val["movimientos_conservados"] and val["precedencias_validas"] and val["reglas_validas"]
            and val["pontones_validos"] and val["fechas_validas"]
        ),
        "interf": val["interferencias"], "cruces": val["cruces"], "ms": val["makespan_min"],
        "terminos": {g: float(m) for g, m in minutos.items()}, "suma": suma, "desbalance": desbalance,
        "reasignadas": reasignadas, "fragmentos": fragmentos, "desviacion": desviacion,
        "secundario": (suma, desbalance, fragmentos, reasignadas, desviacion),
    }


def _comparar_planes(a, b):
    """(a es mejor que b, motivo). Orden lexicográfico acordado; el makespan manda y nunca cede al balance."""
    if b is None:
        return True, "primera candidata"
    if a["factible"] != b["factible"]:
        return a["factible"], "factible frente a infactible" if a["factible"] else "infactible frente a factible"
    if not a["factible"]:
        mejor = (a["cruces"], a["interf"], a["ms"], a["reasignadas"], a["fragmentos"]) \
            < (b["cruces"], b["interf"], b["ms"], b["reasignadas"], b["fragmentos"])
        return mejor, "menos cruces/interferencias" if mejor else "no reduce cruces ni interferencias"
    if abs(a["ms"] - b["ms"]) > TOLERANCIA_MAKESPAN_MIN:
        return a["ms"] < b["ms"], "menor makespan" if a["ms"] < b["ms"] else "mayor makespan"
    mejor = a["secundario"] < b["secundario"]
    return mejor, "empate de makespan y mejor criterio secundario" if mejor else "empate de makespan sin mejora secundaria"


class _Incumbente:
    """Mejor solución conocida; una infactible nunca reemplaza a una factible."""

    def __init__(self):
        self.mejor = None
        self.ms_minimo = float("inf")
        self.primera_factible_ms = None
        self.version = 0

    def ofrecer(self, ev):
        mejor, motivo = _comparar_planes(ev, self.mejor)
        if mejor and ev["factible"] and ev["ms"] > self.ms_minimo + TOLERANCIA_MAKESPAN_MIN:
            return False, "supera el mejor makespan visto por más de la tolerancia"
        if not mejor:
            return False, motivo
        if ev["factible"]:
            self.ms_minimo = min(self.ms_minimo, ev["ms"])
            if self.primera_factible_ms is None:
                self.primera_factible_ms = ev["ms"]
        self.mejor = ev
        self.version += 1
        return True, motivo


@contextlib.contextmanager
def _fase(cx, limite_fase):
    """Limita temporalmente el deadline del contexto al de la fase (nunca más allá del global)."""
    previo = cx["limite"]
    cx["limite"] = min(previo, limite_fase)
    try:
        yield
    finally:
        cx["limite"] = previo


def _fallo(cx, etapa, error):
    cx["fallos"].append({"etapa": etapa, "tipo": type(error).__name__, "mensaje": str(error)})
    if cx["debug"] and not isinstance(error, (ValueError, TimeoutError)):
        raise error


def _historial(cx, fase, metodo, reparacion, tareas, ev, aceptada, motivo):
    cx["iteracion"] += 1
    cx["historial"].append({
        "iteracion": cx["iteracion"], "fase": fase, "metodo": metodo, "reparacion": reparacion,
        "tareas_afectadas": list(tareas),
        "interferencias": ev["interf"] if ev else None, "cruces": ev["cruces"] if ev else None,
        "makespan": round(ev["ms"], 1) if ev else None, "factible": ev["factible"] if ev else False,
        "aceptada": aceptada, "motivo": motivo, "tiempo_s": round(time.monotonic() - cx["t0"], 2),
    })


def _probar(cx, inc, fase, metodo, reparacion, tareas, construir):
    """Construye, valida y ofrece una candidata. Un fallo descarta solo esta candidata."""
    try:
        df = construir()
    except Exception as e:
        _fallo(cx, f"{fase}: {metodo} {reparacion}".strip(), e)
        _historial(cx, fase, metodo, reparacion, tareas, None, False, f"falló: {type(e).__name__}: {e}")
        return None
    if df is None or df.empty:
        _historial(cx, fase, metodo, reparacion, tareas, None, False, "sin solución")
        return None
    try:
        ev = _evaluar_plan(df, cx, metodo)
    except Exception as e:
        _fallo(cx, f"validación: {metodo} {reparacion}".strip(), e)
        _historial(cx, fase, metodo, reparacion, tareas, None, False, f"validación falló: {type(e).__name__}: {e}")
        return None
    if not ev["utilizable"]:
        _historial(cx, fase, metodo, reparacion, tareas, ev, False, "descartada: " + "; ".join(ev["val"]["errores"]))
        return None
    aceptada, motivo = inc.ofrecer(ev)
    _historial(cx, fase, metodo, reparacion, tareas, ev, aceptada, motivo)
    return ev


def _piezas_desde_df(df, cx):
    return [{
        "grua": f["Grúa"], "tarea": cx["tareas_por_id"][f["Parent_Task_ID"]], "movs": int(f["Movimientos"]),
        "ini": pd.Timestamp(f["Inicio"]), "fin": pd.Timestamp(f["Fin"]), "dur": float(f["Duración Neta"]),
        "frag": f["Fragment_ID"],
    } for f in df.to_dict("records")]


def _congelar(piezas, cx, t_corte):
    """Piezas que se conservan: tareas completas que empiezan antes del corte y cuyas predecesoras también."""
    ultimo_inicio = {}
    for p in piezas:
        tid = p["tarea"]["id"]
        ultimo_inicio[tid] = max(ultimo_inicio.get(tid, p["ini"]), p["ini"])
    congeladas = {tid for tid, ini in ultimo_inicio.items() if ini < t_corte}
    cambio = True
    while cambio:
        cambio = False
        for tid in list(congeladas):
            if any(p not in congeladas for p in cx["preds"].get(tid, ())):
                congeladas.discard(tid)
                cambio = True
    return [p for p in piezas if p["tarea"]["id"] in congeladas]


def _construir_variante(cx, congeladas, criterio, excluidas=None, forzadas=None):
    piezas = _despachar(
        cx["tareas"], cx["config"], cx["intervalo_fn"], cx["holgura"], criterio, cx["rango"],
        piezas_previas=congeladas, excluidas=excluidas, forzadas=forzadas,
        limite=cx["limite"], aristas=cx["aristas"],
        permitir_reasignacion=cx["reasignar"], permitir_division=cx["dividir"],
    )
    return _armar_cronograma(piezas)


def _primer_cruce(df, cx):
    """(instante, grúas) del primer incumplimiento del orden físico de grúas, o None."""
    tramos = _tramos_cronograma(df)
    mejor = None
    for i, a in enumerate(tramos):
        for b in tramos[i + 1:]:
            if a["grua"] == b["grua"] or not (a["ini"] < b["fin"] and b["ini"] < a["fin"]):
                continue
            if _viola_orden(a["lo"], a["hi"], cx["rango"][a["grua"]], b["lo"], b["hi"], cx["rango"][b["grua"]]):
                instante = max(a["ini"], b["ini"])
                if mejor is None or instante < mejor[0]:
                    mejor = (instante, {a["grua"], b["grua"]})
    return mejor


def _bloque_conflicto(df, cx, k_previas):
    """Primer conflicto del plan: instante de corte, tarea a reubicar y tareas afectadas."""
    margen = pd.Timedelta(minutes=cx["holgura"])
    interferencias = detectar_interferencias(df, cx["holgura"])
    opciones = []
    if not interferencias.empty:
        f = interferencias.iloc[0]
        opciones.append((f["Desde"], f["Hasta"], {f["Grúa A"], f["Grúa B"]}))
    cruce = _primer_cruce(df, cx)
    if cruce:
        opciones.append((cruce[0], cruce[0], cruce[1]))
    if not opciones:
        return None
    desde, hasta, gruas = min(opciones, key=lambda o: o[0])

    tabla = df.assign(_i=pd.to_datetime(df["Inicio"]), _f=pd.to_datetime(df["Fin"]))
    tabla = tabla[tabla["Grúa"].isin(gruas)].sort_values("_i", kind="stable")
    participantes = tabla[(tabla["_i"] - margen <= hasta) & (tabla["_f"] + margen >= desde)]
    if participantes.empty:
        participantes = tabla[tabla["_i"] <= hasta].tail(1)
    afectadas = list(participantes["Parent_Task_ID"])
    t_corte = participantes["_i"].min()
    for grua in gruas:
        propias = tabla[tabla["Grúa"] == grua]
        primeras = participantes[participantes["Grúa"] == grua]
        if primeras.empty or k_previas <= 0:
            continue
        previas = propias[propias["_i"] < primeras["_i"].min()].tail(k_previas)
        if not previas.empty:
            t_corte = min(t_corte, previas["_i"].min())
            afectadas += list(previas["Parent_Task_ID"])
    segunda = participantes.sort_values("_i", kind="stable").iloc[-1]
    return {
        "t_corte": t_corte, "gruas": gruas, "afectadas": sorted(set(afectadas)),
        "segunda": (segunda["Parent_Task_ID"], segunda["Grúa"]),
    }


def _variantes_reubicacion(cx, tarea_id, grua_actual):
    variantes = [
        ("voraz por inicio", {"criterio": "inicio"}),
        ("voraz por fin", {"criterio": "fin"}),
        ("voraz sin preferencia de grúa", {"criterio": "fin_sin_preferencia"}),
        ("inserción regret-2", {"criterio": "regret2"}),
    ]
    if not cx["reasignar"]:
        return variantes
    variantes.append(
        ("cambiar de grúa", {"criterio": "fin_sin_preferencia", "excluidas": {tarea_id: {grua_actual}}})
    )
    for otra in cx["gruas"]:
        if otra != grua_actual:
            variantes.append((f"forzar a {otra}", {"criterio": "fin_sin_preferencia", "forzadas": {tarea_id: otra}}))
    return variantes


def _reparar_iterativo(cx, inc, ev_actual, fase="reparación"):
    """Repara conflictos uno a uno: congela lo anterior al primer conflicto y reubica el resto.

    En cada iteración prueba varias reubicaciones (retrasar o adelantar, cambiar de grúa, mover un
    bloque pequeño, inserción voraz y regret-2) y se queda con la mejor según el orden lexicográfico.
    """
    for _ in range(MAX_ITERACIONES_REPARACION):
        if ev_actual["factible"] or time.monotonic() > cx["limite"]:
            break
        cx["iter_reparacion"] += 1
        piezas = _piezas_desde_df(ev_actual["df"], cx)
        mejor_iter = None
        for k_previas in (0, 1, 2):
            bloque = _bloque_conflicto(ev_actual["df"], cx, k_previas)
            if bloque is None:
                break
            congeladas = _congelar(piezas, cx, bloque["t_corte"])
            tarea_id, grua_actual = bloque["segunda"]
            for nombre, kwargs in _variantes_reubicacion(cx, tarea_id, grua_actual):
                if time.monotonic() > cx["limite"]:
                    break
                ev = _probar(
                    cx, inc, fase, "Reparación iterativa", f"{nombre} (bloque +{k_previas})", bloque["afectadas"],
                    lambda c=congeladas, kw=kwargs: _construir_variante(cx, c, **kw),
                )
                if ev and _comparar_planes(ev, mejor_iter)[0]:
                    mejor_iter = ev
        if mejor_iter is None or not _comparar_planes(mejor_iter, ev_actual)[0]:
            break
        ev_actual = mejor_iter
    return ev_actual


def _mejorar_local(cx, inc, rnd):
    """Mejora corta del makespan: reubica las últimas tareas de la grúa crítica y lo que sigue."""
    for _ in range(MAX_BARRIDOS_MEJORA):
        if time.monotonic() > cx["limite"] or inc.mejor is None or not inc.mejor["factible"]:
            return
        version = inc.version
        actual = inc.mejor
        piezas = _piezas_desde_df(actual["df"], cx)
        critica = max(actual["terminos"], key=lambda g: (actual["terminos"][g], g))
        de_critica = sorted((p for p in piezas if p["grua"] == critica), key=lambda p: p["ini"])
        tamanos = [k for k in (1, 2, 3, 4, 6, 8) if k <= len(de_critica)]
        rnd.shuffle(tamanos)
        for k in tamanos:
            ultimas = de_critica[-k:]
            congeladas = _congelar(piezas, cx, min(p["ini"] for p in ultimas))
            afectadas = sorted({p["tarea"]["id"] for p in ultimas})
            variantes = [
                (nombre, kw) for nombre, kw in (
                    ("voraz por fin", {"criterio": "fin"}),
                    ("voraz sin preferencia de grúa", {"criterio": "fin_sin_preferencia"}),
                    ("voraz por inicio", {"criterio": "inicio"}),
                    ("inserción regret-2", {"criterio": "regret2"}),
                )
            ]
            for tarea in {p["tarea"]["id"]: p["tarea"] for p in ultimas[-2:]}.values():
                variantes += _variantes_reubicacion(cx, tarea["id"], critica)[4:]
            rnd.shuffle(variantes)
            for nombre, kwargs in variantes:
                if time.monotonic() > cx["limite"]:
                    return
                cx["iter_mejora"] += 1
                _probar(
                    cx, inc, "mejora local", "Mejora local", f"{nombre} (últimas {k} de {critica})", afectadas,
                    lambda c=congeladas, kw=kwargs: _construir_variante(cx, c, **kw),
                )
        if inc.version == version:
            return


def resolver_orden_fisico(df_cronograma, orden_fisico_gruas=None):
    """Orden físico de las grúas en el riel (0 = proa), como dict grúa → posición.

    Acepta una lista de proa a popa o un dict grúa → posición. Si no se entrega, usa
    `orden_original_gruas` solo como aproximación. No depende de la posición inicial que adopte
    cada grúa en la nueva solución.
    """
    gruas = sorted(df_cronograma["Grúa"].unique())
    if orden_fisico_gruas is None:
        return orden_original_gruas(df_cronograma)
    if isinstance(orden_fisico_gruas, dict):
        ordenadas = sorted(orden_fisico_gruas, key=lambda g: (orden_fisico_gruas[g], g))
    else:
        ordenadas = list(orden_fisico_gruas)
    faltan = [g for g in gruas if g not in ordenadas]
    if faltan:
        raise ValueError(f"El orden físico no incluye a las grúas: {faltan}")
    ordenadas = [g for g in ordenadas if g in gruas]
    return {g: i for i, g in enumerate(ordenadas)}


def crear_contexto(
    semilla, config_gruas, intervalo_fn, holgura, rango, limite, permitir_reasignacion=True,
    permitir_division=True, debug=False, t0=None,
):
    """Contexto de optimización: la semilla (ya normalizada) como referencia y las reglas heredadas."""
    tareas = _tareas_desde_cronograma(semilla)
    aristas = construir_precedencias(tareas)
    return {
        "reasignar": bool(permitir_reasignacion), "dividir": bool(permitir_division),
        "inicios": {g: pd.Timestamp(config_gruas[g]["inicio"]) for g in semilla["Grúa"].unique()},
        "semilla": semilla, "tareas": tareas, "tareas_por_id": {t["id"]: t for t in tareas},
        "aristas": aristas, "preds": _predecesoras(aristas), "rango": rango, "config": config_gruas,
        "intervalo_fn": intervalo_fn, "holgura": holgura, "gruas": sorted({t["grua"] for t in tareas}),
        "t_ref": pd.to_datetime(semilla["Inicio"]).min(),
        "ini_semilla": pd.to_datetime(semilla["Inicio"]).groupby(semilla["Parent_Task_ID"]).min(),
        "limite": limite, "debug": debug, "t0": time.monotonic() if t0 is None else t0,
        "iteracion": 0, "historial": [], "fallos": [], "iter_reparacion": 0, "iter_mejora": 0,
    }


def proponer_secuencia(
    df_cronograma, config_gruas, intervalo_fn, holgura=15, tiempo_limite=TIEMPO_MAX_CPSAT,
    random_seed=42, workers=4, activar_reparacion_iterativa=True, activar_mejora_local=True,
    activar_alns=True, porcentaje_vecindad=0.10, permitir_reasignacion=True, permitir_division=True,
    usar_simulated_annealing=True, orden_fisico_gruas=None, debug=False,
    usar_cpsat_local=True, max_iteraciones_alns=None, activar_mejora_local_inicial=True,
    activar_intensificacion_final=True, presupuesto_fases=None,
):
    """Propone una secuencia sin interferencias ni cruces de grúas, con libertad para rehacerla.

    Candidatas (todas se validan con `validar_solucion` y cada una falla de forma aislada):
    - La propia semilla, si ya es factible.
    - Despacho voraz por inicio o por término, con o sin preferencia por la grúa original.
    - Reparto por zonas desde cero: tramos contiguos de bays, en el orden original de las grúas.
    - Optimización CP-SAT (OR-Tools) del makespan, con la mejor candidata como punto de partida
      (se omite si ortools no está instalado o no encuentra solución).
    - Reparación iterativa localizada de la semilla y, si hace falta, de la mejor candidata.
    - Mejora local corta del makespan sobre la mejor solución factible.
    Orden de comparación: factible, cero interferencias, cero cruces, movimientos conservados,
    menor makespan real (empate si difiere ≤ 1 min), menor suma de términos por grúa, menor
    desbalance, menos reasignaciones, menos fragmentación y menor desviación de la semilla.

    Las precedencias de cada bay se heredan de la semilla; no se agregan reglas operacionales.
    En todo instante la grúa de proa queda al menos 8 bays visuales a proa de la siguiente; una
    grúa que espera o ya terminó no restringe a las demás. El tiempo total (`tiempo_limite`) se
    reparte: hasta 50 % para heurísticas y CP-SAT, y el resto para reparación y mejora.
    Si no hay solución completamente factible se devuelve la mejor candidata con
    `attrs["factible"] = False`. Los metadatos quedan en `propuesta.attrs`.
    """
    t0 = time.monotonic()
    total = max(float(tiempo_limite), 0.5)
    limite_total = t0 + total
    fr = {**PRESUPUESTO_FASES, **(presupuesto_fases or {})}
    fraccion_heuristicas = fr["candidatos"] if activar_alns else FRACCION_TIEMPO_HEURISTICAS
    fraccion_cpsat = FRACCION_ALNS_CPSAT if activar_alns else FRACCION_TIEMPO_CPSAT
    limite_heuristicas = t0 + fraccion_heuristicas * total
    semilla = _asegurar_identidad(df_cronograma)
    if "Grúa original" not in semilla.columns:
        semilla["Grúa original"] = semilla["Grúa"]
    rango = resolver_orden_fisico(semilla, orden_fisico_gruas)
    cx = crear_contexto(
        semilla, config_gruas, intervalo_fn, holgura, rango, limite_total,
        permitir_reasignacion, permitir_division, debug, t0,
    )
    tareas, aristas = cx["tareas"], cx["aristas"]
    rnd = random.Random(random_seed)
    inc = _Incumbente()
    ev_semilla = None
    try:
        ev_semilla = _evaluar_plan(semilla, cx, "Semilla")
        if ev_semilla["utilizable"]:
            aceptada, motivo = inc.ofrecer(ev_semilla)
            _historial(cx, "semilla", "Semilla", "", [], ev_semilla, aceptada, motivo)
    except Exception as e:
        _fallo(cx, "evaluación de la semilla", e)

    estrategias = [("Despacho " + criterio, criterio, None) for criterio in ("inicio", "fin", "fin_sin_preferencia")]
    try:
        if not permitir_reasignacion:
            raise ValueError("Reparto por zonas omitido: la reasignación está deshabilitada.")
        estrategias += [("Zonas contiguas", "fin_sin_preferencia", asignacion) for asignacion in _asignaciones_por_zonas(
            semilla, config_gruas, rango
        )]
    except ValueError:
        pass
    except Exception as e:
        _fallo(cx, "reparto por zonas", e)
    for nombre, criterio, asignacion in estrategias:
        if time.monotonic() > limite_heuristicas and inc.mejor is not None:
            break
        _probar(
            cx, inc, "heurística", nombre, "", [],
            lambda c=criterio, a=asignacion: _armar_cronograma(_despachar(
                tareas, config_gruas, intervalo_fn, holgura, c, rango, a, limite=limite_total, aristas=aristas,
                permitir_reasignacion=permitir_reasignacion, permitir_division=permitir_division,
            )),
        )

    tiempo_cpsat = min(fraccion_cpsat * total, limite_heuristicas - time.monotonic())
    if tiempo_cpsat >= 1.0:
        pista = inc.mejor["df"] if inc.mejor is not None and inc.mejor["factible"] else None
        _probar(
            cx, inc, "CP-SAT", "CP-SAT (OR-Tools)", "", [],
            lambda: _proponer_cpsat(
                semilla, config_gruas, intervalo_fn, holgura, rango, tiempo_cpsat, pista,
                workers=workers, semilla_aleatoria=random_seed,
                permitir_reasignacion=permitir_reasignacion, permitir_division=permitir_division,
            ),
        )

    ms_fases = {"inicial": inc.mejor["ms"] if inc.mejor is not None and inc.mejor["factible"] else None}
    tiempos_fases = {}
    snapshots = []

    def _snapshot():
        if inc.mejor is not None and inc.mejor["factible"]:
            snapshots.append(inc.mejor)

    if activar_reparacion_iterativa:
        t_fase = time.monotonic()
        try:
            with _fase(cx, t_fase + fr["reparacion"] * total):
                if ev_semilla is not None and ev_semilla["utilizable"] and not ev_semilla["factible"]:
                    _reparar_iterativo(cx, inc, ev_semilla, "reparación de la semilla")
                if inc.mejor is not None and not inc.mejor["factible"]:
                    _reparar_iterativo(cx, inc, inc.mejor)
        except Exception as e:
            _fallo(cx, "reparación iterativa", e)
        tiempos_fases["reparacion"] = round(time.monotonic() - t_fase, 3)
    ms_fases["inicial"] = inc.mejor["ms"] if inc.mejor is not None and inc.mejor["factible"] else None
    _snapshot()

    estado_alns = None
    factible_ahora = inc.mejor is not None and inc.mejor["factible"]
    if not activar_alns:
        if activar_mejora_local and factible_ahora:
            try:
                _mejorar_local(cx, inc, rnd)
            except Exception as e:
                _fallo(cx, "mejora local", e)
        ms_fases["tras_mejora_inicial"] = inc.mejor["ms"] if factible_ahora else None
    elif factible_ahora:
        if activar_mejora_local and activar_mejora_local_inicial:
            t_fase = time.monotonic()
            try:
                with _fase(cx, t_fase + fr["mejora_inicial"] * total):
                    _mejorar_local(cx, inc, rnd)
            except Exception as e:
                _fallo(cx, "mejora local inicial", e)
            tiempos_fases["mejora_local_inicial"] = round(time.monotonic() - t_fase, 3)
        ms_fases["tras_mejora_inicial"] = inc.mejor["ms"]
        _snapshot()
        reserva = 0.0
        if activar_intensificacion_final:
            reserva = max(fr["final"] * total, 0.75)
            if limite_total - time.monotonic() < 3 * reserva:
                reserva = max(0.0, min(reserva, 0.2 * (limite_total - time.monotonic())))
        t_fase = time.monotonic()
        try:
            import alns
            with _fase(cx, limite_total - reserva):
                estado_alns = alns.ejecutar_alns(
                    cx, inc, rnd, porcentaje_vecindad, usar_simulated_annealing, usar_cpsat_local,
                    max_iteraciones_alns,
                )
        except Exception as e:
            _fallo(cx, "ALNS", e)
        tiempos_fases["alns"] = round(time.monotonic() - t_fase, 3)
        ms_fases["tras_alns"] = inc.mejor["ms"]
        _snapshot()
        if activar_intensificacion_final and limite_total - time.monotonic() >= 0.5:
            t_fase = time.monotonic()
            try:
                with _fase(cx, t_fase + max(fr["final"] * total, 0.75)):
                    _mejorar_local(cx, inc, rnd)
            except Exception as e:
                _fallo(cx, "intensificación final", e)
            tiempos_fases["intensificacion_final"] = round(time.monotonic() - t_fase, 3)
        ms_fases["tras_intensificacion_final"] = inc.mejor["ms"]
        _snapshot()

    mejor = inc.mejor
    for candidato in snapshots:
        if mejor is None or _comparar_planes(candidato, mejor)[0]:
            mejor = candidato
    if mejor is None:
        mejor = ev_semilla
    if mejor is not None:
        propuesta = mejor["df"].reset_index(drop=True).copy()
        metodo = mejor["metodo"] if mejor["factible"] else f"{mejor['metodo']} (no completamente factible)"
    else:
        propuesta = semilla.copy()
        metodo = "Semilla (no se pudo evaluar)"
    ms_semilla = calcular_makespan_min(semilla)
    ms_final = calcular_makespan_min(propuesta)
    mejora = ms_semilla - ms_final
    propuesta.attrs.update({
        "metodo": metodo,
        "factible": bool(mejor["factible"]) if mejor is not None else False,
        "makespan_semilla_min": ms_semilla,
        "makespan_primera_factible_min": inc.primera_factible_ms,
        "makespan_final_min": ms_final,
        "mejora_min": mejora,
        "mejora_porcentual": 100 * mejora / ms_semilla if ms_semilla else 0.0,
        "interferencias_semilla": ev_semilla["interf"] if ev_semilla else None,
        "interferencias_finales": mejor["interf"] if mejor else None,
        "cruces_finales": mejor["cruces"] if mejor else None,
        "movimientos_conservados": bool(mejor["val"]["movimientos_conservados"]) if mejor else False,
        "terminos_por_grua_min": mejor["terminos"] if mejor else {},
        "desbalance_min": mejor["desbalance"] if mejor else None,
        "tareas_reasignadas": mejor["reasignadas"] if mejor else None,
        "fragmentos_creados": mejor["fragmentos"] if mejor else None,
        "tiempo_total_seg": round(time.monotonic() - t0, 2),
        "iteraciones_reparacion": cx["iter_reparacion"],
        "iteraciones_mejora": cx["iter_mejora"],
        "random_seed": random_seed,
        "historial": cx["historial"],
        "fallos": cx["fallos"],
    })
    disponibles = sorted(config_gruas) if hasattr(config_gruas, "keys") else sorted(cx["gruas"])
    usadas = sorted(set(propuesta["Grúa"].dropna())) if "Grúa" in propuesta else []
    propuesta.attrs.update({
        "alns_activado": estado_alns is not None and getattr(estado_alns, "iniciado", False),
        "iteraciones_alns": 0, "candidatas_alns": 0, "candidatas_factibles_alns": 0,
        "candidatas_aceptadas_alns": 0, "mejoras_alns": 0, "operador_destruccion_ganador": None,
        "operador_reparacion_ganador": None, "pesos_destruccion": {}, "pesos_reparacion": {},
        "estadisticas_operadores": {}, "temperatura_final": None, "reinicios_alns": 0,
        "gruas_disponibles": len(disponibles), "gruas_utilizadas": len(usadas),
        "gruas_sin_tareas": [g for g in disponibles if g not in usadas],
        "fragmentos_finales": int(len(propuesta)),
        "tareas_originales": int(propuesta["Parent_Task_ID"].nunique()) if "Parent_Task_ID" in propuesta else None,
        "precedencias_validas": bool(mejor["val"]["precedencias_validas"]) if mejor else False,
        "makespan_inicial_candidatos_min": ms_fases.get("inicial"),
        "makespan_antes_alns_min": ms_fases.get("tras_mejora_inicial"),
        "makespan_tras_alns_min": ms_fases.get("tras_alns"),
        "makespan_mejor_candidata_alns_min": None,
        "makespan_tras_intensificacion_min": ms_fases.get("tras_intensificacion_final"),
        "tiempos_fases_seg": tiempos_fases,
        "estadisticas_diversidad": {}, "estadisticas_cp_local": {},
        "candidatas_distintas": 0, "porcentaje_candidatas_identicas": 0.0,
        "activar_mejora_local_inicial": bool(activar_mejora_local_inicial),
        "activar_intensificacion_final": bool(activar_intensificacion_final),
    })
    if estado_alns is not None:
        resumen = estado_alns.resumen()
        propuesta.attrs.update(resumen)
        propuesta.attrs["makespan_mejor_candidata_alns_min"] = resumen["makespan_mejor_candidata_alns"]
        propuesta.attrs["candidatas_distintas"] = resumen["estadisticas_diversidad"]["candidatas_distintas"]
        propuesta.attrs["porcentaje_candidatas_identicas"] = resumen["estadisticas_diversidad"][
            "porcentaje_candidatas_identicas"
        ]
        propuesta.attrs["alns_activado"] = bool(getattr(estado_alns, "iniciado", False))
    return propuesta


def _asignaciones_por_zonas(df_cronograma, config_gruas, rango, max_candidatas=8):
    """Repartos de bays en tramos contiguos, uno por grúa y respetando el orden original."""
    tareas = _tareas_desde_cronograma(df_cronograma)
    carga, forzada = defaultdict(float), {}
    for t in tareas:
        if t["bay"] is None:
            continue
        carga[t["bay"]] += t["movs"] if t["dur_fija"] is None else 0
        if t["fijo_grua"]:
            forzada[t["bay"]] = t["fijo_grua"]
    bays = sorted(carga)
    gruas = sorted(config_gruas, key=lambda g: rango[g])
    if len(bays) < 2 or len(gruas) < 2:
        return []

    n = len(gruas)
    gmph = [max(float(config_gruas[g]["gmph"]), 1e-9) for g in gruas]
    acumulado = [0.0]
    for b in bays:
        acumulado.append(acumulado[-1] + carga[b])

    def puntuar(cortes):
        limites = (0, *cortes, len(bays))
        mayor = 0.0
        for i in range(n):
            mayor = max(mayor, (acumulado[limites[i + 1]] - acumulado[limites[i]]) / gmph[i])
        penalizacion = 0.0
        for c in cortes:
            if 0 < c < len(bays) and bays[c] - bays[c - 1] <= DISTANCIA_MAX_BAYS:
                penalizacion += min(carga[bays[c]], carga[bays[c - 1]]) / max(gmph)
        return mayor + 0.5 * penalizacion

    if math.comb(len(bays) + n - 1, n - 1) > 200_000:
        return []
    puntuados = sorted(
        (puntuar(c), c) for c in itertools.combinations_with_replacement(range(len(bays) + 1), n - 1)
    )

    resultado = []
    for _, cortes in puntuados:
        limites = (0, *cortes, len(bays))
        asignacion, valida = {}, True
        for i, grua in enumerate(gruas):
            for b in bays[limites[i]:limites[i + 1]]:
                if forzada.get(b, grua) != grua:
                    valida = False
                asignacion[str(b)] = grua
        if valida and asignacion not in resultado:
            resultado.append(asignacion)
        if len(resultado) >= max_candidatas:
            break
    return resultado


def _nueva_pieza(grua, tarea, movs, ini, fin, dur, fragmentos):
    """Pieza de plan: parte de una tarea, con identificador de fragmento único."""
    numero = fragmentos[tarea["id"]]
    fragmentos[tarea["id"]] += 1
    return {
        "grua": grua, "tarea": tarea, "movs": movs, "ini": ini, "fin": fin, "dur": dur,
        "frag": f"{tarea['id']}.{numero}",
    }


def _registrar_pieza(ctx, disponible, pieza):
    tarea, grua = pieza["tarea"], pieza["grua"]
    ctx.setdefault("propias", {}).setdefault(grua, []).append((pieza["ini"], pieza["fin"]))
    if tarea["bay"] is not None:
        ctx["ocupaciones"].append({"grua": grua, "bay": tarea["bay"], "ini": pieza["ini"], "fin": pieza["fin"]})
        previo_bay, previo_fin = ctx["posicion"][grua]
        ctx["tramos"].extend(_segmentos_tarea(grua, tarea["bay"], pieza["ini"], pieza["fin"], previo_bay, previo_fin))
        ctx["posicion"][grua] = (tarea["bay"], pieza["fin"])
    disponible[grua] = max(disponible[grua], pieza["fin"])


def _gruas_permitidas(tarea, gruas, excluidas, forzadas, reasignar=True):
    if tarea["fijo_grua"]:
        return [tarea["fijo_grua"]] if tarea["fijo_grua"] in gruas else []
    if not reasignar:
        return [tarea["grua"]]
    permitidas = list(gruas)
    forzada = (forzadas or {}).get(tarea["id"])
    if forzada in permitidas:
        return [forzada]
    restantes = [g for g in permitidas if g not in (excluidas or {}).get(tarea["id"], ())]
    return restantes or permitidas


def _despachar(
    tareas, config_gruas, intervalo_fn, holgura_min, criterio, rango, asignacion=None,
    piezas_previas=(), excluidas=None, forzadas=None, limite=None, aristas=None,
    permitir_reasignacion=True, permitir_division=True, huecos=False, prioridades=None, sin_dividir=(),
):
    """Despacho voraz de las tareas no congeladas; no modifica `tareas` ni `piezas_previas`.

    `piezas_previas` son piezas ya fijas (tareas completas): ocupan su grúa y sus bays y satisfacen
    precedencias. `excluidas`/`forzadas` (id de tarea → grúas / grúa) orientan la asignación.
    Criterios: "inicio", "fin", "fin_sin_preferencia", "regret2" (primero la tarea con mayor
    pérdida si no se le da su mejor grúa) y "prioridad" (menor `prioridades[id]` primero).
    Con `huecos` las tareas pueden insertarse en huecos libres de la grúa, no solo al final.
    `sin_dividir` son ids de tareas que no se parten. Retorna la lista completa de piezas.
    """
    margen = timedelta(minutes=holgura_min)
    gruas = sorted({t["grua"] for t in tareas})
    aristas = construir_precedencias(tareas) if aristas is None else aristas
    preds = _predecesoras(aristas)

    disponible = {g: pd.Timestamp(config_gruas[g]["inicio"]) for g in gruas}
    inicio_cfg = dict(disponible)
    ctx = {
        "margen": margen, "rango": rango, "ocupaciones": [], "tramos": [],
        "posicion": {g: (None, disponible[g]) for g in gruas}, "dividir": permitir_division,
        "no_dividir": set(sin_dividir),
    }
    piezas = []
    fragmentos = defaultdict(int)
    fin_tarea, completas = {}, set()
    for pieza in sorted(piezas_previas, key=lambda p: (p["ini"], p["grua"])):
        piezas.append(pieza)
        _registrar_pieza(ctx, disponible, pieza)
        tid = pieza["tarea"]["id"]
        fin_tarea[tid] = max(fin_tarea.get(tid, pieza["fin"]), pieza["fin"])
        completas.add(tid)
        fragmentos[tid] = max(fragmentos[tid], 1)

    pendientes = {t["id"]: t for t in tareas if t["id"] not in completas}
    restante = {tid: t["movs"] for tid, t in pendientes.items()}
    listo_propio = {}

    while pendientes:
        if limite is not None and time.monotonic() > limite:
            raise TimeoutError("Se agotó el tiempo disponible del despacho.")
        opciones = defaultdict(list)
        for tid, tarea in pendientes.items():
            if any(p not in completas for p in preds.get(tid, ())):
                continue
            listo = max([fin_tarea[p] for p in preds.get(tid, ())] + [listo_propio.get(tid, pd.Timestamp.min)])
            vista = dict(tarea, movs=restante[tid])
            for grua in _gruas_permitidas(tarea, gruas, excluidas, forzadas, permitir_reasignacion):
                if asignacion and asignacion.get(tarea["clave"], grua) != grua:
                    continue
                if huecos:
                    base = max(inicio_cfg[grua], listo)
                    propias = ctx.get("propias", {}).get(grua, [])
                    candidatos = [base] + sorted({f for _, f in propias if f > base})[:MAX_HUECOS_POR_GRUA]
                else:
                    candidatos = [max(disponible[grua], listo)]
                mejor_grua = None
                for desde in candidatos:
                    resultado = _evaluar(vista, grua, config_gruas[grua], desde.to_pydatetime(), ctx, intervalo_fn)
                    if resultado is None:
                        continue
                    if huecos and any(
                        pd.Timestamp(resultado[0]) < f and i < pd.Timestamp(resultado[1])
                        for i, f in ctx.get("propias", {}).get(grua, [])
                    ):
                        continue
                    preferencia = 0 if (tarea["grua"] == grua or criterio == "fin_sin_preferencia") else 1
                    referencia = resultado[0] if criterio == "inicio" else resultado[1]
                    prioridad = (referencia, preferencia, resultado[0], tarea["orig_ini"], tarea["orden"], grua)
                    if mejor_grua is None or prioridad < mejor_grua[0]:
                        mejor_grua = (prioridad, grua, vista, resultado)
                if mejor_grua is not None:
                    opciones[tid].append(mejor_grua)

        if not opciones:
            raise ValueError(
                "No fue posible construir una secuencia sin interferencias ni cruces de grúas "
                "con la configuración actual."
            )
        mejores = {tid: sorted(lista, key=lambda o: o[0]) for tid, lista in opciones.items()}
        if criterio == "regret2":
            def arrepentimiento(tid):
                lista = mejores[tid]
                if len(lista) < 2:
                    return float("inf")
                return (lista[1][0][0] - lista[0][0][0]).total_seconds()
            tid = min(mejores, key=lambda t: (-arrepentimiento(t), mejores[t][0][0]))
        elif criterio == "prioridad":
            tid = min(mejores, key=lambda t: ((prioridades or {}).get(t, ()), mejores[t][0][0]))
        else:
            tid = min(mejores, key=lambda t: mejores[t][0][0])
        _, grua, vista, (inicio, fin, duracion, movs_parte) = mejores[tid][0]
        tarea = pendientes[tid]

        inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
        pieza = _nueva_pieza(grua, tarea, movs_parte, inicio, fin, duracion, fragmentos)
        piezas.append(pieza)
        _registrar_pieza(ctx, disponible, pieza)
        listo_propio[tid] = fin
        fin_tarea[tid] = max(fin_tarea.get(tid, fin), fin)
        if tarea["dur_fija"] is None and movs_parte < restante[tid]:
            restante[tid] -= movs_parte
        else:
            completas.add(tid)
            del pendientes[tid]
    return piezas


def _construir_propuesta(df_cronograma, config_gruas, intervalo_fn, holgura_min, criterio, rango, asignacion=None):
    tareas = _tareas_desde_cronograma(df_cronograma)
    piezas = _despachar(tareas, config_gruas, intervalo_fn, holgura_min, criterio, rango, asignacion)
    return _armar_cronograma(piezas)


def _pista_partes(partes, plan):
    """Grúa e inicio que un plan existente da a cada parte del modelo, para arrancar la optimización."""
    colas = defaultdict(deque)
    for fila in plan.sort_values("Inicio", kind="stable").to_dict("records"):
        base = re.sub(r"\s*\(\d+\)\s*$", "", str(fila["Acción"]))
        colas[(fila["Bay Real"], fila["Zona"], base)].append({
            "grua": fila["Grúa"], "ini": pd.Timestamp(fila["Inicio"]), "fin": pd.Timestamp(fila["Fin"]),
            "movs": int(fila["Movimientos"]), "usados": 0,
        })
    resultado = [None] * len(partes)
    por_tarea = defaultdict(list)
    for i, p in enumerate(partes):
        por_tarea[p["tarea"]["orden"]].append(i)
    for orden in sorted(por_tarea):
        indices = por_tarea[orden]
        t = partes[indices[0]]["tarea"]
        cola = colas.get((t["bay_real"], t["zona"], t["base"] or "Contexto"))
        for i in indices:
            if not cola:
                break
            fila = cola[0]
            if fila["movs"] <= 0:
                resultado[i] = (fila["grua"], fila["ini"])
                cola.popleft()
                continue
            avance = (fila["fin"] - fila["ini"]) * (fila["usados"] / fila["movs"])
            resultado[i] = (fila["grua"], fila["ini"] + avance)
            fila["usados"] += partes[i]["movs"]
            if fila["usados"] >= fila["movs"]:
                cola.popleft()
    return resultado


def _proponer_cpsat(
    df_cronograma, config_gruas, intervalo_fn, holgura_min, rango, tiempo_max, pista=None,
    workers=4, semilla_aleatoria=42, permitir_reasignacion=True, permitir_division=True,
):
    """Optimiza con CP-SAT (OR-Tools) la asignación y el orden de tareas para minimizar el término.

    El modelo decide qué grúa hace cada parte de cada tarea y cuándo, con estas reglas:
    - Cada parte la hace una sola grúa y una grúa hace una cosa a la vez.
    - Dentro de un bay, cada tarea espera a que termine la anterior (descarga antes que embarque).
    - Si dos partes están a menos de 8 bays visuales, o la grúa de proa quedaría a popa de la de popa,
      no pueden ejecutarse a la vez; a 4 bays o menos de distancia, entre grúas distintas se agrega la holgura.
    El resultado es solo una guía: el plan final se arma con las reglas exactas (descansos, GMPH por turno,
    desplazamientos), así que siempre cumple las restricciones. Retorna None si no hay solución.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None

    gruas = sorted(df_cronograma["Grúa"].unique(), key=lambda g: rango[g])
    if len(gruas) < 2:
        return None
    tareas = _tareas_desde_cronograma(df_cronograma)
    inicios = {g: pd.Timestamp(config_gruas[g]["inicio"]) for g in gruas}
    t0 = min(inicios.values())
    desfase = {g: int((inicios[g] - t0).total_seconds() // 60) for g in gruas}

    total = sum(t["movs"] for t in tareas if t["dur_fija"] is None)
    referencia = max(50, total // len(gruas))
    min_por_mov = {}
    for g in gruas:
        try:
            _, fin, _ = intervalo_fn(config_gruas[g], inicios[g].to_pydatetime(), referencia, None)
            min_por_mov[g] = max((pd.Timestamp(fin) - inicios[g]).total_seconds() / 60 / referencia, 1e-3)
        except ValueError:
            min_por_mov[g] = 60 / max(float(config_gruas[g]["gmph"]), 1e-9)

    max_partes = 4 if len(tareas) <= 60 else 2
    partes = []
    for t in tareas:
        n = 1 if (not permitir_division or t["dur_fija"] is not None or t["movs"] < 2 * MIN_PARTE_CPSAT) else min(
            max_partes, t["movs"] // MIN_PARTE_CPSAT
        )
        base, resto = divmod(t["movs"], n)
        for k in range(n):
            movs = base + (1 if k < resto else 0)
            if t["dur_fija"] is not None:
                duracion = {g: int(math.ceil(t["dur_fija"])) for g in gruas}
            else:
                duracion = {g: int(math.ceil(movs * min_por_mov[g])) for g in gruas}
            if t["fijo_grua"] in gruas:
                permitidas = [t["fijo_grua"]]
            else:
                permitidas = gruas if permitir_reasignacion else [t["grua"]]
            partes.append({"tarea": t, "movs": movs, "dur": duracion, "permitidas": permitidas})

    horizonte = max(desfase.values()) + sum(max(p["dur"].values()) for p in partes) + holgura_min * len(partes) + 1
    modelo = cp_model.CpModel()
    S, E, X, R = [], [], [], []
    intervalos = {g: [] for g in gruas}
    for i, p in enumerate(partes):
        inicio_v = modelo.NewIntVar(0, horizonte, f"s{i}")
        fin_v = modelo.NewIntVar(0, horizonte, f"e{i}")
        x = {g: modelo.NewBoolVar(f"x{i}_{g}") for g in p["permitidas"]}
        modelo.AddExactlyOne(x.values())
        modelo.Add(fin_v == inicio_v + sum(p["dur"][g] * x[g] for g in x))
        for g, lit in x.items():
            modelo.Add(inicio_v >= desfase[g]).OnlyEnforceIf(lit)
            intervalos[g].append(modelo.NewOptionalIntervalVar(inicio_v, p["dur"][g], inicio_v + p["dur"][g], lit, f"i{i}_{g}"))
        r = modelo.NewIntVar(0, len(gruas) - 1, f"r{i}")
        modelo.Add(r == sum(rango[g] * x[g] for g in x))
        S.append(inicio_v); E.append(fin_v); X.append(x); R.append(r)
    for g in gruas:
        modelo.AddNoOverlap(intervalos[g])

    por_tarea = defaultdict(list)
    for i, p in enumerate(partes):
        por_tarea[p["tarea"]["id"]].append(i)
    aristas = construir_precedencias(tareas)
    preds = _predecesoras(aristas)
    sucesoras = defaultdict(list)
    for a, b in aristas:
        sucesoras[a].append(b)
        for i in por_tarea[a]:
            for j in por_tarea[b]:
                modelo.Add(E[i] <= S[j])
    precede = set()
    for origen in por_tarea:
        visitados, pila = set(), list(sucesoras[origen])
        while pila:
            destino = pila.pop()
            if destino not in visitados:
                visitados.add(destino)
                pila.extend(sucesoras[destino])
        for destino in visitados:
            precede.update((i, j) for i in por_tarea[origen] for j in por_tarea[destino])

    con_bay = [i for i, p in enumerate(partes) if p["tarea"]["bay"] is not None]
    for pos, i in enumerate(con_bay):
        for j in con_bay[pos + 1:]:
            bay_i, bay_j = partes[i]["tarea"]["bay"], partes[j]["tarea"]["bay"]
            distancia = abs(bay_i - bay_j)
            ordenadas = (i, j) in precede or (j, i) in precede
            if distancia < SEPARACION_MIN:
                antes = modelo.NewBoolVar(f"a{i}_{j}")
                if not ordenadas:
                    modelo.Add(E[i] <= S[j]).OnlyEnforceIf(antes)
                    modelo.Add(E[j] <= S[i]).OnlyEnforceIf(antes.Not())
                if distancia <= DISTANCIA_MAX_BAYS and holgura_min > 0:
                    distintas = modelo.NewBoolVar(f"d{i}_{j}")
                    modelo.Add(R[i] != R[j]).OnlyEnforceIf(distintas)
                    modelo.Add(R[i] == R[j]).OnlyEnforceIf(distintas.Not())
                    modelo.Add(E[i] + holgura_min <= S[j]).OnlyEnforceIf([antes, distintas])
                    modelo.Add(E[j] + holgura_min <= S[i]).OnlyEnforceIf([antes.Not(), distintas])
            elif not ordenadas:
                popa, proa = (i, j) if bay_i > bay_j else (j, i)
                correcto = modelo.NewBoolVar(f"o{i}_{j}")
                modelo.Add(R[popa] > R[proa]).OnlyEnforceIf(correcto)
                modelo.Add(R[popa] <= R[proa]).OnlyEnforceIf(correcto.Not())
                antes = modelo.NewBoolVar(f"b{i}_{j}")
                modelo.Add(E[i] <= S[j]).OnlyEnforceIf([antes, correcto.Not()])
                modelo.Add(E[j] <= S[i]).OnlyEnforceIf([antes.Not(), correcto.Not()])

    makespan = modelo.NewIntVar(0, horizonte, "makespan")
    termino_grua = {g: modelo.NewIntVar(0, horizonte, f"f{g}") for g in gruas}
    for i in range(len(partes)):
        modelo.Add(makespan >= E[i])
        for g, lit in X[i].items():
            modelo.Add(termino_grua[g] >= E[i]).OnlyEnforceIf(lit)
    modelo.Minimize(100 * makespan + sum(termino_grua.values()))

    if pista is not None:
        for i, dato in enumerate(_pista_partes(partes, pista)):
            if dato is None or dato[0] not in X[i]:
                continue
            grua_pista, inicio_pista = dato
            modelo.AddHint(S[i], min(max(0, int((inicio_pista - t0).total_seconds() // 60)), horizonte))
            for g, lit in X[i].items():
                modelo.AddHint(lit, 1 if g == grua_pista else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(float(tiempo_max), 0.1)
    solver.parameters.num_workers = max(1, int(workers))
    solver.parameters.random_seed = int(semilla_aleatoria)
    estado = solver.Solve(modelo)
    if estado not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    margen = timedelta(minutes=holgura_min)
    disponible = dict(inicios)
    ctx = {
        "margen": margen, "rango": rango, "ocupaciones": [], "tramos": [],
        "posicion": {g: (None, disponible[g]) for g in gruas}, "dividir": permitir_division,
    }
    fin_tarea, fragmentos = {}, defaultdict(int)
    piezas = []
    asignadas = [next(g for g, lit in X[i].items() if solver.Value(lit)) for i in range(len(partes))]
    secuencia = sorted(
        range(len(partes)),
        key=lambda i: (solver.Value(S[i]), solver.Value(E[i]), partes[i]["tarea"]["orden"], i),
    )
    for i in secuencia:
        tarea, grua = partes[i]["tarea"], asignadas[i]
        pendiente = dict(tarea, movs=partes[i]["movs"])
        listo = max((fin_tarea.get(p, pd.Timestamp.min) for p in preds.get(tarea["id"], ())), default=pd.Timestamp.min)
        while True:
            desde = max(disponible[grua], listo)
            resultado = _evaluar(pendiente, grua, config_gruas[grua], desde.to_pydatetime(), ctx, intervalo_fn)
            if resultado is None:
                return None
            inicio, fin, duracion, movs_parte = resultado
            inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
            pieza = _nueva_pieza(grua, tarea, movs_parte, inicio, fin, duracion, fragmentos)
            piezas.append(pieza)
            _registrar_pieza(ctx, disponible, pieza)
            fin_tarea[tarea["id"]] = max(fin_tarea.get(tarea["id"], fin), fin)
            if tarea["dur_fija"] is None and movs_parte < pendiente["movs"]:
                pendiente["movs"] -= movs_parte
                listo = max(listo, fin)
                continue
            break
    return _armar_cronograma(piezas)


def _armar_cronograma(piezas):
    piezas = sorted(piezas, key=lambda p: (p["grua"], p["ini"]))
    unidas = []
    for pieza in piezas:
        previa = unidas[-1] if unidas else None
        t = pieza["tarea"]
        if (
            previa and previa["grua"] == pieza["grua"] and t["tipo"] != "Pontón"
            and previa["tarea"]["id"] == t["id"]
            and pieza["ini"] <= previa["fin"] + pd.Timedelta(minutes=1)
        ):
            previa["movs"] += pieza["movs"]
            previa["fin"] = pieza["fin"]
            previa["dur"] += pieza["dur"]
            continue
        unidas.append({**pieza})

    filas = []
    for p in unidas:
        t = p["tarea"]
        movs = t["movs"] if t["tipo"] == "Pontón" else p["movs"]
        categoria = t["tipo"] if t["tipo"] in ("Pontón", "Otro") else f"{p['grua']} - {t['tipo']}"
        filas.append({
            "Grúa": p["grua"], "Bay Real": t["bay_real"], "Bay Visual": t["bay_visual"],
            "Acción": f"{t['base'] or 'Contexto'} ({int(movs)})", "Movimientos": int(movs),
            "Inicio": p["ini"], "Fin": p["fin"], "Duración Neta": round(p["dur"], 1),
            "Categoría Visual": categoria, "Zona": t["zona"], "Grúa original": t["grua"],
            "Task_ID": t["id"], "Parent_Task_ID": t["id"], "Fragment_ID": p["frag"],
        })
    return pd.DataFrame(filas)


def comparar_cronogramas(df_original, df_propuesta):
    """Tabla de fin y movimientos por grúa, original contra propuesta."""
    filas = []
    for grua in sorted(set(df_original["Grúa"]) | set(df_propuesta["Grúa"])):
        o = df_original[df_original["Grúa"] == grua]
        p = df_propuesta[df_propuesta["Grúa"] == grua]
        fin_o = pd.to_datetime(o["Fin"]).max() if not o.empty else pd.NaT
        fin_p = pd.to_datetime(p["Fin"]).max() if not p.empty else pd.NaT
        filas.append(_fila_comparacion(grua, fin_o, fin_p, o["Movimientos"].sum(), p["Movimientos"].sum()))
    filas.append(_fila_comparacion(
        "Operación completa",
        pd.to_datetime(df_original["Fin"]).max(), pd.to_datetime(df_propuesta["Fin"]).max(),
        df_original["Movimientos"].sum(), df_propuesta["Movimientos"].sum(),
    ))
    return pd.DataFrame(filas)


def _fila_comparacion(nombre, fin_o, fin_p, movs_o, movs_p):
    delta = (fin_p - fin_o).total_seconds() / 60 if pd.notna(fin_o) and pd.notna(fin_p) else None
    formato = lambda f: f.strftime("%d/%m %H:%M") if pd.notna(f) else "-"
    return {
        "Grúa": nombre, "Fin original": formato(fin_o), "Fin propuesta": formato(fin_p),
        "Variación (min)": round(delta) if delta is not None else None,
        "Mov. original": int(movs_o), "Mov. propuesta": int(movs_p),
    }


def construir_colas_trabajo(df_propuesta):
    """Cola ordenada de cada grúa, lista para copiar al TOS."""
    colas = {}
    for grua, grupo in df_propuesta.groupby("Grúa"):
        grupo = grupo.sort_values("Inicio")
        filas = []
        for numero, fila in enumerate(grupo.to_dict("records"), start=1):
            origen = fila.get("Grúa original", grua)
            filas.append({
                "N°": numero,
                "Bay": fila["Bay Real"],
                "Zona": fila["Zona"],
                "Acción": re.sub(r"\s*\(\d+\)\s*$", "", fila["Acción"]),
                "Movimientos": fila["Movimientos"],
                "Inicio": fila["Inicio"].strftime("%d/%m %H:%M"),
                "Fin": fila["Fin"].strftime("%d/%m %H:%M"),
                "Nota": f"Reasignada desde {origen}" if origen != grua else "",
            })
        colas[grua] = pd.DataFrame(filas)
    return colas
