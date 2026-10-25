"""Detección de interferencias entre grúas y propuesta de re-secuenciación.

Dos grúas interfieren cuando trabajan en el mismo bay (visual) o en bays visuales
contiguos con intervalos de tiempo que coinciden o quedan separados por menos que
la holgura de seguridad. Cubierta y bodega del mismo bay cuentan como el mismo bay.
"""
import itertools
import math
import re
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
        if not ya_activos and tarea["dur_fija"] is None and movs >= 2 * MIN_PARTE:
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


def proponer_secuencia(df_cronograma, config_gruas, intervalo_fn, holgura_min=15, tiempo_max_cpsat=TIEMPO_MAX_CPSAT):
    """Propone una secuencia sin interferencias ni cruces de grúas, con libertad para rehacerla.

    Prueba varias estrategias y elige la de menor término de operación (a igualdad, la que
    menos tareas reasigna):
    - Optimización con CP-SAT (OR-Tools) que minimiza el término de la operación, repartiendo
      y partiendo tareas entre grúas (si ortools no está instalado, se omite).
    - Despacho por orden de inicio o de término, con o sin preferencia por la grúa original.
    - Reparto por zonas desde cero: cada grúa recibe un tramo contiguo de bays, en el orden
      original de las grúas, con los tramos equilibrados por carga de trabajo.
    Cada tarea respeta el orden original dentro de su bay (descarga antes que embarque,
    cubierta antes que bodega). En todo instante la grúa de proa queda al menos 8 bays visuales
    a proa de la siguiente; una grúa que espera o ya terminó no restringe a las demás.
    El método elegido queda en `propuesta.attrs["metodo"]`.
    """
    rango = orden_original_gruas(df_cronograma)
    inicios = {g: config_gruas[g]["inicio"] for g in df_cronograma["Grúa"].unique()}
    estrategias = [("Despacho " + criterio, criterio, None) for criterio in ("inicio", "fin", "fin_sin_preferencia")]
    estrategias += [("Zonas contiguas", "fin_sin_preferencia", asignacion) for asignacion in _asignaciones_por_zonas(
        df_cronograma, config_gruas, rango
    )]

    candidatas = []
    ultimo_error = None

    def registrar(nombre, propuesta):
        if propuesta is None or detectar_cruces(propuesta, rango, inicios) \
                or not detectar_interferencias(propuesta, holgura_min).empty:
            return
        reasignadas = int((propuesta["Grúa"] != propuesta["Grúa original"]).sum())
        candidatas.append((pd.to_datetime(propuesta["Fin"]).max(), reasignadas, len(propuesta), nombre, propuesta))

    for nombre, criterio, asignacion in estrategias:
        try:
            propuesta = _construir_propuesta(
                df_cronograma, config_gruas, intervalo_fn, holgura_min, criterio, rango, asignacion
            )
        except ValueError as e:
            ultimo_error = e
            continue
        registrar(nombre, propuesta)
    if tiempo_max_cpsat:
        pista = min(candidatas, key=lambda c: c[:3])[4] if candidatas else None
        registrar("CP-SAT (OR-Tools)", _proponer_cpsat(
            df_cronograma, config_gruas, intervalo_fn, holgura_min, rango, tiempo_max_cpsat, pista
        ))
    if not candidatas:
        raise ultimo_error or ValueError(
            "No fue posible construir una secuencia sin interferencias ni cruces de grúas "
            "con la configuración actual."
        )
    mejor = min(candidatas, key=lambda c: c[:3])
    mejor[4].attrs["metodo"] = mejor[3]
    return mejor[4]


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


def _construir_propuesta(df_cronograma, config_gruas, intervalo_fn, holgura_min, criterio, rango, asignacion=None):
    margen = timedelta(minutes=holgura_min)
    gruas = sorted(df_cronograma["Grúa"].unique())
    tareas = _tareas_desde_cronograma(df_cronograma)

    cadenas = defaultdict(deque)
    for tarea in tareas:
        cadenas[tarea["clave"]].append(tarea)
    listo_cadena = {clave: pd.Timestamp.min for clave in cadenas}
    disponible = {g: pd.Timestamp(config_gruas[g]["inicio"]) for g in gruas}
    ctx = {
        "margen": margen, "rango": rango, "ocupaciones": [], "tramos": [],
        "posicion": {g: (None, disponible[g]) for g in gruas},
    }
    piezas = []

    while any(cadenas.values()):
        mejor = None
        for grua in gruas:
            for clave, cadena in cadenas.items():
                if not cadena:
                    continue
                tarea = cadena[0]
                if tarea["fijo_grua"] and tarea["fijo_grua"] != grua:
                    continue
                if asignacion and asignacion.get(clave, grua) != grua:
                    continue
                desde = max(disponible[grua], listo_cadena[clave])
                resultado = _evaluar(tarea, grua, config_gruas[grua], desde.to_pydatetime(), ctx, intervalo_fn)
                if resultado is None:
                    continue
                preferencia = 0 if (tarea["grua"] == grua or criterio == "fin_sin_preferencia") else 1
                referencia = resultado[0] if criterio == "inicio" else resultado[1]
                prioridad = (
                    referencia, preferencia, resultado[0],
                    tarea["orig_ini"], tarea["orden"], grua,
                )
                if mejor is None or prioridad < mejor[0]:
                    mejor = (prioridad, grua, clave, tarea, resultado)

        if mejor is None:
            raise ValueError(
                "No fue posible construir una secuencia sin interferencias ni cruces de grúas "
                "con la configuración actual."
            )

        _, grua, clave, tarea, (inicio, fin, duracion, movs_parte) = mejor
        inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
        piezas.append({"grua": grua, "tarea": tarea, "movs": movs_parte, "ini": inicio, "fin": fin, "dur": duracion})
        if tarea["bay"] is not None:
            ctx["ocupaciones"].append({"grua": grua, "bay": tarea["bay"], "ini": inicio, "fin": fin})
            previo_bay, previo_fin = ctx["posicion"][grua]
            ctx["tramos"].extend(_segmentos_tarea(grua, tarea["bay"], inicio, fin, previo_bay, previo_fin))
            ctx["posicion"][grua] = (tarea["bay"], fin)
        disponible[grua] = fin
        listo_cadena[clave] = fin
        if tarea["dur_fija"] is None and movs_parte < tarea["movs"]:
            tarea["movs"] -= movs_parte
        else:
            cadenas[clave].popleft()

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


def _proponer_cpsat(df_cronograma, config_gruas, intervalo_fn, holgura_min, rango, tiempo_max, pista=None):
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
        n = 1 if (t["dur_fija"] is not None or t["movs"] < 2 * MIN_PARTE_CPSAT) else min(
            max_partes, t["movs"] // MIN_PARTE_CPSAT
        )
        base, resto = divmod(t["movs"], n)
        for k in range(n):
            movs = base + (1 if k < resto else 0)
            if t["dur_fija"] is not None:
                duracion = {g: int(math.ceil(t["dur_fija"])) for g in gruas}
            else:
                duracion = {g: int(math.ceil(movs * min_por_mov[g])) for g in gruas}
            permitidas = [t["fijo_grua"]] if t["fijo_grua"] in gruas else gruas
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

    por_cadena = defaultdict(lambda: defaultdict(list))
    for i, p in enumerate(partes):
        por_cadena[p["tarea"]["clave"]][p["tarea"]["orden"]].append(i)
    precede = set()
    for ordenes in por_cadena.values():
        claves = sorted(ordenes)
        for a, b in zip(claves, claves[1:]):
            for i in ordenes[a]:
                for j in ordenes[b]:
                    modelo.Add(E[i] <= S[j])
        for a_idx, a in enumerate(claves):
            for b in claves[a_idx + 1:]:
                for i in ordenes[a]:
                    for j in ordenes[b]:
                        precede.add((i, j))

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
    solver.parameters.max_time_in_seconds = tiempo_max
    solver.parameters.num_workers = 8
    solver.parameters.random_seed = 1
    estado = solver.Solve(modelo)
    if estado not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    margen = timedelta(minutes=holgura_min)
    disponible = dict(inicios)
    ctx = {
        "margen": margen, "rango": rango, "ocupaciones": [], "tramos": [],
        "posicion": {g: (None, disponible[g]) for g in gruas},
    }
    terminos = defaultdict(list)
    piezas = []
    asignadas = [next(g for g, lit in X[i].items() if solver.Value(lit)) for i in range(len(partes))]
    secuencia = sorted(
        range(len(partes)),
        key=lambda i: (solver.Value(S[i]), solver.Value(E[i]), partes[i]["tarea"]["orden"], i),
    )
    for i in secuencia:
        tarea, grua = partes[i]["tarea"], asignadas[i]
        pendiente = dict(tarea, movs=partes[i]["movs"])
        listo = max(
            (fin for orden, fin in terminos[tarea["clave"]] if orden < tarea["orden"]),
            default=pd.Timestamp.min,
        )
        while True:
            desde = max(disponible[grua], listo)
            resultado = _evaluar(pendiente, grua, config_gruas[grua], desde.to_pydatetime(), ctx, intervalo_fn)
            if resultado is None:
                return None
            inicio, fin, duracion, movs_parte = resultado
            inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
            piezas.append({"grua": grua, "tarea": tarea, "movs": movs_parte, "ini": inicio, "fin": fin, "dur": duracion})
            if tarea["bay"] is not None:
                ctx["ocupaciones"].append({"grua": grua, "bay": tarea["bay"], "ini": inicio, "fin": fin})
                previo_bay, previo_fin = ctx["posicion"][grua]
                ctx["tramos"].extend(_segmentos_tarea(grua, tarea["bay"], inicio, fin, previo_bay, previo_fin))
                ctx["posicion"][grua] = (tarea["bay"], fin)
            disponible[grua] = fin
            terminos[tarea["clave"]].append((tarea["orden"], fin))
            if tarea["dur_fija"] is None and movs_parte < pendiente["movs"]:
                pendiente["movs"] -= movs_parte
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
            and previa["tarea"]["tipo"] == t["tipo"] and previa["tarea"]["base"] == t["base"]
            and previa["tarea"]["bay_real"] == t["bay_real"] and previa["tarea"]["zona"] == t["zona"]
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
