"""ALNS (Adaptive Large Neighborhood Search) para reducir el makespan de una solución factible.

El módulo no usa Streamlit. Parte de una solución factible, libera una vecindad de tareas
(operador de destrucción), la reinserta (operador de reparación), reconstruye los tiempos reales con
`calcular_intervalo_tarea`, valida globalmente con `validar_solucion` y acepta o rechaza la
candidata. La mejor solución factible se conserva siempre en el incumbente de `interferencias`.

La semilla es solo punto de partida, referencia de identidad, de movimientos y de precedencias
heredadas: nada obliga a conservar sus horarios, grúas, bays iniciales ni primera tarea.
"""
import math
import time
from collections import Counter, OrderedDict, defaultdict

import pandas as pd

import interferencias as ifc

MINIMO_TAREAS = 2
MAXIMO_NORMAL = 0.20
MAXIMO_DIVERSIFICACION = 0.30
PESO_MINIMO = 0.05
PUNTOS = {"mejor": 8.0, "mejora": 5.0, "aceptada": 2.0, "factible": 1.0, "infactible": 0.0, "fallo": 0.0}


class FalloReparacion(Exception):
    """La reparación no produjo una candidata; descarta solo esa candidata."""

    def __init__(self, mensaje, estado=None):
        super().__init__(mensaje)
        self.estado = estado


# --- lectura de una solución (sin mutarla) --------------------------------------------------

def sucesoras(cx):
    if "sucesoras" not in cx:
        mapa = defaultdict(set)
        for a, b in cx["aristas"]:
            mapa[a].add(b)
        cx["sucesoras"] = mapa
    return cx["sucesoras"]


def cierre_sucesoras(ids, cx):
    """(ids liberados con sus sucesoras transitivas, ids agregados por propagación).

    Se liberan también las sucesoras para que ninguna tarea congelada quede antes que una predecesora
    reubicada: así las precedencias heredadas nunca se invierten.
    """
    mapa = sucesoras(cx)
    total, pila = set(ids), list(ids)
    while pila:
        for s in mapa.get(pila.pop(), ()):
            if s not in total:
                total.add(s)
                pila.append(s)
    return total, total - set(ids)


def tabla_tareas(df, cx):
    """Una fila por Parent_Task_ID: inicio, fin, grúa final, bay, movimientos y fragmentos."""
    inicio = pd.to_datetime(df["Inicio"])
    fin = pd.to_datetime(df["Fin"])
    filas = []
    for pid, idx in df.groupby("Parent_Task_ID", sort=True).groups.items():
        t = cx["tareas_por_id"][pid]
        ini_p, fin_p = inicio.loc[idx], fin.loc[idx]
        ultima = df.loc[fin_p.idxmax(), "Grúa"]
        filas.append({
            "id": pid, "ini": ini_p.min(), "fin": fin_p.max(), "grua": ultima,
            "gruas": set(df.loc[idx, "Grúa"]), "bay": t["bay"], "movs": int(t["movs"]),
            "ponton": t["tipo"] == "Pontón", "fragmentos": len(idx),
        })
    return filas


def _elegibles(tabla):
    return [f for f in tabla if not f["ponton"]]


def _grua_critica(df):
    fines = pd.to_datetime(df["Fin"]).groupby(df["Grúa"]).max()
    return max(fines.index, key=lambda g: (fines[g], g))


def limites_vecindad(n_tareas, diversificacion=False):
    """(mínimo, máximo) de tareas originales (Parent_Task_ID) a liberar según el tamaño de la instancia."""
    if n_tareas <= 0:
        return 0, 0
    if n_tareas <= 10:
        minimo, maximo = 2, (5 if diversificacion else 4)
    elif n_tareas <= 30:
        minimo, maximo = 3, (10 if diversificacion else 8)
    elif n_tareas <= 100:
        minimo, maximo = math.ceil(0.05 * n_tareas), math.ceil((0.30 if diversificacion else 0.20) * n_tareas)
    else:
        minimo, maximo = math.ceil(0.05 * n_tareas), math.ceil((0.25 if diversificacion else 0.15) * n_tareas)
    return min(minimo, n_tareas), min(max(minimo, maximo), n_tareas)


def tamano_vecindad(n_tareas, porcentaje, diversificacion=False):
    """Cantidad de tareas originales a liberar: el porcentaje pedido acotado por los límites de la instancia."""
    minimo, maximo = limites_vecindad(n_tareas, diversificacion)
    if maximo == 0:
        return 0
    return int(min(max(minimo, round(porcentaje * n_tareas)), maximo))


def contar_originales(df, cx):
    """Tareas originales (Parent_Task_ID, sin pontones) de una solución; los fragmentos no cuentan."""
    pontones = {t["id"] for t in cx["tareas"] if t["tipo"] == "Pontón"}
    return int(len(set(df["Parent_Task_ID"]) - pontones))


def _completar(ids, tabla, n, cercano_a=None):
    """Quita repetidos y pontones, recorta a n y, si faltan, agrega las tareas más cercanas en el tiempo."""
    elegibles = {f["id"]: f for f in _elegibles(tabla)}
    salida = []
    for i in ids:
        if i in elegibles and i not in salida:
            salida.append(i)
    salida = salida[:n]
    if len(salida) < min(n, len(elegibles)):
        ancla = elegibles[salida[0]]["ini"] if salida else (cercano_a or min(f["ini"] for f in elegibles.values()))
        resto = sorted(
            (f for f in elegibles.values() if f["id"] not in salida),
            key=lambda f: (abs((f["ini"] - ancla).total_seconds()), f["id"]),
        )
        salida += [f["id"] for f in resto[: min(n, len(elegibles)) - len(salida)]]
    return salida


# --- operadores de destrucción --------------------------------------------------------------
# Todos reciben (df, cx, rnd, n, estado), no mutan nada y devuelven una lista de Parent_Task_ID
# (sin pontones, al menos una tarea si existe alguna no pontón).

def critical_crane_removal(df, cx, rnd, n, estado):
    """Tramo final de la grúa que determina el makespan (no se presume que deba conservarlo)."""
    tabla = tabla_tareas(df, cx)
    critica = _grua_critica(df)
    propias = sorted(
        (f for f in _elegibles(tabla) if critica in f["gruas"]), key=lambda f: (f["fin"], f["id"]), reverse=True
    )
    ids = [f["id"] for f in propias]
    if len(ids) > n and rnd.random() < 0.5:
        ids = rnd.sample(ids[: n + 2], n)
    return _completar(ids, tabla, n, propias[0]["ini"] if propias else None)


def temporal_window_removal(df, cx, rnd, n, estado):
    """Tareas más cercanas a una ventana continua, con preferencia por las cercanas al fin."""
    tabla = _elegibles(tabla_tareas(df, cx))
    if not tabla:
        return []
    inicio = min(f["ini"] for f in tabla)
    fin = max(f["fin"] for f in tabla)
    ancla = fin - (fin - inicio) * (0.5 * rnd.random() ** 2)
    medio = lambda f: f["ini"] + (f["fin"] - f["ini"]) / 2
    ordenadas = sorted(tabla, key=lambda f: (abs((medio(f) - ancla).total_seconds()), f["id"]))
    return [f["id"] for f in ordenadas[:n]]


def spatial_block_removal(df, cx, rnd, n, estado):
    """Tareas dentro de un rango continuo de bays (diferencia 4 = contiguo, 8 = dos posiciones)."""
    tabla = _elegibles(tabla_tareas(df, cx))
    con_bay = [f for f in tabla if f["bay"] is not None]
    if not con_bay:
        return _completar([], tabla_tareas(df, cx), n)
    bays = sorted({f["bay"] for f in con_bay})
    posicion = rnd.randrange(len(bays))
    elegidos, derecha, izquierda = [], posicion, posicion - 1
    rango = []
    while len(elegidos) < n and (derecha < len(bays) or izquierda >= 0):
        for lado in (0, 1):
            if lado == 0 and derecha < len(bays):
                bay, derecha = bays[derecha], derecha + 1
            elif lado == 1 and izquierda >= 0:
                bay, izquierda = bays[izquierda], izquierda - 1
            else:
                continue
            rango.append(bay)
            elegidos = [f for f in con_bay if f["bay"] in rango]
            if len(elegidos) >= n:
                break
    centro = bays[posicion]
    elegidos.sort(key=lambda f: (abs(f["bay"] - centro), f["ini"], f["id"]))
    return _completar([f["id"] for f in elegidos[:n]], tabla_tareas(df, cx), n)


def workload_imbalance_removal(df, cx, rnd, n, estado):
    """Trabajo de la grúa que termina más tarde; el reparador decide si otra grúa lo toma."""
    tabla = tabla_tareas(df, cx)
    critica = _grua_critica(df)
    propias = [f for f in _elegibles(tabla) if critica in f["gruas"]]
    propias.sort(key=lambda f: (-f["movs"] * rnd.uniform(0.7, 1.3), f["id"]))
    return _completar([f["id"] for f in propias], tabla, n)


def idle_gap_removal(df, cx, rnd, n, estado):
    """Tareas antes y después de un hueco ocioso, más tareas de otras grúas que puedan bloquearlo."""
    tabla = tabla_tareas(df, cx)
    por_id = {f["id"]: f for f in tabla}
    umbral = pd.Timedelta(minutes=max(cx["holgura"], 15))
    ini, fin = pd.to_datetime(df["Inicio"]), pd.to_datetime(df["Fin"])
    huecos = []
    for grua, idx in df.groupby("Grúa").groups.items():
        orden = sorted(idx, key=lambda i: (ini.loc[i], i))
        for a, b in zip(orden, orden[1:]):
            if ini.loc[b] - fin.loc[a] >= umbral:
                huecos.append((ini.loc[b] - fin.loc[a], grua, a, b))
    if not huecos:
        return related_task_removal(df, cx, rnd, n, estado)
    huecos.sort(key=lambda h: (h[0], h[1], h[2]), reverse=True)
    _, grua, a, b = huecos[rnd.randrange(min(3, len(huecos)))]
    antes, despues = df.loc[a, "Parent_Task_ID"], df.loc[b, "Parent_Task_ID"]
    desde, hasta = fin.loc[a], ini.loc[b]
    ids = [despues, antes]
    bay_siguiente = por_id[despues]["bay"]
    for f in sorted(_elegibles(tabla), key=lambda f: (f["ini"], f["id"])):
        if grua in f["gruas"] or f["id"] in ids:
            continue
        cerca = bay_siguiente is None or f["bay"] is None or abs(f["bay"] - bay_siguiente) < ifc.SEPARACION_MIN
        if f["ini"] < hasta + pd.Timedelta(minutes=cx["holgura"]) and f["fin"] > desde and cerca:
            ids.append(f["id"])
    return _completar(ids, tabla, n)


def related_task_removal(df, cx, rnd, n, estado, base=None):
    """Una tarea base y las más relacionadas: mismo padre, cadena, precedencia, tiempo, espacio y grúa."""
    tabla = tabla_tareas(df, cx)
    elegibles = _elegibles(tabla)
    if not elegibles:
        return []
    por_id = {f["id"]: f for f in tabla}
    base = por_id.get(base) if base in por_id and not por_id[base]["ponton"] else rnd.choice(elegibles)
    duracion = max((max(f["fin"] for f in tabla) - min(f["ini"] for f in tabla)).total_seconds(), 1.0)
    bays = [f["bay"] for f in tabla if f["bay"] is not None]
    ancho = max((max(bays) - min(bays)) if bays else 1, 1)
    vecinos = {a for a, b in cx["aristas"] if b == base["id"]} | sucesoras(cx).get(base["id"], set())

    def distancia(f):
        puntos = abs((f["ini"] - base["ini"]).total_seconds()) / duracion
        if f["bay"] is not None and base["bay"] is not None:
            puntos += abs(f["bay"] - base["bay"]) / ancho
            puntos -= 0.5 if f["bay"] == base["bay"] else 0.0
        puntos -= 1.0 if f["id"] in vecinos else 0.0
        puntos += 0.0 if f["gruas"] & base["gruas"] else 0.2
        return (puntos, f["id"])

    otras = sorted((f for f in elegibles if f["id"] != base["id"]), key=distancia)
    return [base["id"]] + [f["id"] for f in otras[: max(n - 1, 0)]]


def conflict_history_removal(df, cx, rnd, n, estado):
    """Tareas ligadas a conflictos previos o a candidatas infactibles; sin historial, tareas relacionadas."""
    conocidos = {f["id"] for f in _elegibles(tabla_tareas(df, cx))}
    historial = {i: c for i, c in (estado.conflictos.items() if estado else ()) if i in conocidos and c > 0}
    if not historial:
        return related_task_removal(df, cx, rnd, n, estado)
    ids = sorted(historial)
    base = rnd.choices(ids, weights=[historial[i] for i in ids], k=1)[0]
    return related_task_removal(df, cx, rnd, n, estado, base=base)


def random_block_removal(df, cx, rnd, n, estado):
    """Bloque contiguo (por inicio) reproducible: diversificación."""
    tabla = sorted(_elegibles(tabla_tareas(df, cx)), key=lambda f: (f["ini"], f["id"]))
    if not tabla:
        return []
    n = min(n, len(tabla))
    inicio = rnd.randrange(len(tabla) - n + 1)
    return [f["id"] for f in tabla[inicio: inicio + n]]


DESTRUCTORES = {
    "critical_crane_removal": critical_crane_removal,
    "temporal_window_removal": temporal_window_removal,
    "spatial_block_removal": spatial_block_removal,
    "workload_imbalance_removal": workload_imbalance_removal,
    "idle_gap_removal": idle_gap_removal,
    "related_task_removal": related_task_removal,
    "conflict_history_removal": conflict_history_removal,
    "random_block_removal": random_block_removal,
}

# --- reparadores ----------------------------------------------------------------------------
# Reciben (cx, df, liberados, originales, rnd, estado) y devuelven (df candidata, info). Reconstruyen
# siempre con `ifc._despachar`, es decir con `calcular_intervalo_tarea` y el GMPH de la grúa receptora;
# lo no liberado queda congelado (misma grúa, horarios, identidad y movimientos).

def _despachar_vecindad(cx, df, liberados, criterio, prioridades=None, forzadas=None, excluidas=None,
                        sin_dividir=()):
    piezas = ifc._piezas_desde_df(df, cx)
    congeladas = [p for p in piezas if p["tarea"]["id"] not in liberados]
    nuevas = ifc._despachar(
        cx["tareas"], cx["config"], cx["intervalo_fn"], cx["holgura"], criterio, cx["rango"],
        piezas_previas=congeladas, excluidas=excluidas, forzadas=forzadas, limite=cx["limite"],
        aristas=cx["aristas"], permitir_reasignacion=cx["reasignar"], permitir_division=cx["dividir"],
        huecos=True, prioridades=prioridades, sin_dividir=sin_dividir,
    )
    return ifc._armar_cronograma(nuevas)


def greedy_repair(cx, df, liberados, originales, rnd, estado):
    """Reinserta eligiendo en cada paso la alternativa con menor término proyectado."""
    return _despachar_vecindad(cx, df, liberados, "fin"), {}


def regret2_repair(cx, df, liberados, originales, rnd, estado):
    """Reinserta primero la tarea con mayor regret = costo de la 2.ª mejor menos el de la mejor alternativa."""
    return _despachar_vecindad(cx, df, liberados, "regret2"), {}


def critical_first_repair(cx, df, liberados, originales, rnd, estado):
    """Prioriza tramo final, grúa crítica, más movimientos y tareas que bloquean a otras liberadas."""
    tabla = {f["id"]: f for f in tabla_tareas(df, cx)}
    critica = _grua_critica(df)
    fin_total = max(f["fin"] for f in tabla.values())
    inicio_total = min(f["ini"] for f in tabla.values())
    corte = inicio_total + (fin_total - inicio_total) * 0.7
    prioridades = {}
    for i in liberados:
        f = tabla[i]
        bloqueos = sum(
            1 for j in liberados
            if j != i and f["bay"] is not None and tabla[j]["bay"] is not None
            and abs(f["bay"] - tabla[j]["bay"]) < ifc.SEPARACION_MIN
        )
        prioridades[i] = (
            -int(f["fin"] >= corte), -int(critica in f["gruas"]), -f["movs"], -bloqueos, f["id"],
        )
    return _despachar_vecindad(cx, df, liberados, "prioridad", prioridades=prioridades), {}


MAX_ORIGINALES_CP = 12
MAX_LIBERADAS_CP = 18
MAX_GRUAS_CP = 3


def cp_local_elegible(cx, originales, liberados):
    return MINIMO_TAREAS <= len(originales) <= MAX_ORIGINALES_CP and len(liberados) <= MAX_LIBERADAS_CP


def _minutos(ts, t0, techo=False):
    segundos = (pd.Timestamp(ts) - t0).total_seconds()
    return int(math.ceil(segundos / 60)) if techo else int(math.floor(segundos / 60))


def _plan_cp_local(cx, df, liberados, semilla_aleatoria, tiempo):
    """Resuelve con CP-SAT la grúa y el orden de una vecindad pequeña; lo demás queda fijo.

    El modelo usa duraciones aproximadas por grúa: solo sirve de guía (grúa y orden). Los tiempos
    reales los reconstruye `_despachar`. Retorna (forzadas, prioridades, estado) o lanza
    FalloReparacion con el estado del solver (INFEASIBLE, UNKNOWN, ...).
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:
        raise FalloReparacion("ortools no está instalado", "NO_DISPONIBLE") from e

    tabla = {f["id"]: f for f in tabla_tareas(df, cx)}
    ids = sorted(liberados)
    tareas = [cx["tareas_por_id"][i] for i in ids]
    rango, config = cx["rango"], cx["config"]

    if cx["reasignar"]:
        conteo = Counter(tabla[i]["grua"] for i in ids if not tabla[i]["ponton"])
        candidatas = [g for g, _ in conteo.most_common(MAX_GRUAS_CP)]
        fines = pd.to_datetime(df["Fin"]).groupby(df["Grúa"]).max()
        for g in sorted(cx["gruas"], key=lambda g: (fines.get(g, cx["t_ref"]), g)):
            if len(candidatas) >= MAX_GRUAS_CP:
                break
            if g not in candidatas:
                candidatas.append(g)
    else:
        candidatas = sorted({t["grua"] for t in tareas})
    for t in tareas:
        if t["fijo_grua"] and t["fijo_grua"] not in candidatas:
            candidatas.append(t["fijo_grua"])
    candidatas = [g for g in candidatas if g in rango]

    t0 = min(cx["inicios"].values())
    desfase = {g: max(0, _minutos(config[g]["inicio"], t0)) for g in candidatas}
    sueltas = [t for t in tareas if t["dur_fija"] is None]
    referencia = max(50, sum(t["movs"] for t in sueltas) // max(len(candidatas), 1))
    por_mov = {}
    for g in candidatas:
        inicio_g = pd.Timestamp(config[g]["inicio"])
        try:
            _, fin, _ = cx["intervalo_fn"](config[g], inicio_g.to_pydatetime(), referencia, None)
            por_mov[g] = max((pd.Timestamp(fin) - inicio_g).total_seconds() / 60 / referencia, 1e-3)
        except ValueError:
            por_mov[g] = 60 / max(float(config[g]["gmph"]), 1e-9)

    piezas = ifc._piezas_desde_df(df, cx)
    congeladas = [p for p in piezas if p["tarea"]["id"] not in liberados]
    fin_congelada = defaultdict(int)
    fijas = []
    for p in congeladas:
        ini, fin = _minutos(p["ini"], t0), _minutos(p["fin"], t0, techo=True)
        fin_congelada[p["tarea"]["id"]] = max(fin_congelada[p["tarea"]["id"]], fin)
        fijas.append({"grua": p["grua"], "bay": p["tarea"]["bay"], "ini": ini, "fin": max(fin, ini + 1)})

    def duracion(t, g):
        if t["dur_fija"] is not None:
            return max(1, int(math.ceil(t["dur_fija"])))
        return max(1, int(math.ceil(t["movs"] * por_mov[g])))

    horizonte = max([f["fin"] for f in fijas] + [0]) + max(desfase.values(), default=0) + 1
    horizonte += sum(max(duracion(t, g) for g in candidatas) + cx["holgura"] for t in tareas)

    m = cp_model.CpModel()
    S, E, X, R = {}, {}, {}, {}
    intervalos = {g: [] for g in candidatas}
    for t in tareas:
        i = t["id"]
        permitidas = [t["fijo_grua"]] if t["fijo_grua"] in candidatas else candidatas
        if not cx["reasignar"] and not t["fijo_grua"]:
            permitidas = [t["grua"]] if t["grua"] in candidatas else candidatas
        S[i] = m.NewIntVar(0, horizonte, f"s{i}")
        E[i] = m.NewIntVar(0, horizonte, f"e{i}")
        X[i] = {g: m.NewBoolVar(f"x{i}_{g}") for g in permitidas}
        m.AddExactlyOne(X[i].values())
        m.Add(E[i] == S[i] + sum(duracion(t, g) * X[i][g] for g in X[i]))
        for g, lit in X[i].items():
            m.Add(S[i] >= desfase[g]).OnlyEnforceIf(lit)
            intervalos[g].append(m.NewOptionalIntervalVar(S[i], duracion(t, g), S[i] + duracion(t, g), lit, f"i{i}_{g}"))
        R[i] = m.NewIntVar(0, max(rango.values()), f"r{i}")
        m.Add(R[i] == sum(rango[g] * X[i][g] for g in X[i]))
    for k, f in enumerate(fijas):
        if f["grua"] in intervalos:
            intervalos[f["grua"]].append(m.NewFixedSizeIntervalVar(f["ini"], f["fin"] - f["ini"], f"f{k}"))
    for g in candidatas:
        m.AddNoOverlap(intervalos[g])

    ids_set = set(ids)
    internas = [(a, b) for a, b in cx["aristas"] if a in ids_set and b in ids_set]
    for a, b in internas:
        m.Add(E[a] <= S[b])
    for a, b in cx["aristas"]:
        if b in ids_set and a not in ids_set and a in fin_congelada:
            m.Add(S[b] >= fin_congelada[a])
    suc = defaultdict(set)
    for a, b in internas:
        suc[a].add(b)
    precede = set()
    for origen in ids:
        visto, pila = set(), list(suc[origen])
        while pila:
            d = pila.pop()
            if d not in visto:
                visto.add(d)
                pila.extend(suc[d])
        precede.update((origen, d) for d in visto)

    holgura = cx["holgura"]
    con_bay = [t for t in tareas if t["bay"] is not None]
    for pos, ti in enumerate(con_bay):
        i = ti["id"]
        for tj in con_bay[pos + 1:]:
            j = tj["id"]
            distancia = abs(ti["bay"] - tj["bay"])
            ordenadas = (i, j) in precede or (j, i) in precede
            if distancia < ifc.SEPARACION_MIN:
                antes = m.NewBoolVar(f"a{i}_{j}")
                if not ordenadas:
                    m.Add(E[i] <= S[j]).OnlyEnforceIf(antes)
                    m.Add(E[j] <= S[i]).OnlyEnforceIf(antes.Not())
                if distancia <= ifc.DISTANCIA_MAX_BAYS and holgura > 0:
                    distintas = m.NewBoolVar(f"d{i}_{j}")
                    m.Add(R[i] != R[j]).OnlyEnforceIf(distintas)
                    m.Add(R[i] == R[j]).OnlyEnforceIf(distintas.Not())
                    m.Add(E[i] + holgura <= S[j]).OnlyEnforceIf([antes, distintas])
                    m.Add(E[j] + holgura <= S[i]).OnlyEnforceIf([antes.Not(), distintas])
            elif not ordenadas:
                popa, proa = (i, j) if ti["bay"] > tj["bay"] else (j, i)
                correcto = m.NewBoolVar(f"o{i}_{j}")
                m.Add(R[popa] > R[proa]).OnlyEnforceIf(correcto)
                m.Add(R[popa] <= R[proa]).OnlyEnforceIf(correcto.Not())
                antes = m.NewBoolVar(f"b{i}_{j}")
                m.Add(E[i] <= S[j]).OnlyEnforceIf([antes, correcto.Not()])
                m.Add(E[j] <= S[i]).OnlyEnforceIf([antes.Not(), correcto.Not()])

    for k, f in enumerate(fijas):
        if f["bay"] is None or f["grua"] not in rango:
            continue
        rf = rango[f["grua"]]
        for t in con_bay:
            i = t["id"]
            distancia = abs(t["bay"] - f["bay"])
            antes = m.NewBoolVar(f"fa{i}_{k}")
            if distancia < ifc.SEPARACION_MIN:
                m.Add(E[i] <= f["ini"]).OnlyEnforceIf(antes)
                m.Add(S[i] >= f["fin"]).OnlyEnforceIf(antes.Not())
                if distancia <= ifc.DISTANCIA_MAX_BAYS and holgura > 0:
                    distinta = m.NewBoolVar(f"fd{i}_{k}")
                    m.Add(R[i] != rf).OnlyEnforceIf(distinta)
                    m.Add(R[i] == rf).OnlyEnforceIf(distinta.Not())
                    m.Add(E[i] + holgura <= f["ini"]).OnlyEnforceIf([antes, distinta])
                    m.Add(S[i] >= f["fin"] + holgura).OnlyEnforceIf([antes.Not(), distinta])
            else:
                correcto = m.NewBoolVar(f"fo{i}_{k}")
                if t["bay"] > f["bay"]:
                    m.Add(R[i] > rf).OnlyEnforceIf(correcto)
                    m.Add(R[i] <= rf).OnlyEnforceIf(correcto.Not())
                else:
                    m.Add(R[i] < rf).OnlyEnforceIf(correcto)
                    m.Add(R[i] >= rf).OnlyEnforceIf(correcto.Not())
                m.Add(E[i] <= f["ini"]).OnlyEnforceIf([antes, correcto.Not()])
                m.Add(S[i] >= f["fin"]).OnlyEnforceIf([antes.Not(), correcto.Not()])

    makespan = m.NewIntVar(0, horizonte, "makespan")
    m.Add(makespan >= max([f["fin"] for f in fijas] + [0]))
    termino = {g: m.NewIntVar(0, horizonte, f"t{g}") for g in candidatas}
    for i in ids:
        m.Add(makespan >= E[i])
        for g, lit in X[i].items():
            m.Add(termino[g] >= E[i]).OnlyEnforceIf(lit)
    m.Minimize(100 * makespan + sum(termino.values()))

    filas = df.assign(_i=pd.to_datetime(df["Inicio"]))
    for i in ids:
        propias = filas[filas["Parent_Task_ID"] == i].sort_values("_i")
        if propias.empty:
            continue
        grua_pista = propias.iloc[0]["Grúa"]
        if grua_pista in X[i]:
            m.AddHint(S[i], min(max(0, _minutos(propias.iloc[0]["_i"], t0)), horizonte))
            for g, lit in X[i].items():
                m.AddHint(lit, 1 if g == grua_pista else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(float(tiempo), 0.1)
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = int(semilla_aleatoria)
    resultado = solver.Solve(m)
    nombre = solver.StatusName(resultado)
    if resultado not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise FalloReparacion(f"CP-SAT local: {nombre}", nombre)
    forzadas = {i: next(g for g, lit in X[i].items() if solver.Value(lit)) for i in ids}
    prioridades = {i: (solver.Value(S[i]),) for i in ids}
    makespan_ms = solver.Value(makespan) + (t0 - cx["t_ref"]).total_seconds() / 60
    return forzadas, prioridades, nombre, float(makespan_ms)


def local_cp_sat_repair(cx, df, liberados, originales, rnd, estado):
    """CP-SAT en una vecindad pequeña: decide grúa y orden; los tiempos reales se reconstruyen aparte."""
    if not cp_local_elegible(cx, originales, liberados):
        raise FalloReparacion("vecindad fuera del rango del CP-SAT local", "NO_ELEGIBLE")
    restante = cx["limite"] - time.monotonic()
    if restante < 0.7:
        raise FalloReparacion("queda muy poco tiempo para CP-SAT local", "SIN_TIEMPO")
    tiempo = min(1.5, max(0.5, 0.15 * restante), restante - 0.2)
    forzadas, prioridades, nombre, ms_aprox = _plan_cp_local(
        cx, df, liberados, rnd.randrange(1_000_000), tiempo
    )
    candidata = _despachar_vecindad(cx, df, liberados, "prioridad", prioridades=prioridades, forzadas=forzadas)
    return candidata, {"estado_cpsat": nombre, "ms_cp_aprox": ms_aprox}


REPARADORES = {
    "greedy_repair": greedy_repair,
    "regret2_repair": regret2_repair,
    "critical_first_repair": critical_first_repair,
    "local_cp_sat_repair": local_cp_sat_repair,
}

# --- movimientos locales (intensificación) --------------------------------------------------
# Devuelven None si no aplican, o un plan {originales, criterio, forzadas, excluidas, prioridades,
# sin_dividir}. Se reconstruyen y validan igual que cualquier candidata.

def _por_inicio(tabla):
    return sorted(_elegibles(tabla), key=lambda f: (f["ini"], f["id"]))


def relocate_task(df, cx, rnd, estado):
    """Mueve una tarea completa (del tramo final de la grúa crítica) a otra grúa."""
    if not cx["reasignar"] or len(cx["gruas"]) < 2:
        return None
    tabla = tabla_tareas(df, cx)
    critica = _grua_critica(df)
    propias = sorted((f for f in _elegibles(tabla) if f["grua"] == critica), key=lambda f: f["fin"], reverse=True)
    if not propias:
        return None
    f = rnd.choice(propias[: max(2, min(4, len(propias)))])
    destino = rnd.choice([g for g in cx["gruas"] if g not in f["gruas"]] or cx["gruas"])
    return {"originales": [f["id"]], "criterio": "fin_sin_preferencia", "forzadas": {f["id"]: destino}}


def swap_tasks(df, cx, rnd, estado):
    """Intercambia una tarea de la grúa crítica con una de otra grúa."""
    if not cx["reasignar"]:
        return None
    tabla = tabla_tareas(df, cx)
    critica = _grua_critica(df)
    a_lista = [f for f in _elegibles(tabla) if f["gruas"] == {critica}]
    b_lista = [f for f in _elegibles(tabla) if len(f["gruas"]) == 1 and critica not in f["gruas"]]
    if not a_lista or not b_lista:
        return None
    a = rnd.choice(sorted(a_lista, key=lambda f: f["fin"], reverse=True)[:4])
    b = rnd.choice(b_lista)
    return {
        "originales": [a["id"], b["id"]], "criterio": "fin_sin_preferencia",
        "forzadas": {a["id"]: b["grua"], b["id"]: a["grua"]},
    }


def relocate_block(df, cx, rnd, estado):
    """Mueve un bloque pequeño de tareas consecutivas de una grúa a otra."""
    if not cx["reasignar"] or len(cx["gruas"]) < 2:
        return None
    tabla = tabla_tareas(df, cx)
    critica = _grua_critica(df)
    propias = [f for f in _por_inicio(tabla) if f["gruas"] == {critica}]
    if len(propias) < 2:
        return None
    k = min(len(propias), rnd.randint(2, 4))
    inicio = len(propias) - k - rnd.randint(0, max(0, len(propias) - k))
    bloque = propias[max(0, inicio): max(0, inicio) + k]
    destino = rnd.choice([g for g in cx["gruas"] if g != critica])
    return {
        "originales": [f["id"] for f in bloque], "criterio": "fin_sin_preferencia",
        "forzadas": {f["id"]: destino for f in bloque},
    }


def change_first_task(df, cx, rnd, estado):
    """Sustituye la primera tarea de una grúa. El bay inicial cambia solo como consecuencia."""
    tabla = tabla_tareas(df, cx)
    gruas = sorted({g for f in tabla for g in f["gruas"]})
    grua = rnd.choice(gruas)
    ordenadas = _por_inicio(tabla)
    propias = [f for f in ordenadas if f["gruas"] == {grua}]
    if not propias:
        return None
    primera = propias[0]
    candidatas = [f for f in ordenadas if f["id"] != primera["id"] and (cx["reasignar"] or f["gruas"] == {grua})]
    if not candidatas:
        return None
    elegida = rnd.choice(candidatas)
    return {
        "originales": [primera["id"], elegida["id"]], "criterio": "prioridad",
        "forzadas": {elegida["id"]: grua}, "prioridades": {elegida["id"]: (-1,), primera["id"]: (1,)},
    }


def merge_or_redistribute_fragments(df, cx, rnd, estado):
    """Une los fragmentos de una tarea o los redistribuye entre grúas (solo con división permitida).

    No se fijan fracciones a mano: unir se logra reconstruyendo la tarea sin dividirla; redistribuir,
    liberándola con otra grúa preferida para que `_evaluar` la parta donde cabe. En ambos casos los
    movimientos se reconstruyen con las reglas reales y se validan, así que la suma se conserva.
    """
    if not cx["dividir"]:
        return None
    tabla = tabla_tareas(df, cx)
    divididas = [f for f in _elegibles(tabla) if f["fragmentos"] > 1]
    if divididas:
        f = rnd.choice(divididas)
        if rnd.random() < 0.5:
            return {"originales": [f["id"]], "criterio": "fin", "sin_dividir": {f["id"]}}
        return {
            "originales": [f["id"]], "criterio": "fin_sin_preferencia",
            "excluidas": {f["id"]: set(f["gruas"])},
        }
    grandes = [f for f in _elegibles(tabla) if f["movs"] >= 2 * ifc.MIN_PARTE]
    if not grandes:
        return None
    f = max(grandes, key=lambda f: (f["fin"], f["id"]))
    return {"originales": [f["id"]], "criterio": "fin_sin_preferencia"}


MOVIMIENTOS_LOCALES = {
    "relocate_task": relocate_task,
    "swap_tasks": swap_tasks,
    "relocate_block": relocate_block,
    "change_first_task": change_first_task,
    "merge_or_redistribute_fragments": merge_or_redistribute_fragments,
}


def aplicar_movimiento(cx, df, plan):
    liberados, propagadas = cierre_sucesoras(plan["originales"], cx)
    candidata = _despachar_vecindad(
        cx, df, liberados, plan["criterio"], prioridades=plan.get("prioridades"),
        forzadas=plan.get("forzadas"), excluidas=plan.get("excluidas"),
        sin_dividir=plan.get("sin_dividir", ()),
    )
    return candidata, liberados, propagadas

# --- estado y bucle ALNS ----------------------------------------------------------------------

UMBRAL_DIVERSIFICACION = 20
LIMITE_REINICIO = 60
SEGMENTO_PESOS = 10
FACTOR_PESO = 0.20
ENFRIAMIENTO = 0.98
TEMPERATURA_MINIMA = 0.01
PROBABILIDAD_MOVIMIENTO_LOCAL = 0.12
MAX_HISTORIAL = 3000
MAX_VISITADAS = 400
UMBRAL_CP_ESTANCAMIENTO = 10
SEPARACION_CP = 6
PESO_INICIAL_CP = 0.3
INTENSIFICACION = [
    ("critical_crane_removal", "regret2_repair"),
    ("critical_crane_removal", "relocate_task"),
    ("critical_crane_removal", "relocate_block"),
    ("critical_crane_removal", "local_cp_sat_repair"),
]


def elegir_ponderado(pesos, rnd):
    """Elección proporcional a los pesos; pesos no finitos o no positivos se sustituyen por el mínimo."""
    nombres = sorted(pesos)
    if not nombres:
        raise ValueError("No hay operadores entre los que elegir.")
    valores = []
    for nombre in nombres:
        w = pesos[nombre]
        valores.append(float(w) if isinstance(w, (int, float)) and math.isfinite(w) and w > 0 else PESO_MINIMO)
    total = sum(valores)
    if not (math.isfinite(total) and total > 0):
        return nombres[int(rnd.random() * len(nombres)) % len(nombres)]
    umbral, acumulado = rnd.random() * total, 0.0
    for nombre, valor in zip(nombres, valores):
        acumulado += valor
        if umbral < acumulado:
            return nombre
    return nombres[-1]


def actualizar_pesos(pesos, puntuaciones):
    """peso = (1 - 0.20) · peso anterior + 0.20 · puntuación media; el peso nunca baja del mínimo."""
    for nombre, lista in puntuaciones.items():
        if nombre in pesos and lista:
            nuevo = (1 - FACTOR_PESO) * pesos[nombre] + FACTOR_PESO * (sum(lista) / len(lista))
            pesos[nombre] = max(PESO_MINIMO, nuevo) if math.isfinite(nuevo) else PESO_MINIMO


def firma_solucion(df):
    return tuple(sorted(zip(
        df["Parent_Task_ID"], df["Grúa"], pd.to_datetime(df["Inicio"]), pd.to_datetime(df["Fin"]), df["Movimientos"]
    )))


def firma_logica(df):
    """Firma del cronograma independiente del índice: identidad, grúa, orden en la grúa, movimientos y horas."""
    columnas = {
        c: (df[c].astype(str) if c in df.columns else pd.Series([""] * len(df), index=df.index))
        for c in ("Task_ID", "Parent_Task_ID", "Fragment_ID")
    }
    d = pd.DataFrame({
        **columnas, "g": df["Grúa"].astype(str), "mov": df["Movimientos"].astype(int),
        "ini": pd.to_datetime(df["Inicio"]), "fin": pd.to_datetime(df["Fin"]),
    }).sort_values(["g", "ini", "fin", "Parent_Task_ID", "Fragment_ID"], kind="mergesort")
    d["orden"] = d.groupby("g").cumcount()
    return tuple(sorted(
        (r.Task_ID, r.Parent_Task_ID, r.Fragment_ID, r.g, r.orden, r.mov, r.ini, r.fin) for r in d.itertuples()
    ))


def _asignacion(df):
    return frozenset(
        (p, g, int(m)) for (p, g), m in df.groupby(["Parent_Task_ID", "Grúa"])["Movimientos"].sum().items()
    )


def _orden_por_grua(df):
    d = df.assign(_i=pd.to_datetime(df["Inicio"])).sort_values(["_i", "Parent_Task_ID"], kind="mergesort")
    return {g: tuple(x["Parent_Task_ID"]) for g, x in d.groupby("Grúa")}


def _copia_profunda(ev):
    copia = dict(ev)
    copia["df"] = ev["df"].copy(deep=True)
    return copia


def _estadistica_vacia():
    return {
        "usos": 0, "reconstrucciones": 0, "factibles": 0, "aceptadas": 0, "mejoras_actual": 0,
        "nuevas_mejores": 0, "reduccion_makespan": 0.0, "tiempo_s": 0.0, "fallos": 0,
    }


class EstadoALNS:
    """Estado completo del ALNS; toda candidata se construye sobre copias independientes."""

    def __init__(self, cx, inc, rnd, porcentaje=0.10, usar_sa=True, usar_cp=True, max_iteraciones=None):
        self.cx, self.inc, self.rnd = cx, inc, rnd
        self.solucion_semilla = cx["semilla"]
        self.primera_solucion_factible = inc.mejor
        self.solucion_actual = inc.mejor
        self.usar_sa, self.max_iteraciones = usar_sa, max_iteraciones
        self.porcentaje_base = self.porcentaje_actual = float(porcentaje)
        self.pesos_destruccion = {n: 1.0 for n in DESTRUCTORES}
        self.pesos_reparacion = {
            n: (PESO_INICIAL_CP if n == "local_cp_sat_repair" else 1.0)
            for n in REPARADORES if usar_cp or n != "local_cp_sat_repair"
        }
        self.estadisticas = {
            "destruccion": {n: _estadistica_vacia() for n in DESTRUCTORES},
            "reparacion": {n: _estadistica_vacia() for n in self.pesos_reparacion},
            "movimientos": {n: _estadistica_vacia() for n in MOVIMIENTOS_LOCALES},
        }
        self.temperatura = max(1.0, 0.02 * inc.mejor["ms"]) if inc.mejor is not None else 1.0
        self.sin_mejora = 0
        self.reinicios = 0
        self.intensificar = 0
        self.foco = None
        self.conflictos = Counter()
        self.puntos = {"destruccion": defaultdict(list), "reparacion": defaultdict(list)}
        self.iteraciones = self.candidatas = self.candidatas_factibles = 0
        self.candidatas_aceptadas = self.mejoras = 0
        self.estados_cpsat = Counter()
        self.visitadas = OrderedDict()
        if inc.mejor is not None:
            self.visitadas[firma_logica(inc.mejor["df"])] = inc.mejor
        self.cola_intensificacion = []
        self.ultimo_cp = -999
        self.respaldo_mejor = _copia_profunda(inc.mejor) if inc.mejor is not None else None
        self.ms_inicial = inc.mejor["ms"] if inc.mejor is not None else None
        self.mejor_ms_propio = None
        self.diversidad = {
            "candidatas_evaluadas": 0, "candidatas_repetidas": 0, "candidatas_identicas": 0, "candidatas_sin_cambio": 0,
            "candidatas_con_reasignacion": 0, "candidatas_con_cambio_orden": 0,
            "candidatas_con_cambio_fragmentacion": 0, "candidatas_con_makespan_distinto": 0,
            "candidatas_mejoran_makespan": 0, "candidatas_peores_aceptadas": 0,
            "_suma_delta": 0.0, "_n_delta": 0, "mejor_delta_makespan": None, "peor_delta_makespan": None,
        }
        self.cp = {"ejecuciones": 0, "factibles": 0, "mejoran_incumbente": 0, "suma_aprox": 0.0, "suma_real": 0.0}
    def mejor_solucion_factible(self):
        return self.inc.mejor

    @property
    def metricas_actuales(self):
        return self._metricas(self.solucion_actual)

    @property
    def metricas_mejor(self):
        return self._metricas(self.inc.mejor)

    @staticmethod
    def _metricas(ev):
        if ev is None:
            return None
        return {k: ev[k] for k in ("ms", "suma", "desbalance", "reasignadas", "fragmentos", "desviacion")}

    # -- ciclo ---------------------------------------------------------------------------------

    def ejecutar(self):
        """Corre hasta el deadline (o el máximo de iteraciones). Retorna False si no podía comenzar."""
        cx = self.cx
        mejor = self.inc.mejor
        if mejor is None or not mejor["factible"]:
            return False
        if not any(t["tipo"] != "Pontón" for t in cx["tareas"]):
            return True
        try:
            semilla_ev = ifc._evaluar_plan(cx["semilla"], cx, "Semilla")
            if semilla_ev["utilizable"] and not semilla_ev["factible"]:
                self._anotar_conflicto(cx["semilla"])
        except Exception:
            pass
        while time.monotonic() < cx["limite"] - 0.02:
            if self.max_iteraciones is not None and self.iteraciones >= self.max_iteraciones:
                break
            try:
                self._iterar()
            except TimeoutError:
                break
        return True

    def _anotar_conflicto(self, df):
        try:
            bloque = ifc._bloque_conflicto(df, self.cx, 0)
        except Exception:
            return
        if bloque:
            self.conflictos.update(bloque["afectadas"])

    def _elegir_destructor_y_reparador(self):
        diversificando = self.sin_mejora >= UMBRAL_DIVERSIFICACION
        pesos_d = dict(self.pesos_destruccion)
        if diversificando:
            pesos_d["random_block_removal"] *= 3
        if self.foco in pesos_d:
            pesos_d[self.foco] *= 2
        return elegir_ponderado(pesos_d, self.rnd), elegir_ponderado(self.pesos_reparacion, self.rnd)

    def _iterar(self):
        cx, rnd, inc = self.cx, self.rnd, self.inc
        self.iteraciones += 1
        inicio_iteracion = time.monotonic()
        df = self.solucion_actual["df"]
        ms_actual = self.solucion_actual["ms"]
        ms_mejor_antes = inc.mejor["ms"]
        elegibles = contar_originales(df, cx)
        diversificando = self.sin_mejora >= UMBRAL_DIVERSIFICACION
        n = tamano_vecindad(elegibles, self.porcentaje_actual, diversificando)
        _, n_max = limites_vecindad(elegibles, diversificando)
        n = min(n_max, n + rnd.randint(0, 1))

        movimiento, reparador_forzado, destructor_forzado = None, None, None
        en_intensificacion = bool(self.cola_intensificacion)
        if en_intensificacion:
            destructor_forzado, reparador_forzado = self.cola_intensificacion.pop(0)
            if reparador_forzado in MOVIMIENTOS_LOCALES:
                movimiento = reparador_forzado
            elif reparador_forzado == "local_cp_sat_repair" and "local_cp_sat_repair" not in self.pesos_reparacion:
                reparador_forzado = "regret2_repair"
        if movimiento is None:
            if destructor_forzado:
                destructor, reparador = destructor_forzado, reparador_forzado
            else:
                destructor, reparador = self._elegir_destructor_y_reparador()
                if rnd.random() < PROBABILIDAD_MOVIMIENTO_LOCAL:
                    movimiento = rnd.choice(sorted(MOVIMIENTOS_LOCALES))
            if reparador == "local_cp_sat_repair" and not en_intensificacion and not (
                self.sin_mejora >= UMBRAL_CP_ESTANCAMIENTO and self.iteraciones - self.ultimo_cp >= SEPARACION_CP
            ):
                reparador = "regret2_repair"
        else:
            destructor, reparador = "movimiento_local", movimiento
        liberados_originales, propagadas, liberados = [], set(), set()
        candidata, info, error, estado_cp = None, {}, None, None
        try:
            if movimiento:
                destructor, reparador = "movimiento_local", movimiento
                plan = MOVIMIENTOS_LOCALES[movimiento](df, cx, rnd, self)
                if plan is None:
                    error = "el movimiento no aplica a esta solución"
                else:
                    liberados_originales = list(plan["originales"])
                    candidata, liberados, propagadas = aplicar_movimiento(cx, df, plan)
            else:
                liberados_originales = DESTRUCTORES[destructor](df, cx, rnd, n, self)
                if not liberados_originales:
                    raise FalloReparacion("el destructor no devolvió tareas")
                liberados, propagadas = cierre_sucesoras(liberados_originales, cx)
                if reparador == "local_cp_sat_repair" and not cp_local_elegible(cx, liberados_originales, liberados):
                    reparador = "regret2_repair"
                candidata, info = REPARADORES[reparador](cx, df, liberados, liberados_originales, rnd, self)
                if reparador == "local_cp_sat_repair":
                    self.ultimo_cp = self.iteraciones
        except TimeoutError:
            raise
        except FalloReparacion as e:
            error, estado_cp = f"descartada: {e}", e.estado
            self.conflictos.update(liberados_originales)
        except ValueError as e:
            error = f"descartada: no hay reinserción posible ({e})"
            self.conflictos.update(liberados_originales)
        except Exception as e:
            error = f"fallo inesperado: {type(e).__name__}: {e}"
            ifc._fallo(cx, f"ALNS {destructor}/{reparador}", e)
        estado_cp = info.get("estado_cpsat", estado_cp)
        if estado_cp:
            self.estados_cpsat[estado_cp] += 1
            if reparador == "local_cp_sat_repair":
                self.cp["ejecuciones"] += 1

        s_d = self.estadisticas["destruccion"].get(destructor) or self.estadisticas["destruccion"].setdefault(
            destructor, _estadistica_vacia()
        )
        grupo_r = "movimientos" if movimiento else "reparacion"
        s_r = self.estadisticas[grupo_r][reparador]
        s_d["usos"] += 1
        s_r["usos"] += 1
        self.candidatas += 1

        aceptada, motivo, categoria, ms_candidato, factible = False, error or "", "fallo", None, False
        if candidata is not None:
            s_d["reconstrucciones"] += 1
            s_r["reconstrucciones"] += 1
            ev = None
            repetida = False
            try:
                firma = firma_logica(candidata)
                ev = self.visitadas.get(firma)
                if ev is not None:
                    self.visitadas.move_to_end(firma)
                    self.diversidad["candidatas_repetidas"] += 1
                    repetida = True
                else:
                    ev = ifc._evaluar_plan(candidata, cx, f"ALNS ({destructor} + {reparador})")
                    self.visitadas[firma] = ev
                    while len(self.visitadas) > MAX_VISITADAS:
                        self.visitadas.popitem(last=False)
            except Exception as e:
                ev, motivo = None, f"validación falló: {type(e).__name__}: {e}"
                ifc._fallo(cx, "ALNS validación", e)
            if ev is not None:
                self._medir_diversidad(df, ms_actual, ev, repetida)
                ms_candidato, factible = ev["ms"], ev["factible"]
                if not ev["utilizable"]:
                    categoria, motivo = "infactible", "descartada: " + "; ".join(ev["val"]["errores"])
                elif not ev["factible"]:
                    categoria, motivo = "infactible", "infactible: " + "; ".join(ev["val"]["errores"])
                    self._anotar_conflicto(candidata)
                else:
                    s_d["factibles"] += 1
                    s_r["factibles"] += 1
                    self.candidatas_factibles += 1
                    if self.mejor_ms_propio is None or ev["ms"] < self.mejor_ms_propio:
                        self.mejor_ms_propio = ev["ms"]
                    aceptada, motivo, categoria = self._decidir(ev, ms_actual, s_d, s_r)
                    if ev["ms"] > ms_actual + 1e-6 and aceptada:
                        self.diversidad["candidatas_peores_aceptadas"] += 1
                if "ms_cp_aprox" in info:
                    self._medir_cp(info["ms_cp_aprox"], ev, categoria)
        elif error and not error.startswith(("descartada", "el movimiento")):
            categoria = "fallo"
        if categoria == "fallo":
            s_d["fallos"] += 1
            s_r["fallos"] += 1
        elif categoria == "infactible":
            pass

        puntos = PUNTOS.get(categoria, 0.0)
        for grupo, nombre in (("destruccion", destructor), ("reparacion", reparador)):
            self.puntos[grupo][nombre].append(puntos)
        transcurrido = time.monotonic() - inicio_iteracion
        s_d["tiempo_s"] += transcurrido
        s_r["tiempo_s"] += transcurrido

        if categoria != "mejor":
            self.sin_mejora += 1
            if self.sin_mejora % 10 == 0:
                tope = MAXIMO_DIVERSIFICACION if self.sin_mejora >= UMBRAL_DIVERSIFICACION else MAXIMO_NORMAL
                self.porcentaje_actual = min(self.porcentaje_actual * 1.3, tope)
            if self.sin_mejora >= LIMITE_REINICIO:
                self._reiniciar()
        self.temperatura = max(TEMPERATURA_MINIMA, self.temperatura * ENFRIAMIENTO)
        if self.iteraciones % SEGMENTO_PESOS == 0:
            actualizar_pesos(self.pesos_destruccion, self.puntos["destruccion"])
            actualizar_pesos(self.pesos_reparacion, self.puntos["reparacion"])
            self.puntos = {"destruccion": defaultdict(list), "reparacion": defaultdict(list)}

        self._registrar(
            destructor, reparador, n, liberados_originales, propagadas, liberados, estado_cp, ms_actual,
            ms_candidato, ms_mejor_antes, factible, aceptada, motivo,
        )

    def _decidir(self, ev, ms_actual, s_d, s_r):
        """Aceptación de una candidata factible. Retorna (aceptada, motivo, categoría para los puntos)."""
        inc, rnd = self.inc, self.rnd
        previo = inc.mejor["ms"]
        mejora_global, motivo_global = inc.ofrecer(ev)
        if mejora_global:
            reduccion = max(0.0, previo - ev["ms"])
            for s in (s_d, s_r):
                s["aceptadas"] += 1
                s["nuevas_mejores"] += 1
                s["reduccion_makespan"] += reduccion
            self.candidatas_aceptadas += 1
            self.mejoras += 1
            self.solucion_actual = ev
            self.sin_mejora = 0
            self.porcentaje_actual = self.porcentaje_base
            self.respaldo_mejor = _copia_profunda(ev)
            self.cola_intensificacion = list(INTENSIFICACION)
            return True, "nueva mejor solución factible: " + motivo_global, "mejor"
        if firma_solucion(ev["df"]) == firma_solucion(self.solucion_actual["df"]):
            return False, "sin cambios respecto de la solución actual", "factible"
        mejor, motivo = ifc._comparar_planes(ev, self.solucion_actual)
        if mejor:
            reduccion = max(0.0, ms_actual - ev["ms"])
            for s in (s_d, s_r):
                s["aceptadas"] += 1
                s["mejoras_actual"] += 1
                s["reduccion_makespan"] += reduccion
            self.candidatas_aceptadas += 1
            self.solucion_actual = ev
            return True, "mejora la solución actual: " + motivo, "mejora"
        if self.usar_sa:
            delta = ev["ms"] - ms_actual
            probabilidad = 1.0 if delta <= 0 else math.exp(-delta / max(self.temperatura, TEMPERATURA_MINIMA))
            if rnd.random() < probabilidad:
                for s in (s_d, s_r):
                    s["aceptadas"] += 1
                self.candidatas_aceptadas += 1
                self.solucion_actual = ev
                return True, f"aceptada por recocido (Δ={delta:.1f} min, p={probabilidad:.2f})", "aceptada"
        return False, "factible rechazada: " + motivo, "factible"

    def _medir_diversidad(self, df_actual, ms_actual, ev, repetida=False):
        d = self.diversidad
        d["candidatas_evaluadas"] += 1
        df = ev["df"]
        identica = firma_logica(df) == firma_logica(df_actual)
        if identica or repetida:
            d["candidatas_identicas"] += 1
        if identica:
            d["candidatas_sin_cambio"] += 1
            return
        if _asignacion(df) != _asignacion(df_actual):
            d["candidatas_con_reasignacion"] += 1
        if _orden_por_grua(df) != _orden_por_grua(df_actual):
            d["candidatas_con_cambio_orden"] += 1
        if df.groupby("Parent_Task_ID").size().to_dict() != df_actual.groupby("Parent_Task_ID").size().to_dict():
            d["candidatas_con_cambio_fragmentacion"] += 1
        if not ev["factible"]:
            return
        delta = ev["ms"] - ms_actual
        if abs(delta) > 1e-6:
            d["candidatas_con_makespan_distinto"] += 1
        if delta < -1e-6:
            d["candidatas_mejoran_makespan"] += 1
        d["_suma_delta"] += delta
        d["_n_delta"] += 1
        d["mejor_delta_makespan"] = delta if d["mejor_delta_makespan"] is None else min(d["mejor_delta_makespan"], delta)
        d["peor_delta_makespan"] = delta if d["peor_delta_makespan"] is None else max(d["peor_delta_makespan"], delta)

    def _medir_cp(self, ms_aprox, ev, categoria):
        if not ev["factible"]:
            return
        self.cp["factibles"] += 1
        self.cp["suma_aprox"] += ms_aprox
        self.cp["suma_real"] += ev["ms"]
        if categoria == "mejor":
            self.cp["mejoran_incumbente"] += 1

    def estadisticas_diversidad(self):
        d = {k: v for k, v in self.diversidad.items() if not k.startswith("_")}
        n = d["candidatas_evaluadas"]
        d["delta_makespan_promedio"] = self.diversidad["_suma_delta"] / self.diversidad["_n_delta"] \
            if self.diversidad["_n_delta"] else None
        idénticas = d["candidatas_identicas"]
        d["candidatas_distintas"] = max(0, n - idénticas)
        d["porcentaje_candidatas_identicas"] = 100.0 * min(idénticas, n) / n if n else 0.0
        return d

    def estadisticas_cp(self):
        c = self.cp
        f = c["factibles"]
        return {
            "ejecuciones": c["ejecuciones"], "factibles_tras_reconstruir": f,
            "mejoran_incumbente": c["mejoran_incumbente"],
            "ms_aproximado_promedio": c["suma_aprox"] / f if f else None,
            "ms_real_promedio": c["suma_real"] / f if f else None,
            "diferencia_promedio_min": (c["suma_real"] - c["suma_aprox"]) / f if f else None,
        }

    def _reiniciar(self):
        """Vuelve a la mejor solución conservando pesos, estadísticas e historial y cambia el foco."""
        self.solucion_actual = self.inc.mejor
        self.sin_mejora = 0
        self.reinicios += 1
        self.porcentaje_actual = self.porcentaje_base
        self.temperatura = max(self.temperatura, 0.01 * (self.primera_solucion_factible["ms"] or 1.0))
        self.foco = self.rnd.choice(sorted(DESTRUCTORES))

    def _registrar(self, destructor, reparador, n, originales, propagadas, liberados, estado_cp, ms_actual,
                   ms_candidato, ms_mejor, factible, aceptada, motivo):
        cx = self.cx
        cx["iteracion"] += 1
        if len(cx["historial"]) >= MAX_HISTORIAL and not aceptada:
            return
        df = self.solucion_actual["df"]
        cx["historial"].append({
            "iteracion": cx["iteracion"], "fase": "ALNS", "operador_destruccion": destructor,
            "operador_reparacion": reparador, "tamano_vecindad": n,
            "tareas_liberadas": list(originales), "tareas_propagadas": sorted(propagadas),
            "fragmentos_afectados": int(df["Parent_Task_ID"].isin(list(liberados)).sum()) if liberados else 0,
            "estado_cpsat": estado_cp, "makespan_actual": round(ms_actual, 1),
            "makespan_candidato": None if ms_candidato is None else round(ms_candidato, 1),
            "mejor_makespan": round(self.inc.mejor["ms"], 1), "factible": factible, "aceptada": aceptada,
            "motivo": motivo, "temperatura": round(self.temperatura, 3),
            "tiempo_s": round(time.monotonic() - cx["t0"], 2),
        })

    # -- resultados ------------------------------------------------------------------------------

    def ganador(self, grupo):
        stats = self.estadisticas[grupo]
        mejor = max(stats, key=lambda n: (stats[n]["nuevas_mejores"], stats[n]["reduccion_makespan"], n))
        return mejor if stats[mejor]["nuevas_mejores"] > 0 else None

    def resumen(self):
        return {
            "alns_activado": True, "iteraciones_alns": self.iteraciones, "candidatas_alns": self.candidatas,
            "candidatas_factibles_alns": self.candidatas_factibles,
            "candidatas_aceptadas_alns": self.candidatas_aceptadas, "mejoras_alns": self.mejoras,
            "operador_destruccion_ganador": self.ganador("destruccion"),
            "operador_reparacion_ganador": self.ganador("reparacion") or self.ganador("movimientos"),
            "pesos_destruccion": {k: round(v, 4) for k, v in self.pesos_destruccion.items()},
            "pesos_reparacion": {k: round(v, 4) for k, v in self.pesos_reparacion.items()},
            "estadisticas_operadores": self.estadisticas, "temperatura_final": round(self.temperatura, 4),
            "reinicios_alns": self.reinicios, "estados_cpsat_local": dict(self.estados_cpsat),
            "estadisticas_diversidad": self.estadisticas_diversidad(), "estadisticas_cp_local": self.estadisticas_cp(),
            "makespan_inicio_alns": self.ms_inicial, "makespan_mejor_candidata_alns": self.mejor_ms_propio,
        }


def ejecutar_alns(cx, inc, rnd, porcentaje=0.10, usar_sa=True, usar_cp=True, max_iteraciones=None):
    """Ejecuta el ALNS sobre la mejor solución factible del incumbente. Retorna el EstadoALNS."""
    estado = EstadoALNS(cx, inc, rnd, porcentaje, usar_sa, usar_cp, max_iteraciones)
    estado.iniciado = estado.ejecutar()
    return estado
