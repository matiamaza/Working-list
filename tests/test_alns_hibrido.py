import copy
import inspect
import random
import time

import pandas as pd
import pytest

import alns
import interferencias as ifc
from tests.test_alns import EstadoFalso, contexto, plan_factible
from tests.test_alns_bucle import correr, evaluar, movs
from tests.test_interferencias import config, config_gmph, cronograma, fila, intervalo_fn


def semilla_cadenas():
    """Cadenas por bay (misma clave, orden de la semilla) repartidas entre grúas de distinto rendimiento."""
    return cronograma(
        fila("G1", 2, 200, 0), fila("G1", 2, 200, 400), fila("G1", 14, 60, 800), fila("G1", 14, 60, 1000),
        fila("G2", 30, 20, 0), fila("G2", 30, 60, 100), fila("G2", 14, 60, 300), fila("G2", 34, 40, 500),
    )


CFG_CADENAS = config_gmph(30, 60)


def ev_falso(base, ms, metodo, secundario=None):
    ev = copy.copy(base)
    ev["ms"] = ms
    ev["metodo"] = metodo
    if secundario is not None:
        ev["secundario"] = secundario
    return ev


@pytest.fixture
def espia(monkeypatch):
    llamadas = []
    original = ifc._mejorar_local

    def espiar(cx, inc, rnd):
        llamadas.append(inc.mejor["ms"] if inc.mejor is not None else None)
        return original(cx, inc, rnd)

    monkeypatch.setattr(ifc, "_mejorar_local", espiar)
    return llamadas


def hibrido(df, cfg=None, **kw):
    kw.setdefault("tiempo_limite", 4)
    kw.setdefault("max_iteraciones_alns", 30)
    kw.setdefault("usar_cpsat_local", False)
    return ifc.proponer_secuencia(df, cfg or config("G1", "G2"), intervalo_fn, 15, **kw)


# --- garantía de no regresión ------------------------------------------------------------------

def _incumbente_base():
    cx = contexto()
    inc = ifc._Incumbente()
    inc.ofrecer(evaluar(plan_factible(cx), cx))
    return cx, inc


def test_un_resultado_peor_no_reemplaza_al_incumbente():
    _, inc = _incumbente_base()
    mejor = inc.mejor
    peor = ev_falso(mejor, mejor["ms"] + 2, "ALNS_480")
    aceptada, _ = inc.ofrecer(peor)
    assert not aceptada and inc.mejor is mejor


def test_empate_de_makespan_decide_el_criterio_secundario():
    _, inc = _incumbente_base()
    base = inc.mejor
    a = ev_falso(base, 478.0, "A", secundario=(100.0, 10.0, 2, 1, 5.0))
    b = ev_falso(base, 478.5, "B", secundario=(90.0, 10.0, 2, 1, 5.0))
    c = ev_falso(base, 478.0, "C", secundario=(120.0, 10.0, 2, 1, 5.0))
    assert ifc._comparar_planes(b, a)[0] and not ifc._comparar_planes(c, a)[0]
    assert not ifc._comparar_planes(ev_falso(base, 478.0, "D", secundario=a["secundario"]), a)[0]


def test_mejora_local_478_y_alns_480_devuelve_478(monkeypatch):
    marcas = {}
    original = ifc._mejorar_local

    def local_falsa(cx, inc, rnd):
        original(cx, inc, rnd)
        if "local" not in marcas:
            marcas["local"] = inc.ofrecer(ev_falso(inc.mejor, 478.0, "MEJORA_LOCAL_478"))[0]

    def alns_falso(cx, inc, rnd, porcentaje, sa, cp, max_it):
        estado = alns.EstadoALNS(cx, inc, rnd, max_iteraciones=0)
        estado.iniciado = True
        marcas["alns_acepta"] = inc.ofrecer(ev_falso(inc.mejor, 480.0, "ALNS_480"))[0]
        return estado

    monkeypatch.setattr(ifc, "_mejorar_local", local_falsa)
    monkeypatch.setattr(alns, "ejecutar_alns", alns_falso)
    p = hibrido(semilla_cadenas(), CFG_CADENAS)
    assert marcas["local"] and not marcas["alns_acepta"]
    assert p.attrs["metodo"] == "MEJORA_LOCAL_478"
    assert p.attrs["makespan_antes_alns_min"] == 478.0
    assert p.attrs["makespan_tras_alns_min"] == 478.0


def test_la_solucion_final_es_la_mejor_de_todas_las_fases():
    p = hibrido(semilla_cadenas(), CFG_CADENAS)
    a = p.attrs
    fases = [a["makespan_inicial_candidatos_min"], a["makespan_antes_alns_min"], a["makespan_tras_alns_min"],
             a["makespan_tras_intensificacion_min"]]
    assert all(f is not None for f in fases[:3])
    assert a["makespan_final_min"] <= min(f for f in fases if f is not None) + ifc.TOLERANCIA_MAKESPAN_MIN
    assert a["factible"]


def test_la_intensificacion_final_no_empeora():
    p = hibrido(semilla_cadenas(), CFG_CADENAS)
    a = p.attrs
    assert a["makespan_tras_intensificacion_min"] <= a["makespan_tras_alns_min"] + 1e-6
    assert a["makespan_final_min"] <= a["makespan_tras_alns_min"] + ifc.TOLERANCIA_MAKESPAN_MIN


def test_con_alns_el_resultado_no_es_peor_que_el_mejor_candidato_previo():
    for semilla in (42, 43):
        p = hibrido(semilla_cadenas(), CFG_CADENAS, random_seed=semilla)
        assert p.attrs["makespan_final_min"] <= p.attrs["makespan_antes_alns_min"] + ifc.TOLERANCIA_MAKESPAN_MIN


# --- fases y presupuesto -------------------------------------------------------------------------

def test_alns_activo_no_omite_la_mejora_local_inicial(espia):
    p = hibrido(semilla_cadenas(), CFG_CADENAS)
    assert len(espia) >= 1
    assert "mejora_local_inicial" in p.attrs["tiempos_fases_seg"]
    assert p.attrs["iteraciones_alns"] > 0


def test_desactivar_la_mejora_local_inicial_solo_quita_esa_fase(espia):
    p = hibrido(semilla_cadenas(), CFG_CADENAS, activar_mejora_local_inicial=False)
    assert "mejora_local_inicial" not in p.attrs["tiempos_fases_seg"]
    assert "intensificacion_final" in p.attrs["tiempos_fases_seg"]
    assert p.attrs["iteraciones_alns"] > 0 and len(espia) == 1


def test_desactivar_la_intensificacion_final_solo_quita_esa_fase(espia):
    p = hibrido(semilla_cadenas(), CFG_CADENAS, activar_intensificacion_final=False)
    assert "intensificacion_final" not in p.attrs["tiempos_fases_seg"]
    assert "mejora_local_inicial" in p.attrs["tiempos_fases_seg"]
    assert p.attrs["iteraciones_alns"] > 0 and len(espia) == 1


def test_sin_alns_se_mantiene_el_flujo_de_la_etapa_2(espia):
    p = hibrido(semilla_cadenas(), CFG_CADENAS, activar_alns=False)
    assert len(espia) == 1
    assert p.attrs["alns_activado"] is False and p.attrs["iteraciones_alns"] == 0
    assert not [h for h in p.attrs["historial"] if h["fase"] == "ALNS"]
    assert "alns" not in p.attrs["tiempos_fases_seg"]


def test_el_presupuesto_global_se_respeta_con_todas_las_fases():
    t0 = time.monotonic()
    p = ifc.proponer_secuencia(semilla_cadenas(), CFG_CADENAS, intervalo_fn, 15, tiempo_limite=3)
    assert time.monotonic() - t0 < 3 + 2.5
    assert p.attrs["tiempo_total_seg"] <= 3 + 2.5


def test_el_tiempo_no_usado_pasa_al_alns():
    p = ifc.proponer_secuencia(semilla_cadenas(), CFG_CADENAS, intervalo_fn, 15, tiempo_limite=4)
    t = p.attrs["tiempos_fases_seg"]
    assert t["alns"] >= 0.3 * 4
    assert sum(t.values()) <= p.attrs["tiempo_total_seg"] + 0.2


def test_presupuesto_configurable_y_parametros_nuevos_compatibles():
    firma = list(inspect.signature(ifc.proponer_secuencia).parameters)
    esperados = ["df_cronograma", "config_gruas", "intervalo_fn", "holgura", "tiempo_limite", "random_seed",
                 "workers", "activar_reparacion_iterativa", "activar_mejora_local", "activar_alns",
                 "porcentaje_vecindad", "permitir_reasignacion", "permitir_division",
                 "usar_simulated_annealing", "orden_fisico_gruas", "debug"]
    assert firma[:len(esperados)] == esperados
    for nombre in ("activar_mejora_local_inicial", "activar_intensificacion_final", "usar_cpsat_local"):
        assert inspect.signature(ifc.proponer_secuencia).parameters[nombre].default is True
    p = hibrido(semilla_cadenas(), CFG_CADENAS, presupuesto_fases={"final": 0.1})
    assert p.attrs["factible"]


def test_grua_sin_tareas_sigue_siendo_valida_en_el_hibrido():
    p = hibrido(cronograma(fila("G1", 2, 30, 0)), config("G1", "G2"), max_iteraciones_alns=10)
    assert p.attrs["factible"] and p.attrs["gruas_disponibles"] == 2
    assert p.attrs["gruas_utilizadas"] >= 1


# --- precedencias en el bucle completo -------------------------------------------------------

def _resultado_cadenas(**kw):
    df = semilla_cadenas()
    return df, hibrido(df, CFG_CADENAS, **kw)


def test_precedencias_del_resultado_completo_con_alns_activo():
    df, p = _resultado_cadenas(max_iteraciones_alns=60)
    assert p.attrs["alns_activado"] and p.attrs["factible"]
    assert ifc.verificar_precedencias(p, df) == []
    assert p.attrs["precedencias_validas"]
    assert p.attrs["tareas_reasignadas"] >= 1 and p.attrs["fragmentos_creados"] >= 1
    original = ifc._asegurar_identidad(df)
    cadenas = ifc.construir_precedencias(ifc._tareas_desde_cronograma(original))
    gruas = p.groupby("Parent_Task_ID")["Grúa"].agg(lambda s: set(s))
    assert any(gruas[a] != gruas[b] or len(gruas[a] | gruas[b]) > 1 for a, b in cadenas)
    inicio = pd.to_datetime(p["Inicio"]).groupby(p["Parent_Task_ID"]).min()
    fin = pd.to_datetime(p["Fin"]).groupby(p["Parent_Task_ID"]).max()
    for a, b in cadenas:
        assert fin[a] <= inicio[b]


def test_precedencias_se_conservan_con_division():
    df, p = _resultado_cadenas(permitir_division=True, permitir_reasignacion=True)
    assert p.groupby("Parent_Task_ID").size().max() >= 1
    assert ifc.verificar_precedencias(p, df) == [] and movs(p) == movs(ifc._asegurar_identidad(df))


def test_precedencias_se_conservan_con_reasignacion():
    df, p = _resultado_cadenas(permitir_division=False)
    assert p.attrs["tareas_reasignadas"] >= 1
    assert ifc.verificar_precedencias(p, df) == []


def test_el_orden_heredado_se_mantiene_sin_reasignacion():
    df, p = _resultado_cadenas(permitir_reasignacion=False, permitir_division=False)
    assert ifc.verificar_precedencias(p, df) == []
    assert p.attrs["tareas_reasignadas"] == 0


def _invertida(df, cx):
    a, b = cx["aristas"][0]
    ini_a = pd.to_datetime(df[df["Parent_Task_ID"] == a]["Inicio"]).min()
    ini_b = pd.to_datetime(df[df["Parent_Task_ID"] == b]["Inicio"]).min()
    inv = df.copy(deep=True)
    for col in ("Inicio", "Fin"):
        inv[col] = pd.to_datetime(inv[col])
    mascara = inv["Parent_Task_ID"] == b
    for col in ("Inicio", "Fin"):
        inv.loc[mascara, col] = inv.loc[mascara, col] - (ini_b - (ini_a - pd.Timedelta(minutes=10)))
    return inv


def test_una_candidata_alns_que_invierte_el_orden_se_rechaza(monkeypatch):
    semilla = semilla_cadenas()
    cx = contexto(semilla)
    cx["config"] = CFG_CADENAS
    inc = ifc._Incumbente()
    inc.ofrecer(evaluar(plan_factible(cx), cx))
    estado = alns.EstadoALNS(cx, inc, random.Random(1))
    base = inc.mejor
    inv = _invertida(base["df"], cx)
    assert ifc.verificar_precedencias(inv, semilla) != []
    assert not evaluar(inv, cx)["utilizable"]
    monkeypatch.setattr(alns, "PROBABILIDAD_MOVIMIENTO_LOCAL", 0.0)
    monkeypatch.setattr(estado, "_elegir_destructor_y_reparador", lambda: ("critical_crane_removal", "greedy_repair"))
    monkeypatch.setitem(alns.REPARADORES, "greedy_repair", lambda *a, **k: (inv, {}))
    antes = estado.candidatas_factibles
    estado._iterar()
    assert inc.mejor is base and estado.solucion_actual is base
    assert estado.candidatas_factibles == antes
    assert cx["historial"][-1]["aceptada"] is False


def test_las_reparaciones_y_movimientos_no_rompen_aristas():
    semilla = semilla_cadenas()
    cx = contexto(semilla)
    aristas = list(cx["aristas"])
    df = plan_factible(cx)
    estado = EstadoFalso()
    for nombre, reparador in alns.REPARADORES.items():
        if nombre == "local_cp_sat_repair":
            continue
        for s in range(4):
            rnd = random.Random(s)
            orig = alns.DESTRUCTORES["related_task_removal"](df, cx, rnd, 3, estado)
            lib, _ = alns.cierre_sucesoras(orig, cx)
            try:
                cand, _ = reparador(cx, df, lib, orig, rnd, estado)
            except (alns.FalloReparacion, ValueError):
                continue
            assert ifc.verificar_precedencias(cand, semilla) == []
    for mov in alns.MOVIMIENTOS_LOCALES.values():
        for s in range(4):
            plan = mov(df, cx, random.Random(s), estado)
            if plan is None:
                continue
            try:
                cand, _, _ = alns.aplicar_movimiento(cx, df, plan)
            except (alns.FalloReparacion, ValueError):
                continue
            assert ifc.verificar_precedencias(cand, semilla) == []
    assert cx["aristas"] == aristas


def test_cp_local_respeta_precedencias_externas():
    semilla = semilla_cadenas()
    cx = contexto(semilla)
    cx["config"] = CFG_CADENAS
    df = plan_factible(cx)
    pontones = {t["id"] for t in cx["tareas"] if t["tipo"] == "Pontón"}
    a, b = cx["aristas"][0]
    otra = next(t["id"] for t in cx["tareas"] if t["id"] not in {a, b} | pontones)
    orig = [b, otra]
    lib, _ = alns.cierre_sucesoras(orig, cx)
    assert a not in lib  # la predecesora queda congelada: precedencia externa
    assert alns.cp_local_elegible(cx, orig, lib)
    cand, info = alns.REPARADORES["local_cp_sat_repair"](cx, df, lib, orig, random.Random(4), EstadoFalso())
    assert info["estado_cpsat"] in {"OPTIMAL", "FEASIBLE"} and "ms_cp_aprox" in info
    assert ifc.verificar_precedencias(cand, semilla) == []
    assert movs(cand) == movs(df)


# --- firmas, repetidas y diversidad -------------------------------------------------------------

def test_las_firmas_detectan_candidatas_identicas():
    cx = contexto()
    df = plan_factible(cx)
    barajada = df.sample(frac=1, random_state=3).reset_index(drop=True)
    assert alns.firma_logica(df) == alns.firma_logica(barajada)
    otra = df.copy(deep=True)
    otra["Inicio"] = pd.to_datetime(otra["Inicio"])
    otra.loc[otra.index[0], "Inicio"] += pd.Timedelta(minutes=1)
    assert alns.firma_logica(df) != alns.firma_logica(otra)
    cambio_grua = df.copy(deep=True)
    cambio_grua.loc[cambio_grua.index[0], "Grúa"] = "GX"
    assert alns.firma_logica(df) != alns.firma_logica(cambio_grua)


def test_las_candidatas_identicas_no_se_evaluan_de_nuevo(monkeypatch):
    cx = contexto()
    inc = ifc._Incumbente()
    inc.ofrecer(evaluar(plan_factible(cx), cx))
    estado = alns.EstadoALNS(cx, inc, random.Random(2))
    df = inc.mejor["df"]
    llamadas = []
    original = ifc._evaluar_plan
    monkeypatch.setattr(ifc, "_evaluar_plan", lambda *a, **k: llamadas.append(1) or original(*a, **k))
    monkeypatch.setattr(alns, "PROBABILIDAD_MOVIMIENTO_LOCAL", 0.0)
    monkeypatch.setattr(estado, "_elegir_destructor_y_reparador", lambda: ("critical_crane_removal", "greedy_repair"))
    monkeypatch.setitem(alns.REPARADORES, "greedy_repair", lambda *a, **k: (df.sample(frac=1).reset_index(drop=True), {}))
    for _ in range(6):
        estado._iterar()
    assert not llamadas
    d = estado.estadisticas_diversidad()
    assert d["candidatas_repetidas"] == 6 and d["candidatas_sin_cambio"] == 6
    assert d["candidatas_distintas"] == 0 and d["porcentaje_candidatas_identicas"] == 100.0


def test_el_tamano_de_vecindad_usa_parent_task_id():
    df = semilla_cadenas()
    p = hibrido(df, CFG_CADENAS, max_iteraciones_alns=5)
    assert len(p) > p["Parent_Task_ID"].nunique()
    cx = contexto(df)
    assert alns.contar_originales(p, cx) == p["Parent_Task_ID"].nunique()
    assert alns.limites_vecindad(8) == (2, 4)
    assert alns.limites_vecindad(20) == (3, 8)
    assert alns.limites_vecindad(50) == (3, 10)
    assert alns.limites_vecindad(200) == (10, 30)
    assert alns.limites_vecindad(200, True) == (10, 50)
    assert alns.tamano_vecindad(200, 0.5) == 30 and alns.tamano_vecindad(200, 0.0) == 10


def test_las_estadisticas_de_diversidad_son_coherentes():
    p = hibrido(semilla_cadenas(), CFG_CADENAS, max_iteraciones_alns=40)
    d = p.attrs["estadisticas_diversidad"]
    assert d["candidatas_evaluadas"] >= d["candidatas_identicas"] >= d["candidatas_sin_cambio"] >= 0
    assert d["candidatas_distintas"] == d["candidatas_evaluadas"] - d["candidatas_identicas"]
    assert 0.0 <= d["porcentaje_candidatas_identicas"] <= 100.0
    if d["delta_makespan_promedio"] is not None:
        assert d["mejor_delta_makespan"] <= d["delta_makespan_promedio"] + 1e-9 <= d["peor_delta_makespan"] + 2e-9
    for clave in ("candidatas_sin_cambio", "candidatas_con_reasignacion", "candidatas_con_cambio_orden",
                  "candidatas_con_cambio_fragmentacion", "candidatas_con_makespan_distinto",
                  "delta_makespan_promedio", "mejor_delta_makespan", "peor_delta_makespan"):
        assert clave in d
    cp = p.attrs["estadisticas_cp_local"]
    assert {"ejecuciones", "factibles_tras_reconstruir", "mejoran_incumbente"} <= set(cp)


def test_cp_local_no_se_ejecuta_en_cada_iteracion():
    p = hibrido(semilla_cadenas(), CFG_CADENAS, usar_cpsat_local=True, max_iteraciones_alns=60)
    cp = p.attrs["estadisticas_cp_local"]
    assert cp["ejecuciones"] <= p.attrs["iteraciones_alns"] // 2
    assert p.attrs["pesos_reparacion"]["local_cp_sat_repair"] <= 1.0
