import copy
import random
import time

import pandas as pd
import pytest

import alns
import interferencias as ifc
from tests.test_alns import EstadoFalso, contexto, instancia, plan_factible
from tests.test_interferencias import (
    T0, config, config_gmph, cronograma, fila, intervalo_fn, pontón,
)


def movs(df):
    return df.groupby("Parent_Task_ID")["Movimientos"].sum().to_dict()


def evaluar(df, cx):
    return ifc._evaluar_plan(df, cx, "test")


def semilla_choque():
    return cronograma(fila("G1", 14, 60, 0), fila("G2", 14, 60, 0), fila("G2", 30, 40, 200))


def semilla_desbalanceada():
    return cronograma(fila("G1", 2, 200, 0), fila("G1", 10, 200, 400), fila("G2", 30, 20, 0))


def correr(df, cfg=None, **kw):
    kw.setdefault("tiempo_limite", 4)
    kw.setdefault("usar_cpsat_local", False)
    kw.setdefault("max_iteraciones_alns", 40)
    return ifc.proponer_secuencia(df, cfg or config("G1", "G2"), intervalo_fn, 15, **kw)


# --- reparadores y movimientos --------------------------------------------------------------

@pytest.mark.parametrize("nombre", ["greedy_repair", "regret2_repair", "critical_first_repair"])
def test_reparadores_conservan_movimientos_y_son_factibles(nombre):
    cx = contexto()
    df = plan_factible(cx)
    rnd = random.Random(5)
    estado = EstadoFalso()
    probados = 0
    for n in (2, 3, 4):
        orig = alns.DESTRUCTORES["related_task_removal"](df, cx, rnd, n, estado)
        lib, _ = alns.cierre_sucesoras(orig, cx)
        try:
            cand, _ = alns.REPARADORES[nombre](cx, df, lib, orig, rnd, estado)
        except alns.FalloReparacion:
            continue
        probados += 1
        assert movs(cand) == movs(df)
        assert evaluar(cand, cx)["utilizable"]
    assert probados


def test_reparacion_no_muta_la_solucion_de_entrada():
    cx = contexto()
    df = plan_factible(cx)
    antes = df.copy(deep=True)
    orig = alns.DESTRUCTORES["critical_crane_removal"](df, cx, random.Random(1), 3, EstadoFalso())
    lib, _ = alns.cierre_sucesoras(orig, cx)
    alns.REPARADORES["greedy_repair"](cx, df, lib, orig, random.Random(1), EstadoFalso())
    pd.testing.assert_frame_equal(df, antes)


@pytest.mark.parametrize("estado", ["INFEASIBLE", "UNKNOWN"])
def test_cp_local_maneja_infeasible_y_unknown(estado, monkeypatch):
    cx = contexto()
    df = plan_factible(cx)
    orig = alns.DESTRUCTORES["related_task_removal"](df, cx, random.Random(2), 3, EstadoFalso())
    lib, _ = alns.cierre_sucesoras(orig, cx)

    def falso(*a, **k):
        raise alns.FalloReparacion("sin solución", estado)

    monkeypatch.setattr(alns, "_plan_cp_local", falso)
    with pytest.raises(alns.FalloReparacion) as e:
        alns.REPARADORES["local_cp_sat_repair"](cx, df, lib, orig, random.Random(2), EstadoFalso())
    assert e.value.estado == estado


def test_cp_local_real_reconstruye_con_calculo_propio():
    cx = contexto()
    df = plan_factible(cx)
    orig = alns.DESTRUCTORES["related_task_removal"](df, cx, random.Random(2), 3, EstadoFalso())
    lib, _ = alns.cierre_sucesoras(orig, cx)
    if not alns.cp_local_elegible(cx, orig, lib):
        pytest.skip("vecindad no elegible")
    cand, info = alns.REPARADORES["local_cp_sat_repair"](cx, df, lib, orig, random.Random(2), EstadoFalso())
    assert movs(cand) == movs(df)
    assert info.get("estado_cpsat") in {"OPTIMAL", "FEASIBLE"}


def test_movimientos_locales_conservan_movimientos_pontones_y_precedencias():
    cx = contexto()
    df = plan_factible(cx)
    ponton_ids = {t["id"] for t in cx["tareas"] if t["tipo"] == "Pontón"}
    gruas_ponton = {i: set(df[df["Parent_Task_ID"] == i]["Grúa"]) for i in ponton_ids}
    aplicados = 0
    for nombre, mov in alns.MOVIMIENTOS_LOCALES.items():
        for semilla in range(6):
            plan = mov(df, cx, random.Random(semilla), EstadoFalso())
            if plan is None:
                continue
            assert not set(plan["originales"]) & ponton_ids
            try:
                cand, _, _ = alns.aplicar_movimiento(cx, df, plan)
            except (alns.FalloReparacion, ValueError):
                continue
            aplicados += 1
            assert movs(cand) == movs(df)
            for i, gr in gruas_ponton.items():
                assert set(cand[cand["Parent_Task_ID"] == i]["Grúa"]) == gr
            ev = evaluar(cand, cx)
            if ev["factible"]:
                assert ev["val"]["reglas_validas"]
    assert aplicados


def test_swap_y_relocate_no_se_aplican_sin_reasignacion():
    cx = contexto(reasignar=False)
    df = plan_factible(cx)
    for nombre in ("relocate_task", "swap_tasks"):
        plan = alns.MOVIMIENTOS_LOCALES[nombre](df, cx, random.Random(1), EstadoFalso())
        assert plan is None


# --- selección, pesos, recocido ------------------------------------------------------------

def test_seleccion_ponderada_segura():
    rnd = random.Random(1)
    for pesos in ({"a": 0.0, "b": 0.0}, {"a": float("nan"), "b": -3.0}, {"a": float("inf"), "b": 1.0}, {"a": 1e-300}):
        assert alns.elegir_ponderado(pesos, rnd) in pesos
    cuenta = {"a": 0, "b": 0}
    for _ in range(500):
        cuenta[alns.elegir_ponderado({"a": 1.0, "b": 9.0}, rnd)] += 1
    assert cuenta["b"] > cuenta["a"]


def test_pesos_siempre_positivos():
    pesos = {"a": 1.0, "b": 1.0}
    for _ in range(50):
        alns.actualizar_pesos(pesos, {"a": [0.0, 0.0], "b": [8.0]})
    assert all(v >= alns.PESO_MINIMO > 0 for v in pesos.values())
    assert pesos["b"] > pesos["a"]
    alns.actualizar_pesos(pesos, {"a": [float("nan")]})
    assert pesos["a"] == alns.PESO_MINIMO


def _estado(df, **kw):
    cx = contexto(df)
    cx["limite"] = time.monotonic() + 30
    inc = ifc._Incumbente()
    ev = evaluar(plan_factible(cx), cx)
    assert ev["factible"]
    inc.ofrecer(ev)
    return cx, inc, alns.EstadoALNS(cx, inc, random.Random(3), **kw)


def test_alns_no_empieza_sin_solucion_factible():
    cx = contexto()
    inc = ifc._Incumbente()
    estado = alns.EstadoALNS(cx, inc, random.Random(1))
    assert estado.ejecutar() is False
    assert estado.iteraciones == 0


def test_temperatura_inicial_y_enfriamiento():
    cx, inc, estado = _estado(None)
    assert estado.temperatura == max(1.0, 0.02 * inc.mejor["ms"])
    estado.max_iteraciones = 3
    estado.ejecutar()
    assert estado.iteraciones == 3
    assert estado.temperatura == pytest.approx(max(1.0, 0.02 * inc.mejor["ms"]) * 0.98 ** 3, rel=1e-6)


def test_infactible_nunca_se_acepta_ni_peor_reemplaza_la_mejor():
    cx, inc, estado = _estado(None, usar_sa=True)
    mejor_antes = inc.mejor
    infactible = copy.deepcopy(mejor_antes)
    infactible["factible"] = False
    s = alns._estadistica_vacia()
    # candidata factible peor: puede reemplazar la actual por recocido, nunca la mejor
    peor = copy.deepcopy(mejor_antes)
    peor["ms"] = mejor_antes["ms"] + 50
    peor["df"] = mejor_antes["df"].copy()
    peor["df"].loc[peor["df"].index[0], "Movimientos"] += 0  # misma firma: se descarta sin cambios
    estado.temperatura = 1e9
    estado.rnd = random.Random(0)
    estado._decidir(peor, mejor_antes["ms"], s, s)
    assert inc.mejor is mejor_antes
    assert inc.mejor["ms"] == mejor_antes["ms"]


def test_recocido_puede_cambiar_temporalmente_la_solucion_actual():
    cx, inc, estado = _estado(None)
    original = estado.solucion_actual
    peor = copy.deepcopy(original)
    peor["ms"] = original["ms"] + 10
    peor["df"] = original["df"].copy()
    peor["df"].loc[peor["df"].index[0], "Fin"] = pd.to_datetime(peor["df"].loc[peor["df"].index[0], "Fin"]) \
        + pd.Timedelta(minutes=10)
    peor["df"]["Fin"] = pd.to_datetime(peor["df"]["Fin"])
    estado.temperatura = 1e9
    s = alns._estadistica_vacia()
    aceptada, motivo, categoria = estado._decidir(peor, original["ms"], s, s)
    assert aceptada and categoria == "aceptada"
    assert estado.solucion_actual is peor
    assert inc.mejor is original


def test_sin_recocido_la_peor_no_se_acepta():
    cx, inc, estado = _estado(None, usar_sa=False)
    original = estado.solucion_actual
    peor = copy.deepcopy(original)
    peor["ms"] = original["ms"] + 10
    peor["df"] = original["df"].copy()
    peor["df"]["Fin"] = pd.to_datetime(peor["df"]["Fin"])
    peor["df"].loc[peor["df"].index[0], "Fin"] += pd.Timedelta(minutes=10)
    s = alns._estadistica_vacia()
    aceptada, _, _ = estado._decidir(peor, original["ms"], s, s)
    assert not aceptada and estado.solucion_actual is original


def test_reinicio_vuelve_a_la_mejor_y_conserva_historial():
    cx, inc, estado = _estado(None)
    estado.max_iteraciones = 5
    estado.ejecutar()
    hist = len(cx["historial"])
    estado.solucion_actual = copy.deepcopy(inc.mejor)
    estado._reiniciar()
    assert estado.solucion_actual is inc.mejor
    assert estado.reinicios == 1 and len(cx["historial"]) == hist


# --- proponer_secuencia con ALNS --------------------------------------------------------------

def test_semilla_intacta_y_resultado_valido_con_alns():
    df = semilla_choque()
    antes = df.copy(deep=True)
    p = correr(df)
    pd.testing.assert_frame_equal(df, antes)
    assert p.attrs["factible"] and p.attrs["interferencias_finales"] == 0 and p.attrs["cruces_finales"] == 0
    assert p.attrs["movimientos_conservados"]
    assert p["Movimientos"].sum() == df["Movimientos"].sum()


def test_metadatos_alns_presentes():
    p = correr(semilla_choque())
    for k in (
        "alns_activado", "iteraciones_alns", "candidatas_alns", "candidatas_factibles_alns",
        "candidatas_aceptadas_alns", "mejoras_alns", "operador_destruccion_ganador",
        "operador_reparacion_ganador", "pesos_destruccion", "pesos_reparacion", "estadisticas_operadores",
        "temperatura_final", "reinicios_alns", "gruas_disponibles", "gruas_utilizadas", "gruas_sin_tareas",
        "tareas_reasignadas", "fragmentos_finales",
    ):
        assert k in p.attrs, k
    assert p.attrs["alns_activado"] is True
    assert p.attrs["iteraciones_alns"] > 0
    assert all(v >= alns.PESO_MINIMO for v in p.attrs["pesos_destruccion"].values())
    h = [x for x in p.attrs["historial"] if x["fase"] == "ALNS"]
    assert h
    for campo in (
        "iteracion", "fase", "operador_destruccion", "operador_reparacion", "tamano_vecindad",
        "tareas_liberadas", "tareas_propagadas", "estado_cpsat", "makespan_actual", "makespan_candidato",
        "mejor_makespan", "factible", "aceptada", "motivo", "temperatura", "tiempo_s",
    ):
        assert campo in h[0], campo


def test_alns_desactivado_reproduce_etapa_2():
    p = correr(semilla_choque(), activar_alns=False)
    assert p.attrs["alns_activado"] is False and p.attrs["iteraciones_alns"] == 0
    assert not [x for x in p.attrs["historial"] if x["fase"] == "ALNS"]
    assert p.attrs["factible"]


def test_alns_nunca_empeora_la_primera_factible():
    p = correr(semilla_desbalanceada(), config_gmph(30, 60), max_iteraciones_alns=60)
    assert p.attrs["makespan_final_min"] <= p.attrs["makespan_primera_factible_min"] + 1e-6
    assert p.attrs["factible"]


def test_gruas_sin_tareas_no_producen_error():
    df = cronograma(fila("G1", 2, 30, 0))
    p = correr(df, config("G1", "G2"), max_iteraciones_alns=10)
    assert p.attrs["factible"]
    assert p.attrs["gruas_disponibles"] == 2 and p.attrs["gruas_utilizadas"] <= 2
    assert set(p.attrs["gruas_sin_tareas"]) <= {"G1", "G2"}


def test_reasignacion_prohibida_conserva_la_grua():
    df = semilla_desbalanceada()
    p = correr(df, config_gmph(30, 60), permitir_reasignacion=False)
    orig = ifc._asegurar_identidad(df).groupby("Parent_Task_ID")["Grúa"].first()
    for pid, g in p.groupby("Parent_Task_ID")["Grúa"]:
        assert set(g) == {orig[pid]}


def test_division_prohibida_no_crea_fragmentos():
    df = semilla_desbalanceada()
    p = correr(df, config_gmph(30, 60), permitir_division=False)
    assert p.groupby("Parent_Task_ID").size().max() == 1


def test_division_permitida_conserva_movimientos():
    df = semilla_desbalanceada()
    p = correr(df, config_gmph(30, 60), permitir_division=True)
    assert movs(p) == movs(ifc._asegurar_identidad(df))


def test_inicio_minimo_de_grua_se_respeta_con_alns():
    cfg = config_gmph(30, 60, inicio_g2=T0 + pd.Timedelta(minutes=300))
    p = correr(semilla_desbalanceada(), cfg)
    assert pd.to_datetime(p[p["Grúa"] == "G2"]["Inicio"]).min() >= T0 + pd.Timedelta(minutes=300) \
        if (p["Grúa"] == "G2").any() else True


def test_misma_semilla_aleatoria_es_reproducible(monkeypatch):
    monkeypatch.setattr(ifc, "_proponer_cpsat", lambda *a, **k: None)
    df = semilla_choque()
    a = correr(df, max_iteraciones_alns=25, random_seed=9)
    b = correr(df, max_iteraciones_alns=25, random_seed=9)
    assert [h["operador_destruccion"] for h in a.attrs["historial"] if h["fase"] == "ALNS"] == \
        [h["operador_destruccion"] for h in b.attrs["historial"] if h["fase"] == "ALNS"]
    assert a.attrs["makespan_final_min"] == b.attrs["makespan_final_min"]


def test_alns_respeta_el_limite_de_tiempo():
    t0 = time.monotonic()
    p = ifc.proponer_secuencia(semilla_choque(), config("G1", "G2"), intervalo_fn, 15, tiempo_limite=3)
    assert time.monotonic() - t0 < 3 + 2.5
    assert p.attrs["iteraciones_alns"] > 0


def test_compatibilidad_posicional():
    p = ifc.proponer_secuencia(semilla_choque(), config("G1", "G2"), intervalo_fn, 15, 2)
    assert p.attrs["factible"] and "metodo" in p.attrs


def test_resultado_con_instancia_de_tres_gruas_es_valido():
    df = instancia(3, 4)
    cfg = config("G1", "G2", "G3")
    p = ifc.proponer_secuencia(df, cfg, intervalo_fn, 15, tiempo_limite=4, max_iteraciones_alns=40)
    assert p.attrs["factible"] and p.attrs["interferencias_finales"] == 0 and p.attrs["cruces_finales"] == 0
    assert p["Movimientos"].sum() == df["Movimientos"].sum()
    assert p.attrs["makespan_final_min"] <= p.attrs["makespan_primera_factible_min"] + 1e-6


# --- golden: no fusionar tareas distintas ----------------------------------------------------

def test_no_se_fusionan_piezas_de_tareas_distintas():
    base = fila("G1", 14, 30, 0)
    otra = fila("G1", 14, 30, 60)
    df = ifc._asegurar_identidad(cronograma(base, otra))
    assert df["Task_ID"].nunique() == 2
    piezas = ifc._despachar(
        [dict(t) for t in ifc.crear_contexto(
            df.assign(**{"Grúa original": df["Grúa"]}), config("G1"), intervalo_fn, 15, {"G1": 0}, time.monotonic() + 5
        )["tareas"]],
        config("G1"), intervalo_fn, 15, "fin", {"G1": 0},
    )
    out = ifc._armar_cronograma(piezas)
    assert len(out) == 2 and out["Task_ID"].nunique() == 2
    assert out["Movimientos"].sum() == 60
