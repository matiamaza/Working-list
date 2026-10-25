import copy
import time

import pandas as pd
import pytest

import alns
import interferencias as ifc
from tests.test_interferencias import config, cronograma, fila, intervalo_fn, pontón


def instancia(n_gruas=3, tareas_por_grua=5, con_ponton=True):
    """Instancia pequeña y determinista: cada grúa con su tramo de bays y algo de trabajo cruzado."""
    filas = []
    for k in range(n_gruas):
        t = 0
        for j in range(tareas_por_grua):
            bay = 2 + 12 * k + 4 * (j % 3)
            filas.append(fila(f"G{k + 1}", bay, 30 + 10 * ((j + k) % 4), t, tipo="Descarga" if j % 2 == 0 else "Embarque"))
            t += 130 + 10 * j
    if con_ponton:
        filas.append(pontón("G1", 6, 20))
    return cronograma(*filas)


def contexto(semilla=None, reasignar=True, dividir=True, limite_s=30, gmph=30):
    semilla = ifc._asegurar_identidad(instancia() if semilla is None else semilla)
    semilla["Grúa original"] = semilla["Grúa"]
    gruas = sorted(semilla["Grúa"].unique())
    cfg = config(*gruas, gmph=gmph)
    rango = ifc.resolver_orden_fisico(semilla, gruas)
    cx = ifc.crear_contexto(
        semilla, cfg, intervalo_fn, 15, rango, time.monotonic() + limite_s, reasignar, dividir
    )
    return cx


def plan_factible(cx, criterio="fin"):
    piezas = ifc._despachar(
        cx["tareas"], cx["config"], cx["intervalo_fn"], cx["holgura"], criterio, cx["rango"],
        aristas=cx["aristas"], permitir_reasignacion=cx["reasignar"], permitir_division=cx["dividir"],
    )
    return ifc._armar_cronograma(piezas)


class EstadoFalso:
    def __init__(self, conflictos=None):
        self.conflictos = conflictos or {}


@pytest.fixture
def cx():
    return contexto()


@pytest.fixture
def df(cx):
    return plan_factible(cx)


@pytest.mark.parametrize("nombre", sorted(alns.DESTRUCTORES))
def test_destructor_devuelve_tareas_validas_sin_pontones(nombre, cx, df):
    pontones = {t["id"] for t in cx["tareas"] if t["tipo"] == "Pontón"}
    assert pontones
    import random
    rnd = random.Random(7)
    for n in (2, 3, 5):
        ids = alns.DESTRUCTORES[nombre](df, cx, rnd, n, EstadoFalso())
        assert 1 <= len(ids) <= n
        assert len(set(ids)) == len(ids)
        assert set(ids) <= set(df["Parent_Task_ID"])
        assert not set(ids) & pontones


@pytest.mark.parametrize("nombre", sorted(alns.DESTRUCTORES))
def test_destructor_no_muta_la_solucion_ni_el_contexto(nombre, cx, df):
    import random
    antes_df = df.copy(deep=True)
    antes_semilla = cx["semilla"].copy(deep=True)
    antes_tareas = copy.deepcopy(cx["tareas"])
    alns.DESTRUCTORES[nombre](df, cx, random.Random(3), 4, EstadoFalso())
    pd.testing.assert_frame_equal(df, antes_df)
    pd.testing.assert_frame_equal(cx["semilla"], antes_semilla)
    assert cx["tareas"] == antes_tareas


@pytest.mark.parametrize("nombre", sorted(alns.DESTRUCTORES))
def test_destructor_es_reproducible_con_la_misma_semilla(nombre, cx, df):
    import random
    a = alns.DESTRUCTORES[nombre](df, cx, random.Random(11), 4, EstadoFalso())
    b = alns.DESTRUCTORES[nombre](df, cx, random.Random(11), 4, EstadoFalso())
    assert a == b


def test_critical_crane_removal_toma_el_tramo_final_de_la_grua_critica(cx, df):
    import random
    critica = alns._grua_critica(df)
    ids = alns.critical_crane_removal(df, cx, random.Random(1), 2, EstadoFalso())
    asignadas = {g for i in ids for g in set(df.loc[df["Parent_Task_ID"] == i, "Grúa"])}
    assert critica in asignadas


def test_spatial_block_removal_libera_bays_cercanos(cx, df):
    import random
    ids = alns.spatial_block_removal(df, cx, random.Random(2), 4, EstadoFalso())
    bays = sorted(cx["tareas_por_id"][i]["bay"] for i in ids)
    assert bays[-1] - bays[0] <= 3 * ifc.SEPARACION_MIN


def test_conflict_history_usa_el_historial_si_existe_y_si_no_hace_fallback(cx, df):
    import random
    objetivo = next(t["id"] for t in cx["tareas"] if t["tipo"] != "Pontón")
    con = alns.conflict_history_removal(df, cx, random.Random(5), 3, EstadoFalso({objetivo: 50}))
    assert objetivo in con
    sin = alns.conflict_history_removal(df, cx, random.Random(5), 3, EstadoFalso())
    assert 1 <= len(sin) <= 3


def test_destructor_con_una_sola_tarea_elegible_devuelve_esa_tarea():
    cx = contexto(cronograma(fila("G1", 2, 40, 0), pontón("G1", 6, 100), fila("G2", 30, 0, 0)).iloc[:2].copy())
    df = plan_factible(cx)
    import random
    for nombre, f in alns.DESTRUCTORES.items():
        ids = f(df, cx, random.Random(1), 3, EstadoFalso())
        assert len(ids) == 1, nombre


def test_tamano_de_vecindad_respeta_minimo_y_topes():
    assert alns.tamano_vecindad(40, 0.10) == 4
    assert alns.tamano_vecindad(10, 0.10) == 2
    assert alns.tamano_vecindad(40, 0.90) == 8  # tope normal del 20 %
    assert alns.tamano_vecindad(40, 0.90, alns.MAXIMO_DIVERSIFICACION) == 12
    assert alns.tamano_vecindad(1, 0.10) == 1
    assert alns.tamano_vecindad(0, 0.10) == 0


def test_cierre_de_sucesoras_propaga_y_cuenta_lo_agregado(cx):
    primero = min((t for t in cx["tareas"] if t["tipo"] != "Pontón"), key=lambda t: t["orden"])
    total, propagadas = alns.cierre_sucesoras([primero["id"]], cx)
    assert primero["id"] in total and not propagadas & {primero["id"]}
    assert total == {primero["id"]} | propagadas
    for a, b in cx["aristas"]:
        if a in total:
            assert b in total
