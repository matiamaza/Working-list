"""Pruebas del motor de interferencias y de la propuesta de secuencia (sin Streamlit)."""
import pathlib
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import interferencias as ifc  # noqa: E402

T0 = datetime(2026, 1, 5, 8, 0)


def hora(minutos):
    return T0 + timedelta(minutes=minutos)


def intervalo_fn(params, inicio, movs, dur_fija=None):
    inicio = pd.Timestamp(inicio).to_pydatetime()
    dur = dur_fija if dur_fija is not None else movs / params["gmph"] * 60
    return inicio, inicio + timedelta(minutes=dur), round(dur, 1)


def intervalo_con_error(params, inicio, movs, dur_fija=None):
    raise ValueError("La tarea supera los turnos GMPH configurados para esta grúa.")


def fila(grua, bay, movs, ini_min, tipo="Descarga", gmph=30, zona="Cubierta", sufijo="D", accion=None):
    dur = movs / gmph * 60
    accion = accion or ("DISCH" if tipo == "Descarga" else "LOAD")
    return {
        "Grúa": grua, "Bay Real": f"{bay:02d}{sufijo}", "Bay Visual": f"{bay:02d}",
        "Acción": f"{accion} ({movs})", "Movimientos": movs,
        "Inicio": hora(ini_min), "Fin": hora(ini_min + dur), "Duración Neta": round(dur, 1),
        "Categoría Visual": f"{grua} - {tipo}", "Zona": zona,
    }


def pontón(grua, bay, ini_min):
    return {
        "Grúa": grua, "Bay Real": f"{bay:02d}", "Bay Visual": f"{bay:02d}",
        "Acción": "PONTOON OPEN (0)", "Movimientos": 0,
        "Inicio": hora(ini_min), "Fin": hora(ini_min + 15), "Duración Neta": 15.0,
        "Categoría Visual": "Pontón", "Zona": "Sin identificar",
    }


def cronograma(*filas):
    return pd.DataFrame(list(filas))


def config(*gruas, gmph=30):
    return {g: {"inicio": T0, "gmph": gmph, "gmph_schedule": None} for g in gruas}


@pytest.fixture
def semilla_conflicto():
    """G1 y G2 coinciden en el bay 14 y comparten bays contiguos."""
    return cronograma(
        fila("G1", 10, 50, 0),
        fila("G1", 14, 40, 100),
        fila("G1", 14, 20, 180, tipo="Embarque"),
        pontón("G1", 18, 250),
        fila("G2", 14, 60, 0),
        fila("G2", 18, 40, 120),
        fila("G2", 22, 30, 200, tipo="Embarque"),
    )


@pytest.fixture
def semilla_factible():
    return cronograma(
        fila("G1", 2, 60, 0),
        fila("G1", 6, 30, 120),
        fila("G2", 30, 60, 0),
        fila("G2", 34, 30, 120),
    )


# --- 1 a 8: detección de interferencias -------------------------------------------------

def test_tareas_sin_solape_no_interfieren():
    df = cronograma(fila("G1", 14, 30, 0), fila("G2", 14, 30, 200))
    assert ifc.detectar_interferencias(df, 15).empty


def test_tareas_que_se_tocan_con_holgura_cero():
    df = cronograma(fila("G1", 14, 30, 0), fila("G2", 14, 30, 60))
    assert ifc.detectar_interferencias(df, 0).empty
    res = ifc.detectar_interferencias(df, 15)
    assert len(res) == 1 and res.iloc[0]["Tipo"] == "Holgura insuficiente"


def test_simultaneas_en_el_mismo_bay():
    df = cronograma(fila("G1", 14, 60, 0), fila("G2", 14, 60, 30))
    res = ifc.detectar_interferencias(df, 15)
    assert len(res) == 1
    assert res.iloc[0]["Tipo"] == "Simultáneas" and res.iloc[0]["Ubicación"] == "Mismo bay"


def test_simultaneas_en_bays_contiguos():
    df = cronograma(fila("G1", 14, 60, 0), fila("G2", 18, 60, 30))
    res = ifc.detectar_interferencias(df, 15)
    assert len(res) == 1 and res.iloc[0]["Ubicación"] == "Bay contiguo"


def test_separadas_por_dos_posiciones_no_interfieren():
    df = cronograma(fila("G1", 14, 60, 0), fila("G2", 22, 60, 0))
    assert ifc.detectar_interferencias(df, 15).empty


@pytest.mark.parametrize("diferencia,esperado", [(0, 1), (4, 1), (8, 0), (12, 0)])
def test_diferencia_de_bays_visuales(diferencia, esperado):
    df = cronograma(fila("G1", 14, 60, 0), fila("G2", 14 + diferencia, 60, 0))
    assert len(ifc.detectar_interferencias(df, 15)) == esperado


def test_constantes_de_distancia():
    assert ifc.DISTANCIA_MAX_BAYS == 4
    assert ifc.SEPARACION_MIN == 8


def test_cubierta_y_bodega_del_mismo_bay_son_el_mismo_bay():
    df = cronograma(
        fila("G1", 22, 60, 0, sufijo="D", zona="Cubierta"),
        fila("G2", 22, 60, 0, sufijo="H", zona="Bodega"),
    )
    res = ifc.detectar_interferencias(df, 15)
    assert len(res) == 1 and res.iloc[0]["Ubicación"] == "Mismo bay"


# --- 9 y 10: cruces y orden físico ------------------------------------------------------

def test_orden_fisico_de_gruas_por_primer_bay():
    df = cronograma(fila("G2", 22, 30, 0), fila("G1", 6, 30, 0), fila("G3", 40, 30, 0))
    assert ifc.orden_original_gruas(df) == {"G1": 0, "G2": 1, "G3": 2}


def test_cruce_detectado_cuando_la_grua_de_proa_pasa_a_la_de_popa():
    df = cronograma(
        fila("G1", 10, 30, 0), fila("G2", 30, 30, 0),
        fila("G1", 38, 60, 60), fila("G2", 34, 60, 60),
    )
    assert ifc.detectar_cruces(df) == [("G1", "G2")]


def test_sin_cruce_con_separacion_de_8():
    df = cronograma(fila("G1", 10, 60, 0), fila("G2", 18, 60, 0))
    assert ifc.detectar_cruces(df) == []


def test_cruce_con_separacion_menor_a_8():
    df = cronograma(fila("G1", 10, 60, 0), fila("G2", 14, 60, 0))
    assert ifc.detectar_cruces(df) == [("G1", "G2")]


# --- 11 a 14: propiedades de la propuesta -----------------------------------------------

def _secuencia_por_bay(df):
    resultado = {}
    for bay, grupo in df.sort_values("Inicio").groupby("Bay Visual"):
        acciones = [str(a).split(" (")[0] for a in grupo["Acción"]]
        resultado[bay] = [a for i, a in enumerate(acciones) if i == 0 or a != acciones[i - 1]]
    return resultado


def _movimientos_por_clave(df):
    claves = df["Bay Real"].astype(str) + "|" + df["Acción"].astype(str).str.replace(r"\s*\(\d+\)$", "", regex=True)
    return df.groupby(claves)["Movimientos"].sum().to_dict()


def test_propuesta_sin_interferencias_ni_cruces(semilla_conflicto):
    assert not ifc.detectar_interferencias(semilla_conflicto, 15).empty
    propuesta = ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_fn, 15)
    assert ifc.detectar_interferencias(propuesta, 15).empty
    assert ifc.detectar_cruces(propuesta) == []


def test_precedencias_heredadas_se_conservan(semilla_conflicto):
    propuesta = ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_fn, 15)
    esperado = _secuencia_por_bay(semilla_conflicto)
    obtenido = _secuencia_por_bay(propuesta)
    for bay, secuencia in esperado.items():
        assert obtenido[bay] == secuencia, f"El orden del bay {bay} cambió"


def test_ponton_permanece_en_su_grua(semilla_conflicto):
    propuesta = ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_fn, 15)
    pontones = propuesta[propuesta["Categoría Visual"] == "Pontón"]
    assert list(pontones["Grúa"]) == ["G1"] and list(pontones["Movimientos"]) == [0]


def test_movimientos_conservados(semilla_conflicto):
    propuesta = ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_fn, 15)
    assert _movimientos_por_clave(propuesta) == _movimientos_por_clave(semilla_conflicto)
    assert (propuesta["Movimientos"] >= 0).all()


def test_makespan_real(semilla_factible):
    # 08:00 → la última tarea (30 movs a 30 GMPH, desde el minuto 120) termina en el minuto 180
    assert ifc.calcular_makespan_min(semilla_factible) == pytest.approx(180.0)


# --- 15 y 16: errores y fallback --------------------------------------------------------

def test_value_error_del_callback_no_rompe_la_propuesta(semilla_conflicto):
    propuesta = ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_con_error, 15)
    assert propuesta.attrs["factible"] is False


def test_fallo_de_cpsat_no_elimina_candidatas_heuristicas(semilla_conflicto, monkeypatch):
    def falla(*args, **kwargs):
        raise RuntimeError("CP-SAT no disponible")

    monkeypatch.setattr(ifc, "_proponer_cpsat", falla)
    propuesta = ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_fn, 15)
    assert ifc.detectar_interferencias(propuesta, 15).empty
    assert propuesta.attrs["factible"] is True


def test_cpsat_omitido_o_sin_solucion(semilla_conflicto, monkeypatch):
    monkeypatch.setattr(ifc, "_proponer_cpsat", lambda *a, **k: None)
    propuesta = ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_fn, 15)
    assert propuesta.attrs["factible"] is True
    assert "CP-SAT" not in propuesta.attrs["metodo"]


# --- inmutabilidad, identidad, metadatos y tiempo ---------------------------------------

def test_la_semilla_no_se_modifica(semilla_conflicto):
    copia = semilla_conflicto.copy(deep=True)
    ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_fn, 15)
    pd.testing.assert_frame_equal(semilla_conflicto, copia)
    assert semilla_conflicto.attrs == copia.attrs


def test_identidad_estable_de_tareas(semilla_conflicto):
    propuesta = ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_fn, 15)
    for columna in ("Task_ID", "Parent_Task_ID", "Fragment_ID"):
        assert columna in propuesta.columns
    assert propuesta["Fragment_ID"].is_unique
    assert set(propuesta["Parent_Task_ID"]) == set(ifc._asegurar_identidad(semilla_conflicto)["Task_ID"])


def test_metadatos_de_la_propuesta(semilla_conflicto):
    propuesta = ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_fn, 15, random_seed=7)
    for clave in (
        "metodo", "factible", "makespan_semilla_min", "makespan_primera_factible_min", "makespan_final_min",
        "mejora_min", "mejora_porcentual", "interferencias_semilla", "interferencias_finales",
        "cruces_finales", "movimientos_conservados", "tiempo_total_seg", "iteraciones_reparacion",
        "iteraciones_mejora", "random_seed", "historial",
    ):
        assert clave in propuesta.attrs, clave
    assert propuesta.attrs["random_seed"] == 7
    assert propuesta.attrs["movimientos_conservados"] is True
    assert propuesta.attrs["interferencias_finales"] == 0


def test_el_makespan_final_no_supera_al_de_la_primera_factible(semilla_conflicto):
    propuesta = ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_fn, 15)
    assert propuesta.attrs["makespan_final_min"] == pytest.approx(ifc.calcular_makespan_min(propuesta))
    assert propuesta.attrs["makespan_final_min"] <= propuesta.attrs["makespan_primera_factible_min"] + 1e-6


def test_semilla_ya_factible_se_conserva(semilla_factible):
    propuesta = ifc.proponer_secuencia(semilla_factible, config("G1", "G2"), intervalo_fn, 15)
    assert propuesta.attrs["factible"] is True
    assert propuesta.attrs["makespan_final_min"] <= ifc.calcular_makespan_min(semilla_factible) + 1e-6


def test_respeta_el_limite_de_tiempo(semilla_conflicto):
    inicio = time.monotonic()
    ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_fn, 15, tiempo_limite=3)
    assert time.monotonic() - inicio < 6


# --- reparación iterativa y validación --------------------------------------------------

def test_reparacion_iterativa_sola_logra_factibilidad(semilla_conflicto, monkeypatch):
    def falla(*a, **k):
        raise ValueError("sin heurísticas")
    monkeypatch.setattr(ifc, "_despachar", falla)
    monkeypatch.setattr(ifc, "_proponer_cpsat", falla)
    # Sin _despachar la propuesta cae a la semilla no factible, sin lanzar errores.
    propuesta = ifc.proponer_secuencia(semilla_conflicto, config("G1", "G2"), intervalo_fn, 15, tiempo_limite=5)
    assert propuesta.attrs["factible"] is False
    assert propuesta.attrs["fallos"]


def test_reparacion_directa_elimina_conflictos(semilla_conflicto):
    sem = ifc._asegurar_identidad(semilla_conflicto)
    sem["Grúa original"] = sem["Grúa"]
    tareas = ifc._tareas_desde_cronograma(sem)
    aristas = ifc.construir_precedencias(tareas)
    cx = {
        "semilla": sem, "tareas": tareas, "tareas_por_id": {t["id"]: t for t in tareas}, "aristas": aristas,
        "preds": ifc._predecesoras(aristas), "rango": ifc.orden_original_gruas(sem), "config": config("G1", "G2"),
        "intervalo_fn": intervalo_fn, "holgura": 15, "gruas": ["G1", "G2"],
        "t_ref": pd.to_datetime(sem["Inicio"]).min(),
        "ini_semilla": pd.to_datetime(sem["Inicio"]).groupby(sem["Parent_Task_ID"]).min(),
        "limite": time.monotonic() + 10, "debug": True, "t0": time.monotonic(), "iteracion": 0,
        "historial": [], "fallos": [], "iter_reparacion": 0, "iter_mejora": 0,
        "reasignar": True, "dividir": True, "inicios": {"G1": T0, "G2": T0},
    }
    inc = ifc._Incumbente()
    ev = ifc._evaluar_plan(sem, cx, "Semilla")
    assert not ev["factible"]
    final = ifc._reparar_iterativo(cx, inc, ev)
    assert final["factible"] and final["val"]["movimientos_conservados"]
    assert inc.mejor["factible"]


def test_validar_solucion_detecta_movimientos_perdidos(semilla_factible):
    sol = semilla_factible.copy()
    sol.loc[0, "Movimientos"] -= 1
    assert not ifc.validar_solucion(sol, semilla_factible, 15)["factible"]
    assert ifc.validar_solucion(semilla_factible, semilla_factible, 15)["factible"]

# --- libertad del optimizador: semilla solo como referencia ----------------------------

def config_gmph(g1, g2, inicio_g2=T0):
    return {
        "G1": {"inicio": T0, "gmph": g1, "gmph_schedule": None},
        "G2": {"inicio": inicio_g2, "gmph": g2, "gmph_schedule": None},
    }


@pytest.fixture
def semilla_choque():
    """G1 y G2 empiezan a la vez en el bay 14; G2 tiene luego trabajo en el bay 30."""
    return cronograma(fila("G1", 14, 60, 0), fila("G2", 14, 60, 0), fila("G2", 30, 40, 200))


@pytest.fixture
def semilla_desbalanceada():
    """G1 concentra casi todo el trabajo; G2 (el doble de rápida) casi no tiene."""
    return cronograma(fila("G1", 2, 200, 0), fila("G1", 10, 200, 400), fila("G2", 30, 20, 0))


def _primera(df, grua):
    f = df[df["Grúa"] == grua].sort_values("Inicio").iloc[0]
    return f["Bay Visual"], f["Inicio"]


def test_la_semilla_original_queda_intacta(semilla_choque):
    copia = semilla_choque.copy(deep=True)
    ifc.proponer_secuencia(semilla_choque, config("G1", "G2"), intervalo_fn, 15, tiempo_limite=3)
    pd.testing.assert_frame_equal(semilla_choque, copia)


def test_la_solucion_final_puede_tener_horarios_y_primeras_tareas_distintos(semilla_choque):
    p = ifc.proponer_secuencia(semilla_choque, config("G1", "G2"), intervalo_fn, 15, tiempo_limite=4)
    assert p.attrs["factible"]
    assert p.attrs["makespan_final_min"] < p.attrs["makespan_semilla_min"]
    # G2 pasa a comenzar en otro bay y con otra primera tarea que en la semilla.
    assert _primera(p, "G2")[0] != _primera(semilla_choque, "G2")[0]
    assert not p.sort_values(["Grúa", "Inicio"])["Inicio"].reset_index(drop=True).equals(
        semilla_choque.sort_values(["Grúa", "Inicio"])["Inicio"].reset_index(drop=True)
    )


def test_con_reasignacion_una_tarea_cambia_de_grua(semilla_choque):
    p = ifc.proponer_secuencia(
        semilla_choque, config("G1", "G2"), intervalo_fn, 15, tiempo_limite=4, permitir_reasignacion=True
    )
    assert p.attrs["factible"] and (p["Grúa"] != p["Grúa original"]).any()


def test_sin_reasignacion_las_tareas_conservan_su_grua(semilla_choque):
    p = ifc.proponer_secuencia(
        semilla_choque, config("G1", "G2"), intervalo_fn, 15, tiempo_limite=4, permitir_reasignacion=False
    )
    assert p.attrs["factible"]
    assert (p["Grúa"] == p["Grúa original"]).all()
    # Los tiempos y la posición en la secuencia sí pueden cambiar.
    assert _primera(p, "G2")[0] == "30"


def test_con_division_una_tarea_se_reparte_entre_gruas(semilla_desbalanceada):
    p = ifc.proponer_secuencia(
        semilla_desbalanceada, config_gmph(30, 60), intervalo_fn, 15, tiempo_limite=8, workers=2,
        permitir_division=True,
    )
    assert p.attrs["factible"] and p.attrs["movimientos_conservados"]
    assert p["Parent_Task_ID"].duplicated().any()
    assert p["Fragment_ID"].is_unique
    assert p.groupby("Parent_Task_ID")["Movimientos"].sum().sum() == 420
    assert p.attrs["makespan_final_min"] < 400


def test_sin_division_ninguna_tarea_se_parte(semilla_desbalanceada):
    p = ifc.proponer_secuencia(
        semilla_desbalanceada, config_gmph(30, 60), intervalo_fn, 15, tiempo_limite=8, workers=2,
        permitir_division=False,
    )
    assert p.attrs["factible"] and not p["Parent_Task_ID"].duplicated().any()
    assert p["Movimientos"].sum() == 420


def test_menor_makespan_gana_aunque_cambie_mucho_la_semilla(semilla_desbalanceada):
    p = ifc.proponer_secuencia(
        semilla_desbalanceada, config_gmph(30, 60), intervalo_fn, 15, tiempo_limite=8, workers=2
    )
    assert p.attrs["factible"]
    assert p.attrs["makespan_final_min"] <= 0.5 * p.attrs["makespan_semilla_min"]
    assert (p["Grúa"] != p["Grúa original"]).any()


def test_el_tiempo_se_recalcula_con_el_rendimiento_de_la_grua_receptora(semilla_desbalanceada):
    p = ifc.proponer_secuencia(
        semilla_desbalanceada, config_gmph(30, 60), intervalo_fn, 15, tiempo_limite=8, workers=2
    )
    movidas = p[p["Grúa"] != p["Grúa original"]]
    assert not movidas.empty
    for _, f in p.iterrows():
        factor = 1 if f["Grúa"] == "G2" else 2  # minutos por movimiento: G2 a 60 GMPH, G1 a 30
        assert f["Duración Neta"] == pytest.approx(f["Movimientos"] * factor, abs=0.2)


def test_el_inicio_configurado_es_un_minimo_no_una_hora_fija(semilla_choque):
    minimo = hora(120)
    p = ifc.proponer_secuencia(
        semilla_choque, config_gmph(30, 30, inicio_g2=minimo), intervalo_fn, 15, tiempo_limite=4
    )
    assert p.attrs["factible"]
    g2 = p[p["Grúa"] == "G2"]
    assert g2.empty or (pd.to_datetime(g2["Inicio"]) >= minimo).all()
    invalida = ifc.validar_solucion(
        semilla_choque, semilla_choque, 15, inicios_minimos={"G1": T0, "G2": minimo}
    )
    assert not invalida["fechas_validas"]


def test_el_orden_fisico_explicito_prevalece_sobre_la_posicion_de_la_semilla():
    # Según la semilla G2 está a proa; el orden físico real dice lo contrario.
    semilla = cronograma(
        fila("G1", 30, 60, 0), fila("G1", 34, 30, 120), fila("G2", 2, 60, 0), fila("G2", 6, 30, 120)
    )
    orden = ["G1", "G2"]
    assert ifc.resolver_orden_fisico(semilla) == {"G2": 0, "G1": 1}
    assert ifc.resolver_orden_fisico(semilla, orden) == {"G1": 0, "G2": 1}
    p = ifc.proponer_secuencia(
        semilla, config("G1", "G2"), intervalo_fn, 15, tiempo_limite=4, orden_fisico_gruas=orden
    )
    assert p.attrs["factible"]
    assert ifc.detectar_cruces(p, {"G1": 0, "G2": 1}) == []


def test_orden_fisico_incompleto_es_un_error_claro(semilla_choque):
    with pytest.raises(ValueError):
        ifc.resolver_orden_fisico(semilla_choque, ["G1"])
