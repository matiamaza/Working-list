import pandas as pd
import pytest

import interferencias as ifc
from tests.test_interferencias import config, cronograma, fila, intervalo_fn


def semilla_conflicto():
    return cronograma(
        fila("G1", 10, 50, 0), fila("G1", 14, 40, 100), fila("G1", 14, 20, 180, tipo="Embarque"),
        fila("G2", 14, 60, 0), fila("G2", 18, 40, 120), fila("G2", 22, 30, 200, tipo="Embarque"),
    )


def semilla_factible():
    return cronograma(fila("G1", 2, 60, 0), fila("G1", 6, 30, 120), fila("G2", 30, 60, 0), fila("G2", 34, 30, 120))


def proponer(df, **kw):
    kw.setdefault("tiempo_limite", 3)
    kw.setdefault("max_iteraciones_alns", 10)
    kw.setdefault("usar_cpsat_local", False)
    return ifc.proponer_secuencia(df, config("G1", "G2"), intervalo_fn, 15, **kw)


def test_metricas_con_semilla_infactible():
    p = proponer(semilla_conflicto())
    a = p.attrs
    assert a["semilla_factible"] is False
    assert a["interferencias_semilla"] > 0
    assert a["factible"] and a["estado_solucion"] == "Solución reparada"
    primera, final, semilla = a["makespan_primera_factible_min"], a["makespan_final_min"], a["makespan_semilla_min"]
    assert a["variacion_para_factibilidad_min"] == pytest.approx(primera - semilla)
    assert a["mejora_desde_primera_factible_min"] == pytest.approx(primera - final)
    assert a["mejora_desde_primera_factible_min"] >= -ifc.TOLERANCIA_MAKESPAN_MIN


def test_semilla_factible_no_reporta_variacion_de_factibilidad():
    a = proponer(semilla_factible()).attrs
    assert a["semilla_factible"] is True
    assert a["variacion_para_factibilidad_min"] == 0.0
    assert a["estado_solucion"] == "Solución factible mantenida"


def test_los_cuatro_estados_finales():
    base = {"factible": True, "makespan_semilla_min": 200.0, "makespan_final_min": 150.0, "semilla_factible": True}
    assert ifc.estado_solucion(base) == "Solución factible mejorada"
    assert ifc.estado_solucion({**base, "makespan_final_min": 200.5}) == "Solución factible mantenida"
    assert ifc.estado_solucion({**base, "semilla_factible": False, "makespan_final_min": 300.0}) == "Solución reparada"
    assert ifc.estado_solucion({**base, "factible": False}) == "Solución parcialmente reparada"


def test_compatibilidad_con_attrs_antiguos():
    assert ifc.estado_solucion({}) == "Solución parcialmente reparada"
    antiguo = {"factible": True, "makespan_semilla_min": 100.0, "makespan_final_min": 90.0, "interferencias_semilla": 2}
    assert ifc.estado_solucion(antiguo) == "Solución reparada"
    assert ifc.estado_solucion({"factible": True, "makespan_semilla_min": 100.0, "makespan_final_min": 80.0}) == (
        "Solución factible mejorada"
    )
    assert ifc.estado_solucion(None) == "Solución parcialmente reparada"


def test_modo_reproducible_fuerza_un_worker(monkeypatch):
    vistos = []
    original = ifc._proponer_cpsat

    def espiar(*args, **kw):
        vistos.append(kw.get("workers"))
        return original(*args, **kw)

    monkeypatch.setattr(ifc, "_proponer_cpsat", espiar)
    p = proponer(semilla_conflicto(), tiempo_limite=4, workers=4, modo_reproducible=True)
    assert p.attrs["workers"] == 1 and p.attrs["modo_reproducible"] is True
    assert vistos and set(vistos) == {1}
    q = proponer(semilla_conflicto(), tiempo_limite=4, workers=4)
    assert q.attrs["workers"] == 4 and q.attrs["modo_reproducible"] is False


def test_attrs_de_trazabilidad():
    a = proponer(semilla_conflicto(), tiempo_limite=4, random_seed=7).attrs
    for clave in ("random_seed", "workers", "tiempo_limite", "modo_reproducible", "version_ortools",
                  "fecha_ejecucion", "firma_entrada", "cpsat_global_estado", "cpsat_global_mejor_cota"):
        assert clave in a
    assert a["random_seed"] == 7 and a["tiempo_limite"] == 4
    assert a["firma_entrada"] == ifc.firma_entrada(semilla_conflicto(), config("G1", "G2"))


def test_firma_de_entrada_estable():
    df, cfg = semilla_conflicto(), config("G1", "G2")
    firma = ifc.firma_entrada(df, cfg)
    assert firma == ifc.firma_entrada(df.copy(), dict(cfg))
    barajado = df.sample(frac=1, random_state=3).reset_index(drop=True)
    assert firma == ifc.firma_entrada(barajado, cfg)
    otro = df.copy()
    otro.loc[0, "Movimientos"] += 1
    assert firma != ifc.firma_entrada(otro, cfg)
    assert firma != ifc.firma_entrada(df, config("G1", "G2", gmph=25))


def test_orden_fisico_completo_y_sin_duplicados():
    df = semilla_conflicto()
    assert ifc.resolver_orden_fisico(df, ["G2", "G1"]) == {"G2": 0, "G1": 1}
    assert ifc.resolver_orden_fisico(df, {"G1": 1, "G2": 0}) == {"G2": 0, "G1": 1}
    with pytest.raises(ValueError):
        ifc.resolver_orden_fisico(df, ["G1"])
    with pytest.raises(ValueError):
        ifc.resolver_orden_fisico(df, ["G1", "G1", "G2"])


def test_orden_fisico_manual_no_fija_la_primera_tarea():
    df = semilla_conflicto()
    p = proponer(df, orden_fisico_gruas=["G1", "G2"])
    assert p.attrs["factible"] and p.attrs["cruces_finales"] == 0


def test_reporte_de_cp_local_sin_ejecuciones():
    assert ifc.resumen_cp_local({}) == "Sin ejecuciones"
    assert ifc.resumen_cp_local(None) == "Sin ejecuciones"
    assert ifc.resumen_cp_local({"ejecuciones": 0}) == "Sin ejecuciones"
    stats = {"ejecuciones": 2}
    assert ifc.resumen_cp_local(stats) is stats
    p = proponer(semilla_conflicto(), activar_alns=False)
    assert ifc.resumen_cp_local(p.attrs["estadisticas_cp_local"]) == "Sin ejecuciones"
    q = proponer(semilla_conflicto(), usar_cpsat_local=False)
    cp = q.attrs["estadisticas_cp_local"]
    assert cp["ejecuciones"] == 0 and ifc.resumen_cp_local(cp) == "Sin ejecuciones"
    assert set(cp["estados"]) == {"OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN"}
    assert cp["diferencia_maxima_min"] is None and cp["tiempo_promedio_seg"] is None
