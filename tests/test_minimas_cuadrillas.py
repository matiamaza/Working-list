"""Pruebas de la secuencia de mínimas cuadrillas en el último turno."""
import pathlib
import re
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import interferencias as ifc  # noqa: E402
from tests.test_interferencias import config, cronograma, fila, hora, intervalo_fn  # noqa: E402


def tipo(nombre):
    texto = nombre.upper()
    return "STS" if "STS" in texto else ("MHC" if re.fullmatch(r"G\s*\d+", texto) else "Otra")


def semilla_tres_gruas():
    return cronograma(
        fila("STS 1", 2, 80, 0), fila("STS 1", 6, 40, 160),
        fila("G5", 18, 80, 0), fila("G5", 22, 40, 160),
        fila("STS 2", 34, 80, 0), fila("STS 2", 38, 40, 160),
    )


def test_ventana_ultimo_turno():
    df = pd.DataFrame({"Inicio": [hora(0)], "Fin": [hora(240)]})
    assert ifc.ventana_ultimo_turno(df) == (hora(0), hora(450))
    df = pd.DataFrame({"Inicio": [hora(0)], "Fin": [hora(450)]})
    assert ifc.ventana_ultimo_turno(df) == (hora(0), hora(450))


def test_ahorra_una_cuadrilla_y_prefiere_sts():
    semilla, cfg = semilla_tres_gruas(), config("STS 1", "G5", "STS 2")
    base = ifc.proponer_secuencia(semilla, cfg, intervalo_fn, tiempo_limite=5)
    prop = ifc.proponer_minimas_cuadrillas(semilla, base, cfg, intervalo_fn, 15, tipo, 10)

    assert prop.attrs["cuadrillas_ahorradas"] == 1
    assert prop.attrs["subconjunto"] == ["STS 1", "STS 2"]
    assert set(prop["Grúa"]) == {"STS 1", "STS 2"}
    assert prop["Fin"].max() <= prop.attrs["ventana_turno"][1]
    assert prop["Movimientos"].sum() == semilla["Movimientos"].sum()
    assert ifc.detectar_interferencias(prop, 15).empty


def test_sin_ahorro_devuelve_la_base():
    semilla = cronograma(
        fila("STS 1", 2, 200, 0), fila("G5", 18, 200, 0), fila("STS 2", 34, 200, 0),
    )
    cfg = config("STS 1", "G5", "STS 2")
    base = ifc.proponer_secuencia(semilla, cfg, intervalo_fn, tiempo_limite=5)
    prop = ifc.proponer_minimas_cuadrillas(semilla, base, cfg, intervalo_fn, 15, tipo, 10)

    assert prop.attrs["cuadrillas_ahorradas"] == 0
    assert len(prop.attrs["gruas_turno_final"]) == 3
    assert prop["Movimientos"].sum() == 600


def test_corta_la_cola_en_el_inicio_del_ultimo_turno():
    semilla = cronograma(
        fila('STS 1', 2, 300, 0), fila('STS 1', 6, 30, 600),
        fila('G5', 18, 300, 0), fila('G5', 22, 30, 600),
    )
    cfg = config('STS 1', 'G5')
    base = ifc.proponer_secuencia(semilla, cfg, intervalo_fn, tiempo_limite=5)
    prop = ifc.proponer_minimas_cuadrillas(semilla, base, cfg, intervalo_fn, 15, tipo, 10)
    ini_v = prop.attrs['ventana_turno'][0]
    assert prop.attrs['cuadrillas_ahorradas'] == 1
    assert prop['Movimientos'].sum() == semilla['Movimientos'].sum()
    assert ifc.detectar_interferencias(prop, 15).empty
    sale = (set(semilla['Grúa']) - set(prop.attrs['subconjunto'])).pop()
    assert prop[(prop['Grúa'] == sale) & (pd.to_datetime(prop['Fin']) > ini_v)].empty
    assert prop[prop['Grúa'] == sale]['Movimientos'].sum() == 225




def test_entrega_hasta_tres_alternativas_distintas():
    semilla, cfg = semilla_tres_gruas(), config('STS 1', 'G5', 'STS 2')
    base = ifc.proponer_secuencia(semilla, cfg, intervalo_fn, tiempo_limite=5)
    alts = base.attrs['alternativas']
    assert 1 <= len(alts) <= 3
    assert alts[0]['makespan_min'] == base.attrs['makespan_final_min']
    assert all(a['df']['Movimientos'].sum() == semilla['Movimientos'].sum() for a in alts)
    prop = ifc.proponer_minimas_cuadrillas(semilla, base, cfg, intervalo_fn, 15, tipo, 10)
    alts = prop.attrs['alternativas']
    assert 1 <= len(alts) <= 3
    assert alts[0]['df']['Fin'].max() == prop['Fin'].max()
    for a in alts:
        assert len(ifc.gruas_en_ventana(a['df'], prop.attrs['ventana_turno'])) == len(prop.attrs['gruas_turno_final'])
        assert ifc.detectar_interferencias(a['df'], 15).empty



def test_contar_traslados_y_penalizacion():
    df = cronograma(fila('STS 1', 2, 10, 0), fila('STS 1', 6, 10, 60), fila('STS 1', 2, 10, 120), fila('G5', 18, 10, 0))
    assert ifc.contar_traslados(df) == 2
    semilla, cfg = semilla_tres_gruas(), config('STS 1', 'G5', 'STS 2')
    base = ifc.proponer_secuencia(semilla, cfg, intervalo_fn, tiempo_limite=5, penalizacion_traslado_min=5)
    sin = ifc.proponer_secuencia(semilla, cfg, intervalo_fn, tiempo_limite=5, penalizacion_traslado_min=0)
    assert base.attrs['makespan_final_min'] <= sin.attrs['makespan_final_min'] + 1
    for a in base.attrs['alternativas']:
        assert a['traslados'] == ifc.contar_traslados(a['df'])
        assert a['makespan_penalizado_min'] == a['makespan_min'] + 5 * a['traslados']
    pen = [a['makespan_penalizado_min'] for a in base.attrs['alternativas'][1:]]
    assert pen == sorted(pen)

