import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pdfplumber
import re
import os
import io
import html
import time as time_module
from datetime import datetime, timedelta, time
from interferencias import (
    comparar_cronogramas,
    construir_colas_trabajo,
    detectar_interferencias,
    estado_solucion,
    orden_original_gruas,
    proponer_secuencia,
    resumen_cp_local,
)

# --- 1. CONFIGURACIÓN INICIAL Y ESTILOS ---
st.set_page_config(page_title="Portal Operaciones SVTI", layout="wide", page_icon="⚓")

st.markdown("""
    <style>
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: #f0f2f6;
        text-align: center;
        border: 1px solid #d1d5db;
        transition: 0.3s;
        height: 100%;
    }
    .card:hover { transform: scale(1.02); border-color: #3a506b; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .card h3 { color: #1f77b4; margin-bottom: 10px; }
    .card p { color: #555; }
    </style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# --- 2. CONFIGURACIÓN GLOBAL (Del Simulador) ---
BREAKS_CONFIG = [
    (2, 20, 3, 0, "COLACIÓN T3"),
    (6, 30, 8, 0, "CAMBIO TURNO"),
    (11, 20, 12, 0, "COLACIÓN T1"),
    (18, 50, 19, 30, "COLACIÓN T2")
]

PARES_COLORES_GRUAS = [
    ("#D35400", "#F39C12"),
    ("#2980B9", "#5DADE2"),
    ("#27AE60", "#58D68D"),
    ("#8E44AD", "#AF7AC5"),
    ("#C0392B", "#E74C3C"),
]
GANTT_BG_COLOR = "#1b263b"
NOMBRES_GRUA = ["STS 1", "STS 2", "G5", "G6", "G9", "G10"]

def tipo_grua(nombre):
    texto = str(nombre).upper().strip()
    if "STS" in texto:
        return "STS"
    if "MHC" in texto or re.fullmatch(r"G\s*\d+", texto):
        return "MHC"
    return "Otra"

def obtener_colores_gruas(lista_gruas):
    """Asigna el par de colores de descarga/embarque a cada grúa."""
    return {
        nombre: PARES_COLORES_GRUAS[indice % len(PARES_COLORES_GRUAS)]
        for indice, nombre in enumerate(lista_gruas)
    }

def mostrar_leyenda_grua(nombre_grua, colores, destino=st):
    """Muestra junto a los controles el color de cada operación de la grúa."""
    color_descarga, color_embarque = colores
    nombre_seguro = html.escape(str(nombre_grua))
    destino.markdown(
        f"<div style='margin:4px 0 2px;font-size:0.85rem'>"
        f"<b>{nombre_seguro}</b> &nbsp;"
        f"<span style='color:{color_descarga}'>■ Descarga</span> &nbsp;"
        f"<span style='color:{color_embarque}'>■ Embarque</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

def clasificar_zona_tarea(row):
    """Detecta cubierta/bodega si el PDF o Excel contiene esa indicación textual."""
    bay = str(row.get('Bay', '')).upper().strip()
    bay_match = re.search(r"\b\d+\s*([DH])\b", bay)
    if bay_match:
        return "Cubierta" if bay_match.group(1) == "D" else "Bodega"

    texto = " ".join(str(valor) for valor in row.values if pd.notna(valor)).upper()
    if re.search(r"\b(CUBIERTA|DECK|ON\s*DECK|UPPER\s*DECK)\b", texto):
        return "Cubierta"
    if re.search(r"\b(BODEGA|HOLD|UNDER\s*DECK)\b", texto):
        return "Bodega"
    return "Sin identificar"

def generar_hovertemplates_gantt(bays_reales):
    templates = []
    for bay_real in bays_reales:
        zona = "" if re.search(r"\d+\s*[HD]\s*$", str(bay_real).upper()) else "<br>%{customdata[2]}"
        templates.append(
            "<b>%{customdata[0]}</b><br>%{customdata[1]} Movimientos"
            f"{zona}<br>In: %{{customdata[3]|%H:%M}}"
            "<br>Out: %{customdata[4]|%H:%M}"
            "<br>Neto: %{customdata[5]} min<extra></extra>"
        )
    return templates

# --- 3. FUNCIONES DEL MÓDULO 1: SIMULADOR ---

def obtener_bay_visual(bay_real):
    try:
        match = re.search(r"(\d+)", str(bay_real))
        if match:
            b = float(match.group(1))
            k = round((b - 2) / 4)
            bay_visual = 2 + (4 * k)
            return int(max(2, bay_visual))
        return 0
    except: return 0

def normalizar_columnas(df):
    nuevas_columnas = {}
    for col in df.columns:
        col_str = str(col).upper().strip().replace('\n', ' ')
        if "TOTAL" in col_str and "M" in col_str: nuevas_columnas[col] = "Total M"
        elif "MOV" in col_str and "TOTAL" not in col_str: nuevas_columnas[col] = "Mov"
        elif "ACTION" in col_str: nuevas_columnas[col] = "Action Name"
        elif "BAY" in col_str: nuevas_columnas[col] = "Bay"
        elif "SEQ" in col_str: nuevas_columnas[col] = "Seq"
        elif col not in nuevas_columnas: nuevas_columnas[col] = col_str
    return df.rename(columns=nuevas_columnas)

@st.cache_data
def procesar_pdf(pdf_file):
    datos_totales = []
    vessel_name = None
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i, pagina in enumerate(pdf.pages):
                texto = pagina.extract_text()
                if i == 0 and texto:
                    m = re.search(r"\b\d{1,6}\s+([A-Z][A-Z0-9\- ]{1,80}?)\s+[A-Z0-9]{1,6}\b", texto, re.IGNORECASE)
                    if m: vessel_name = m.group(1).strip()
                
                grua_match = re.search(r"Crane No\.:\s*([\w\d]+)", texto)
                nombre_grua = grua_match.group(1) if grua_match else f"Grúa_{i+1}"

                tabla = pagina.extract_table({"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 4})
                
                if tabla and len(tabla) > 1:
                    indice_header = -1
                    for idx, fila in enumerate(tabla[:15]):
                        fila_texto = " ".join([str(celda).upper() for celda in fila if celda is not None])
                        if "BAY" in fila_texto and ("MOV" in fila_texto or "ACTION" in fila_texto):
                            indice_header = idx
                            break
                    
                    if indice_header != -1:
                        headers = tabla[indice_header]
                        data = tabla[indice_header + 1:]
                        df = pd.DataFrame(data, columns=headers)
                        df = normalizar_columnas(df)
                        if 'Mov' not in df.columns: df['Mov'] = 0
                        df = df.dropna(subset=['Bay'])
                        df = df[df['Bay'].astype(str).str.strip() != '']
                        df['Grúa'] = nombre_grua
                        datos_totales.append(df)
    except Exception as e: return None, None
    
    if datos_totales: return pd.concat(datos_totales, ignore_index=True), vessel_name
    return None, vessel_name
def procesar_excel(uploaded_file):
    try:
        # 1. Carga inteligente de archivo
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file)
                if len(df.columns) < 2: 
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';')
            except: return None, None
        else:
            # Intenta leer hoja específica, si no, la primera disponible
            try:
                df = pd.read_excel(uploaded_file, sheet_name='SECUENCIA_EJECUTIVA')
            except:
                df = pd.read_excel(uploaded_file)

        # 2. Normalización de columnas para el nuevo formato
        # Mapeamos las columnas del Excel a las que usa el simulador
        col_map = {
            'Crane': 'Grúa',
            'Grua_Optima': 'Grúa',
            'Start': 'Seq',           # Usamos 'Start' para ordenar la secuencia (1o, 2o, 3o...)
            'Secuencia_Global': 'Seq',
            'Bay': 'Bay',
            'Operacion': 'Action Name',
            'Tipo_Movimiento': 'Action Name'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # 3. Lógica Clave: Definir la cantidad de movimientos (Mov)
        # Si existe la columna 'Duration' o pares Start/End, asumimos que la diferencia 
        # es la CANTIDAD de contenedores (Work Queue), no el tiempo.
        
        if 'Duration' in df.columns:
            # Caso Nuevo Formato: Asumimos que Duration = Cantidad de Movimientos
            df['Mov'] = pd.to_numeric(df['Duration'], errors='coerce').fillna(0)
        elif 'End' in df.columns and 'Seq' in df.columns: # 'Seq' es el antiguo 'Start' renombrado arriba
            # Alternativa: Calculamos la diferencia
            df['Mov'] = pd.to_numeric(df['End'], errors='coerce') - pd.to_numeric(df['Seq'], errors='coerce')
        elif 'Mov' not in df.columns:
            # Por defecto, si no hay info, 1 movimiento por fila
            df['Mov'] = 1
            
        # Asegurar que no haya movimientos negativos o cero por error
        df['Mov'] = df['Mov'].apply(lambda x: x if x > 0 else 1)

        # 4. Traducción estándar de acciones (Embarque/Descarga)
        def traducir_accion(texto):
            t = str(texto).upper()
            if "DESCARGA" in t or "REESTIBA OFF" in t: return "DISCH"
            if "EMBARQUE" in t or "REESTIBA ON" in t: return "LOAD"
            return t 
            
        if 'Action Name' in df.columns:
            df['Action Name'] = df['Action Name'].apply(traducir_accion)

        # 5. Limpieza final
        df = df.dropna(subset=['Bay'])
        vessel_name = uploaded_file.name.replace('.xlsx', '').replace('.csv', '')
        
        return df, vessel_name

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
        return None, None
    
    
def calcular_fin_con_descansos(inicio_dt, duracion_minutos):
    cursor = inicio_dt
    minutos_pendientes = duracion_minutos
    while minutos_pendientes > 0.001:
        dia_cursor = cursor.date()
        siguiente_corte = None
        salto_hasta = None
        for h_ini, m_ini, h_fin, m_fin, _ in BREAKS_CONFIG:
            b_inicio = datetime.combine(dia_cursor, time(h_ini, m_ini))
            b_fin = datetime.combine(dia_cursor, time(h_fin, m_fin))
            if b_fin < b_inicio: b_fin += timedelta(days=1)
            if b_inicio <= cursor < b_fin:
                cursor = b_fin
                siguiente_corte = None 
                break 
            if b_inicio > cursor:
                if siguiente_corte is None or b_inicio < siguiente_corte:
                    siguiente_corte = b_inicio
                    salto_hasta = b_fin
        if siguiente_corte is None:
             siguiente_corte = datetime.combine(dia_cursor + timedelta(days=1), time(0,0))
             salto_hasta = siguiente_corte 
        tiempo_hasta_corte = (siguiente_corte - cursor).total_seconds() / 60
        if tiempo_hasta_corte >= minutos_pendientes:
            cursor += timedelta(minutes=minutos_pendientes)
            minutos_pendientes = 0
        else:
            cursor = siguiente_corte
            minutos_pendientes -= tiempo_hasta_corte
            if salto_hasta and salto_hasta > siguiente_corte: cursor = salto_hasta
    return cursor

def _fin_descanso_actual(cursor):
    """Devuelve el fin del descanso activo en cursor, si existe."""
    for offset in (-1, 0):
        dia = cursor.date() + timedelta(days=offset)
        for h_ini, m_ini, h_fin, m_fin, _ in BREAKS_CONFIG:
            b_inicio = datetime.combine(dia, time(h_ini, m_ini))
            b_fin = datetime.combine(dia, time(h_fin, m_fin))
            if b_fin < b_inicio:
                b_fin += timedelta(days=1)
            if b_inicio <= cursor < b_fin:
                return b_fin
    return None

def generar_turnos_trabajo(inicio_dt, cantidad=9):
    """Genera los próximos turnos de 08:00–15:30, 15:30–23:00 y 23:00–06:30."""
    turnos = []
    fecha = inicio_dt.date() - timedelta(days=1)
    while len(turnos) < cantidad:
        turnos_del_dia = [
            (datetime.combine(fecha, time(8, 0)), datetime.combine(fecha, time(15, 30))),
            (datetime.combine(fecha, time(15, 30)), datetime.combine(fecha, time(23, 0))),
            (datetime.combine(fecha, time(23, 0)), datetime.combine(fecha + timedelta(days=1), time(6, 30))),
        ]
        turnos.extend((inicio, fin) for inicio, fin in turnos_del_dia if fin > inicio_dt)
        fecha += timedelta(days=1)
    return turnos[:cantidad]

def _gmph_en_hora(cursor, gmph_schedule):
    """Busca el rendimiento del turno absoluto que contiene cursor."""
    for inicio, fin, gmph in gmph_schedule:
        if inicio <= cursor < fin:
            return float(gmph)
    return None

def _siguiente_cambio_horario(cursor, gmph_schedule):
    """Encuentra el próximo fin de turno o inicio de descanso."""
    candidatos = [fin for inicio, fin, _ in gmph_schedule if fin > cursor]
    for offset in (-1, 0, 1):
        dia = cursor.date() + timedelta(days=offset)
        for h_ini, m_ini, h_fin, m_fin, _ in BREAKS_CONFIG:
            b_inicio = datetime.combine(dia, time(h_ini, m_ini))
            b_fin = datetime.combine(dia, time(h_fin, m_fin))
            if b_fin < b_inicio:
                b_fin += timedelta(days=1)
            if b_inicio > cursor:
                candidatos.append(b_inicio)
            if b_fin > cursor:
                candidatos.append(b_fin)
    return min(candidatos) if candidatos else None

def calcular_intervalo_con_gmph(inicio_dt, movimientos, gmph_schedule):
    """Calcula inicio efectivo, fin y minutos netos con GMPH por turno."""
    cursor = inicio_dt
    movimientos_pendientes = float(movimientos)
    inicio_efectivo = None
    minutos_netos = 0.0

    while movimientos_pendientes > 0.000001:
        fin_descanso = _fin_descanso_actual(cursor)
        if fin_descanso is not None:
            cursor = fin_descanso
            continue

        gmph_actual = _gmph_en_hora(cursor, gmph_schedule)
        if gmph_actual is None:
            siguientes_turnos = [
                inicio for inicio, fin, _ in gmph_schedule if inicio > cursor and fin > cursor
            ]
            if not siguientes_turnos:
                raise ValueError(
                    "La tarea supera los turnos GMPH configurados para esta grúa. Revise los rendimientos y amplíe los turnos configurados."
                )
            cursor = min(siguientes_turnos)
            continue

        siguiente_cambio = _siguiente_cambio_horario(cursor, gmph_schedule)
        if siguiente_cambio is None:
            raise ValueError(
                "La tarea supera los turnos GMPH configurados para esta grúa. Revise los rendimientos y amplíe los turnos configurados."
            )

        if inicio_efectivo is None:
            inicio_efectivo = cursor

        minutos_disponibles = (siguiente_cambio - cursor).total_seconds() / 60
        minutos_necesarios = movimientos_pendientes * 60 / gmph_actual
        minutos_trabajados = min(minutos_disponibles, minutos_necesarios)
        minutos_netos += minutos_trabajados
        movimientos_pendientes -= minutos_trabajados * gmph_actual / 60
        cursor += timedelta(minutes=minutos_trabajados)

    return inicio_efectivo, cursor, round(minutos_netos, 1)

def estimar_turnos_necesarios(movimientos, inicio_dt, gmph_promedio=20, max_turnos=9):
    """Estima los turnos requeridos según la cola y un GMPH promedio de referencia."""
    if movimientos <= 0:
        return 1

    turnos = generar_turnos_trabajo(inicio_dt, cantidad=max_turnos)
    schedule = [(inicio, fin, gmph_promedio) for inicio, fin in turnos]
    try:
        _, fin_estimado, _ = calcular_intervalo_con_gmph(
            inicio_dt, movimientos, schedule
        )
    except ValueError:
        return max_turnos

    for indice, (_, fin_turno) in enumerate(turnos, start=1):
        if fin_estimado <= fin_turno:
            return indice
    return max_turnos

def calcular_intervalo_tarea(params, inicio, movimientos, duracion_fija=None):
    """Inicio efectivo, fin y minutos netos de una tarea, con las reglas de descansos y GMPH del cronograma."""
    cursor = pd.Timestamp(inicio).to_pydatetime()
    fin_descanso = _fin_descanso_actual(cursor)
    while fin_descanso is not None:
        cursor = fin_descanso
        fin_descanso = _fin_descanso_actual(cursor)

    schedule = params.get('gmph_schedule')
    if duracion_fija is None and schedule:
        return calcular_intervalo_con_gmph(cursor, movimientos, schedule)

    duracion = duracion_fija if duracion_fija is not None else movimientos / params.get('gmph', 20) * 60
    return cursor, calcular_fin_con_descansos(cursor, duracion), round(duracion, 1)

def estimar_movimientos_por_grua(df):
    """Suma la cola por grúa, interpretando correctamente Total M acumulado."""
    resultados = {}
    if df is None or df.empty or 'Grúa' not in df.columns:
        return resultados

    datos = df.copy()
    if 'Seq' in datos.columns:
        datos['Seq'] = pd.to_numeric(datos['Seq'], errors='coerce')
        datos = datos.sort_values(['Grúa', 'Seq'])

    for nombre_grua, grupo in datos.groupby('Grúa'):
        total_estimado = 0.0
        last_total_m = 0.0
        for _, row in grupo.iterrows():
            total_acumulado = extraer_numero(row.get('Total M', ''))
            mov_fila = extraer_numero(row.get('Mov', '')) or 0
            if total_acumulado is not None:
                delta = total_acumulado - last_total_m
                if delta >= 0:
                    movimientos = delta
                    last_total_m = total_acumulado
                else:
                    movimientos = mov_fila
                    last_total_m += mov_fila
            else:
                movimientos = mov_fila
                last_total_m += mov_fila
            total_estimado += max(0, movimientos)
        resultados[nombre_grua] = total_estimado
    return resultados

def configurar_gmph_por_grua(
    nombre_grua, gmph_default, minimo, maximo, clave_gmph, inicio_dt, movimientos_estimados
):
    """Muestra el GMPH constante o permite definirlo para cada uno de 9 turnos."""
    horas_turno_efectivas = 6.83
    usar_contenedores_turno = st.checkbox(
        "Ingresar contenedores deseados por turno",
        value=False,
        key=f"usar_contenedores_turno_{clave_gmph}",
        help=(
            "Al activarlo, el rendimiento ingresado por grúa se interpreta como "
            "contenedores por turno y se convierte usando 6,83 horas efectivas."
        ),
    )
    factor_conversion = (
        horas_turno_efectivas
        if usar_contenedores_turno
        else 1
    )
    unidad_clave = "contenedores_turno" if usar_contenedores_turno else "gmph"
    minimo_ingresado = float(minimo) * factor_conversion
    maximo_ingresado = float(maximo) * factor_conversion
    paso_ingresado = 1.0 if factor_conversion != 1 else 0.1
    etiqueta_rendimiento = (
        "Contenedores deseados por turno" if usar_contenedores_turno else "GMPH"
    )

    def capturar_gmph(etiqueta, clave, valor_default=gmph_default):
        valor_inicial = float(valor_default) * factor_conversion
        if factor_conversion != 1:
            valor_inicial = float(round(valor_inicial))
        valor_ingresado = st.number_input(
            etiqueta,
            minimo_ingresado,
            maximo_ingresado,
            valor_inicial,
            paso_ingresado,
            key=clave,
        )
        return float(valor_ingresado) / factor_conversion

    usar_rangos = st.checkbox(
        "Definir rendimiento por turno",
        value=False,
        key=f"usar_rangos_gmph_{clave_gmph}"
    )
    if not usar_rangos:
        gmph = capturar_gmph(
            etiqueta_rendimiento,
            clave_gmph if factor_conversion == 1 else f"{clave_gmph}_{unidad_clave}",
        )
        st.caption(
            f"Proyección calculada a {gmph:.2f} GMPH"
            + (
                f" desde {horas_turno_efectivas:.2f} horas efectivas por turno."
                if usar_contenedores_turno else "."
            )
        )
        return gmph, None

    turnos_estimados = estimar_turnos_necesarios(
        movimientos_estimados, inicio_dt, gmph_promedio=20
    )
    rendimiento_referencia = 20 * factor_conversion
    unidad_referencia = (
        f"{rendimiento_referencia:.0f} contenedores por turno (20 GMPH)"
        if usar_contenedores_turno else "20 GMPH"
    )
    st.caption(
        f"Cola estimada: {movimientos_estimados:,.0f} movimientos. "
        f"A {unidad_referencia} se proyectan {turnos_estimados} turno(s)."
    )
    cantidad_turnos = st.number_input(
        "Turnos a configurar",
        min_value=1,
        max_value=9,
        value=turnos_estimados,
        step=1,
        key=f"cantidad_turnos_{clave_gmph}_{turnos_estimados}",
        help="El valor inicial se estima con 20 movimientos por hora; se puede ajustar hasta 9.",
    )
    st.caption(
        f"Ingresa el rendimiento en {etiqueta_rendimiento.lower()}; "
        "se convierte a GMPH para la proyección. "
        "Los descansos siguen excluidos."
    )
    turnos = generar_turnos_trabajo(inicio_dt, cantidad=int(cantidad_turnos))
    columnas = st.columns(3)
    schedule = []
    for indice, (inicio_turno, fin_turno) in enumerate(turnos):
        etiqueta = (
            f"Turno {indice + 1} · {inicio_turno:%d/%m} "
            f"{inicio_turno:%H:%M}–{fin_turno:%H:%M}"
        )
        clave_turno = f"gmph_turno_{clave_gmph}_{indice + 1}_{inicio_turno:%Y%m%d}"
        if factor_conversion != 1:
            clave_turno = f"{clave_turno}_{unidad_clave}"
        with columnas[indice % 3]:
            etiqueta_turno = f"{etiqueta} · {etiqueta_rendimiento}"
            gmph_turno = capturar_gmph(etiqueta_turno, clave_turno)
            if usar_contenedores_turno:
                st.caption(f"Equivale a {gmph_turno:.2f} GMPH")
        schedule.append((inicio_turno, fin_turno, gmph_turno))
    return gmph_default, schedule

def extraer_numero(texto):
    match = re.search(r'(\d+)', str(texto))
    return float(match.group(1)) if match else None

def calcular_cronograma(df, config_gruas):
    cronograma = []
    if 'Seq' in df.columns:
        df['Seq'] = pd.to_numeric(df['Seq'], errors='coerce')
        df = df.sort_values(by=['Grúa', 'Seq'])

    for nombre_grua, grupo_df in df.groupby("Grúa"):
        params = config_gruas.get(nombre_grua, {'inicio': datetime.now(), 'gmph': 20})
        tiempo_actual, gmph = params['inicio'], params.get('gmph', 20)
        gmph_schedule = params.get('gmph_schedule')
        last_total_m = 0
        
        for index, row in grupo_df.iterrows():
            raw_action, raw_bay = str(row.get('Action Name', '')).upper(), str(row.get('Bay', '')).upper()
            zona_tarea = clasificar_zona_tarea(row)
            current_total_m = extraer_numero(row.get('Total M', ''))
            movimientos = 0
            
            if current_total_m is not None:
                delta = current_total_m - last_total_m
                if delta >= 0: movimientos, last_total_m = delta, current_total_m
                else: movimientos, last_total_m = extraer_numero(row.get('Mov', '')) or 0, last_total_m + (extraer_numero(row.get('Mov', '')) or 0)
            else:
                movimientos = extraer_numero(row.get('Mov', '')) or 0
                last_total_m += movimientos
            
            texto_fila = f"{raw_bay} {raw_action}"
            bay_visual_num = obtener_bay_visual(raw_bay)
            if bay_visual_num == 0 and "PONTOON" not in texto_fila: continue

            bay_visual_str = f"{bay_visual_num:02d}" 
            duracion_minutos, categoria_visual = 0, "Otro"
            
            if "PONTOON" in texto_fila or "OPEN" in texto_fila or "CLOSE" in texto_fila:
                duracion_minutos, categoria_visual = 15, "Pontón"
                if bay_visual_num == 0: bay_visual_str = "General" 
            elif movimientos > 0:
                if gmph_schedule:
                    inicio_tarea, tiempo_fin, duracion_minutos = calcular_intervalo_con_gmph(
                        tiempo_actual, movimientos, gmph_schedule
                    )
                else:
                    inicio_tarea = tiempo_actual
                    duracion_minutos = (movimientos / gmph) * 60
                kws_descarga = ["DISCH", "DSCH", "BLISCH", "DISH", "DDI", "DHI", "SCH", "DCH", "REESTIBA OFF"]
                kws_embarque = ["LOAD", "HOAD", "DOAD", "LOD", "LOA", "O AD", " HL", " DL", "AD TWI", "REESTIBA ON"]
                
                if any(kw in texto_fila for kw in kws_descarga): categoria_visual = f"{nombre_grua} - Descarga"
                elif any(kw in texto_fila for kw in kws_embarque): categoria_visual = f"{nombre_grua} - Embarque"

            if duracion_minutos > 0:
                if not gmph_schedule or "PONTOON" in texto_fila or "OPEN" in texto_fila or "CLOSE" in texto_fila:
                    inicio_tarea = tiempo_actual
                    tiempo_fin = calcular_fin_con_descansos(tiempo_actual, duracion_minutos)
                cronograma.append({
                    "Grúa": nombre_grua, "Bay Real": raw_bay, "Bay Visual": bay_visual_str,
                    "Acción": f"{raw_action.replace(chr(10), ' ') or 'Contexto'} ({int(movimientos)})",
                    "Movimientos": int(movimientos),
                    "Inicio": inicio_tarea, "Fin": tiempo_fin, "Duración Neta": round(duracion_minutos, 1),
                    "Categoría Visual": categoria_visual, "Zona": zona_tarea
                })
                tiempo_actual = tiempo_fin
    return pd.DataFrame(cronograma)

def _tipo_operacion_programa(row):
    accion = str(row.get("Acción", "")).upper()
    categoria = str(row.get("Categoría Visual", "")).upper()
    bay = str(row.get("Bay Real", "")).upper()
    if "PONT" in categoria or "PONTOON" in accion or "PONTOON" in bay:
        return "Pontoons"
    if "REESTIBA" in accion or "RESTOW" in accion:
        return "Restow"
    if "DESCARGA" in categoria or any(palabra in accion for palabra in ("DISCH", "DSCH", "DESCARGA")):
        return "Discharge"
    if "EMBARQUE" in categoria or any(palabra in accion for palabra in ("LOAD", "EMBARQUE")):
        return "Load"
    return None

def construir_programa_trabajo(df_cronograma):
    operaciones = ["Discharge", "Load", "Restow", "Pontoons"]
    gruas = list(dict.fromkeys(df_cronograma["Grúa"].astype(str)))
    gruas.sort(key=lambda grua: ({"STS": 0, "MHC": 1}.get(tipo_grua(grua), 2), grua))

    programa = df_cronograma.copy()
    programa["Operation"] = programa.apply(_tipo_operacion_programa, axis=1)
    programa = programa[programa["Operation"].isin(operaciones)].copy()
    programa["Quantity"] = pd.to_numeric(programa["Movimientos"], errors="coerce").fillna(0)
    es_ponton = programa["Operation"] == "Pontoons"
    programa.loc[es_ponton, "Quantity"] = programa.loc[es_ponton, "Quantity"].where(
        programa.loc[es_ponton, "Quantity"] > 0, 1
    )

    filas = []
    for operacion in operaciones:
        fila = {"Operation": operacion}
        datos_operacion = programa[programa["Operation"] == operacion]
        for grua in gruas:
            cantidad = datos_operacion.loc[datos_operacion["Grúa"] == grua, "Quantity"].sum()
            fila[grua] = int(cantidad)
        fila["Total"] = sum(fila[grua] for grua in gruas)
        filas.append(fila)

    tabla = pd.DataFrame(filas, columns=["Operation", *gruas, "Total"])
    totales = {"Operation": "Total"}
    for grua in gruas:
        totales[grua] = int(tabla[grua].sum())
    totales["Total"] = int(tabla["Total"].sum())
    tabla.loc[len(tabla)] = totales

    resumen = {
        operacion: int(tabla.loc[tabla["Operation"] == operacion, "Total"].iloc[0])
        for operacion in operaciones
    }
    return tabla, resumen

def calcular_turnos_con_gruas(df_cronograma):
    inicios = pd.to_datetime(df_cronograma["Inicio"])
    fines = pd.to_datetime(df_cronograma["Fin"])
    inicio_operacion = inicios.min()
    fin_operacion = fines.max()
    turnos = [
        (time(8, 0), time(15, 30)),
        (time(15, 30), time(23, 0)),
        (time(23, 0), time(6, 30)),
    ]
    filas_turnos = []

    for dia in pd.date_range(inicio_operacion.date() - timedelta(days=1), fin_operacion.date(), freq="D").date:
        for hora_inicio, hora_fin in turnos:
            inicio_turno = datetime.combine(dia, hora_inicio)
            fin_turno = datetime.combine(dia, hora_fin)
            if fin_turno <= inicio_turno:
                fin_turno += timedelta(days=1)
            tareas_turno = (inicios < fin_turno) & (fines > inicio_turno)
            if tareas_turno.any():
                tipos = pd.Series(
                    df_cronograma.loc[tareas_turno, "Grúa"].astype(str).unique()
                ).map(tipo_grua).value_counts()
                filas_turnos.append({
                    "turno": len(filas_turnos) + 1,
                    "STS": int(tipos.get("STS", 0)),
                    "MHC": int(tipos.get("MHC", 0)),
                    "Otra": int(tipos.get("Otra", 0)),
                })

    return filas_turnos

def descripcion_cuadrillas(fila):
    partes = [f"{fila[t]} {t if t != 'Otra' else 'otra(s)'}" for t in ("STS", "MHC", "Otra") if fila[t]]
    detalle = partes[0] if len(partes) == 1 else ", ".join(partes[:-1]) + " y " + partes[-1] if partes else ""
    total = fila["STS"] + fila["MHC"] + fila["Otra"]
    return f"{total} cuadrilla{'s' if total != 1 else ''}: {detalle}"

def construir_tabla_turnos_trabajados(df_cronograma):
    return pd.DataFrame([
        {"Gangs shifts": f"TURNO {f['turno']}", "Gangs / STS / MHC": descripcion_cuadrillas(f)}
        for f in calcular_turnos_con_gruas(df_cronograma)
    ])

def generar_reporte_pdf(df_cronograma, tabla_programa, nombre_buque, alias_gruas=None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    azul, gris = colors.HexColor("#1F3A5F"), colors.HexColor("#F2F4F7")
    est_titulo = ParagraphStyle("t", fontSize=18, leading=22, textColor=azul, fontName="Helvetica-Bold")
    est_sub = ParagraphStyle("s", fontSize=11, leading=14, textColor=colors.HexColor("#444444"))
    est_sec = ParagraphStyle("h", fontSize=11, leading=14, textColor=colors.white, fontName="Helvetica-Bold",
                             backColor=azul, borderPadding=(3, 4, 3, 4))

    inicio = pd.to_datetime(df_cronograma["Inicio"]).min()
    fin = pd.to_datetime(df_cronograma["Fin"]).max()
    horas = max((fin - inicio).total_seconds() / 3600, 1e-9)
    horas_grua = (pd.to_datetime(df_cronograma["Fin"]) - pd.to_datetime(df_cronograma["Inicio"])).dt.total_seconds().sum() / 3600

    gruas = list(tabla_programa.columns[1:-1])
    fila = lambda op: tabla_programa[tabla_programa["Operation"] == op].iloc[0]
    desc, emb, tot = fila("Discharge"), fila("Load"), fila("Total")
    total_movs = int(tot["Total"])

    def kpi(titulo, valor):
        t = Table([[titulo], [valor]], colWidths=[3.3 * cm], rowHeights=[0.7 * cm, 1.1 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), gris), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.grey), ("FONTSIZE", (0, 1), (-1, 1), 16),
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"), ("TEXTCOLOR", (0, 1), (-1, 1), azul),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        return t

    def fila_kpi(items):
        t = Table([[kpi(a, b) for a, b in items]], hAlign="LEFT")
        t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
        return t

    elementos = [
        Paragraph("REPORTE EJECUTIVO DE PLANIFICACIÓN", est_titulo),
        Paragraph(f"M/N {html.escape(str(nombre_buque).upper())}", est_sub),
        Paragraph(f"Fecha de operación: {inicio:%d/%m/%Y}", est_sub),
        Spacer(1, 0.5 * cm),
        Paragraph("MOVIMIENTOS", est_sec), Spacer(1, 0.2 * cm),
        fila_kpi([("MOVIMIENTOS TOTALES", f"{total_movs:,}"), ("DESCARGA", f"{int(desc['Total']):,}"),
                  ("EMBARQUE", f"{int(emb['Total']):,}"), ("GRÚAS", str(len(gruas)))]),
        Spacer(1, 0.5 * cm),
        Paragraph("VENTANA OPERACIONAL", est_sec), Spacer(1, 0.2 * cm),
        fila_kpi([("INICIO", f"{inicio:%d/%m %H:%M}"), ("TÉRMINO EST.", f"{fin:%d/%m %H:%M}"),
                  ("DURACIÓN", f"{horas:.1f} h"), ("PRODUCTIVIDAD", f"{total_movs / horas:.0f} mov/h"),
                  ("CRANE DENSITY", f"{horas_grua / horas:.2f}")]),
        Spacer(1, 0.5 * cm),
        Paragraph("DISTRIBUCIÓN POR GRÚA", est_sec), Spacer(1, 0.2 * cm),
    ]

    filas = [["Grúa", "Tipo", "Bahías", "Descarga", "Embarque", "Total", "Participación"]]
    for g in gruas:
        original = (alias_gruas or {}).get(g, g)
        bays = pd.to_numeric(
            df_cronograma.loc[df_cronograma["Grúa"].astype(str).eq(str(original)), "Bay Visual"], errors="coerce"
        ).dropna()
        rango = f"{int(bays.min()):02d}-{int(bays.max()):02d}" if not bays.empty else "-"
        total_g = int(tot[g])
        filas.append([g, tipo_grua(original), rango, int(desc[g]), int(emb[g]), total_g,
                      f"{(total_g / total_movs * 100) if total_movs else 0:.1f}%"])
    filas.append(["TOTAL", "", "", int(desc["Total"]), int(emb["Total"]), total_movs, "100.0%"])

    tabla = Table(filas, repeatRows=1, colWidths=[2.6 * cm, 1.8 * cm, 2.6 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm, 3 * cm])
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), azul), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), gris), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, gris]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elementos += [tabla, Spacer(1, 0.6 * cm), Paragraph("PARTICIPACIÓN POR GRÚA (MOVIMIENTOS)", est_sec),
                  Spacer(1, 0.2 * cm)]

    dibujo = Drawing(16 * cm, 7 * cm)
    grafico = VerticalBarChart()
    grafico.x, grafico.y, grafico.width, grafico.height = 40, 30, 16 * cm - 60, 7 * cm - 50
    grafico.data = [[int(desc[g]) for g in gruas], [int(emb[g]) for g in gruas]]
    grafico.categoryAxis.categoryNames = [str(g) for g in gruas]
    grafico.bars[0].fillColor, grafico.bars[1].fillColor = azul, colors.HexColor("#E08A1E")
    grafico.valueAxis.valueMin = 0
    grafico.categoryAxis.labels.fontSize = grafico.valueAxis.labels.fontSize = 8
    dibujo.add(grafico)
    elementos += [dibujo, Paragraph("Azul: descarga | Naranja: embarque", est_sub), Spacer(1, 0.5 * cm),
                  Paragraph("CUADRILLAS POR TURNO DE TRABAJO", est_sec), Spacer(1, 0.2 * cm)]

    filas_turnos = [["Turno", "Cuadrillas", "STS", "MHC", "Detalle"]]
    for f in calcular_turnos_con_gruas(df_cronograma):
        filas_turnos.append([f"Turno {f['turno']}", f["STS"] + f["MHC"] + f["Otra"], f["STS"], f["MHC"],
                             descripcion_cuadrillas(f)])
    tabla_turnos = Table(filas_turnos, repeatRows=1, colWidths=[2.4 * cm, 2.4 * cm, 1.8 * cm, 1.8 * cm, 8.6 * cm])
    tabla_turnos.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), azul), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (1, 0), (3, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, gris]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elementos.append(tabla_turnos)

    def pie(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.2 * cm, f"M/N {nombre_buque} - Planificación de nave")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
        canvas.restoreState()

    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm).build(elementos, onFirstPage=pie, onLaterPages=pie)
    return buffer.getvalue()

# --- 4. FUNCIONES DEL MÓDULO 2: VISUALIZADOR 3D ---

# --- 6.5 MÓDULO NUEVO: PLANIFICADOR ONE SPHERE (STS) ---


# --- A. LÓGICA DE NEGOCIO (Adaptada de tu script) ---

def determinar_sts(row, tier):
    """Lógica específica de One Sphere para determinar restricción STS"""
    try:
        row, tier = int(row), int(tier)
        es_cubierta = (tier >= 80)
        es_impar = (row % 2 != 0)
        
        if es_cubierta:
            # Cubierta: STS si Row > 13 y es impar
            if row > 13 and es_impar: return 'Si'
        else:
            # Bodega: STS si Row 7-17 y es impar
            if es_impar and 7 <= row <= 17: return 'Si'
        return 'No'
    except:
        return 'No'

def parse_baplie_completo_stream(file_bytes):
    """Parser robusto que lee desde memoria (BytesIO) en lugar de ruta de archivo"""
    try:
        content = file_bytes.getvalue().decode('utf-8', errors='ignore')
        segments = content.replace('\n', '').split("'")
        
        container_list = []
        current_container = {}
        vessel_name = "Unknown"
        in_group = False
        
        for seg in segments:
            seg = seg.strip()
            parts = seg.split('+')
            tag = parts[0]
            
            if tag == 'TDT' and len(parts) > 8:
                try: vessel_name = parts[8].split(':')[-1]
                except: pass

            if tag == 'LOC' and len(parts) > 1 and parts[1] == '147':
                if current_container: container_list.append(current_container)
                
                stow = parts[2].split(':')[0]
                bay = int(stow[0:3]) if len(stow) >= 6 else 0
                row = int(stow[3:5]) if len(stow) >= 6 else 0
                tier = int(stow[5:7]) if len(stow) >= 6 else 0
                    
                current_container = {
                    'ID': '', 'ISO': '', 'Peso': 0.0, 'POD': '', 'POL': '', 'Operador': '',
                    'Bay': bay, 'Row': row, 'Tier': tier, 'Stowage': stow, 'Buque': vessel_name
                }
                in_group = True
                
            elif in_group:
                if tag == 'EQD' and len(parts) > 2:
                    current_container['ID'] = parts[2]
                    if len(parts) > 3: current_container['ISO'] = parts[3]
                elif tag == 'LOC':
                    if parts[1] == '9': current_container['POL'] = parts[2].split(':')[0]
                    elif parts[1] == '11': current_container['POD'] = parts[2].split(':')[0]
                elif tag == 'MEA' and len(parts) > 3 and ':' in parts[3]:
                    try: current_container['Peso'] = float(parts[3].split(':')[1])
                    except: pass
                elif tag == 'NAD' and len(parts) > 2 and parts[1] == 'CA':
                    current_container['Operador'] = parts[2]

        if current_container: container_list.append(current_container)
        return pd.DataFrame(container_list)
    except Exception as e:
        st.error(f"Error al leer BAPLIE: {e}")
        return pd.DataFrame()

def generar_movimientos_onesphere(file_arr, file_dep, puerto_obj='CLSVE'):
    """Genera la lista maestra de movimientos detectando STS, Dig-Outs y Restibas"""
    
    # 1. Parsing
    df_arr = parse_baplie_completo_stream(file_arr)
    df_dep = parse_baplie_completo_stream(file_dep)
    
    if df_arr.empty or df_dep.empty: return pd.DataFrame()

    lista_movimientos = []

    # 2. Descargas
    descargas = df_arr[df_arr['POD'] == puerto_obj].copy()
    for _, row in descargas.iterrows():
        lista_movimientos.append({
            'ID_Contenedor': row['ID'], 'Tipo_Movimiento': 'DESCARGA',
            'Prioridad_Base': 20, 'Bay': row['Bay'], 'Row': row['Row'], 'Tier': row['Tier'],
            'Origen': row['Stowage'], 'Destino': 'YARD',
            'ISO': row['ISO'], 'Peso': row['Peso'], 'Operador': row['Operador']
        })

    # 3. Embarques
    embarques = df_dep[df_dep['POL'] == puerto_obj].copy()
    for _, row in embarques.iterrows():
        lista_movimientos.append({
            'ID_Contenedor': row['ID'], 'Tipo_Movimiento': 'EMBARQUE',
            'Prioridad_Base': 30, 'Bay': row['Bay'], 'Row': row['Row'], 'Tier': row['Tier'],
            'Origen': 'YARD', 'Destino': row['Stowage'],
            'ISO': row['ISO'], 'Peso': row['Peso'], 'Operador': row['Operador']
        })

    # 4. Reestibas (Lógica compleja)
    transito_arr = df_arr[df_arr['POD'] != puerto_obj]
    transito_dep = df_dep[df_dep['POL'] != puerto_obj]
    
    if not transito_arr.empty and not transito_dep.empty:
        merged = pd.merge(transito_arr, transito_dep[['ID', 'Stowage', 'Bay', 'Row', 'Tier']], 
                          on='ID', suffixes=('_In', '_Out'), how='inner')

        # 4.1 Shiftings (Cambio de posición)
        shifts = merged[merged['Stowage_In'] != merged['Stowage_Out']].copy()
        for _, row in shifts.iterrows():
            # Movimiento de bajada
            lista_movimientos.append({
                'ID_Contenedor': row['ID'], 'Tipo_Movimiento': 'REESTIBA OFF', 'Prioridad_Base': 10,
                'Bay': row['Bay_In'], 'Row': row['Row_In'], 'Tier': row['Tier_In'],
                'Origen': row['Stowage_In'], 'Destino': 'BUFFER',
                'ISO': row['ISO'], 'Peso': row['Peso'], 'Operador': row['Operador']
            })
            # Movimiento de subida
            lista_movimientos.append({
                'ID_Contenedor': row['ID'], 'Tipo_Movimiento': 'REESTIBA ON', 'Prioridad_Base': 40,
                'Bay': row['Bay_Out'], 'Row': row['Row_Out'], 'Tier': row['Tier_Out'],
                'Origen': 'BUFFER', 'Destino': row['Stowage_Out'],
                'ISO': row['ISO'], 'Peso': row['Peso'], 'Operador': row['Operador']
            })

        # 4.2 Dig-Outs (Movimientos forzados por descarga abajo)
        quietos = merged[merged['Stowage_In'] == merged['Stowage_Out']].copy()
        ids_quietos = set(quietos['ID'])
        
        # Identificar pilas donde hay descargas
        if not descargas.empty:
            pilas_con_descarga = descargas[['Bay', 'Row']].drop_duplicates()
            
            for _, coord in pilas_con_descarga.iterrows():
                b, r = coord['Bay'], coord['Row']
                # Analizamos la pila de arriba a abajo o viceversa
                pila = df_arr[(df_arr['Bay'] == b) & (df_arr['Row'] == r)].sort_values('Tier')
                
                hay_descarga_abajo = False
                # Recorremos para ver si hay algo de tránsito tapando una descarga
                for _, cont in pila.iterrows():
                    if cont['POD'] == puerto_obj:
                        hay_descarga_abajo = True
                    else:
                        # Si es de tránsito y hay una descarga abajo, es Dig-Out
                        if hay_descarga_abajo and cont['ID'] in ids_quietos:
                            lista_movimientos.append({
                                'ID_Contenedor': cont['ID'], 'Tipo_Movimiento': 'REESTIBA OFF (DIG)', 'Prioridad_Base': 1,
                                'Bay': cont['Bay'], 'Row': cont['Row'], 'Tier': cont['Tier'],
                                'Origen': cont['Stowage'], 'Destino': 'BUFFER',
                                'ISO': cont['ISO'], 'Peso': cont['Peso'], 'Operador': cont['Operador']
                            })
                            lista_movimientos.append({
                                'ID_Contenedor': cont['ID'], 'Tipo_Movimiento': 'REESTIBA ON (RET)', 'Prioridad_Base': 45,
                                'Bay': cont['Bay'], 'Row': cont['Row'], 'Tier': cont['Tier'],
                                'Origen': 'BUFFER', 'Destino': cont['Stowage'],
                                'ISO': cont['ISO'], 'Peso': cont['Peso'], 'Operador': cont['Operador']
                            })

    df_final = pd.DataFrame(lista_movimientos)
    if df_final.empty: return df_final

    # 5. Aplicar Lógica STS
    df_final['STS'] = df_final.apply(lambda x: determinar_sts(x['Row'], x['Tier']), axis=1)
    
    return df_final

# --- B. INTERFAZ GRÁFICA ---

def preparar_datos_editor(df_op):
    """Agrupa por Bay, Operación Y CONDICIÓN STS (Si/No)"""
    if df_op.empty: return pd.DataFrame()
    
    # 1. Categoría Visual de Operación
    def categorizar(txt):
        t = str(txt).upper()
        if "DESCARGA" in t or "OFF" in t: return "📥 DISCH"
        if "EMBARQUE" in t or "ON" in t: return "📤 LOAD"
        return "🔄 OTRO"

    df_op['Op_Sim'] = df_op['Tipo_Movimiento'].apply(categorizar)

    # 2. Agrupar por Bay, Operación Y STS (Aquí ocurre la separación mágica)
    resumen = df_op.groupby(['Bay', 'Op_Sim', 'STS']).agg(
        Total_Mov=('ID_Contenedor', 'count')
    ).reset_index()
    
    # 3. Crear una columna visual para el STS
    def etiqueta_sts(valor):
        return "🔴 Solo STS" if valor == 'Si' else "🟢 Standard"
    
    resumen['Condicion'] = resumen['STS'].apply(etiqueta_sts)

    # 4. Ordenamiento lógico
    # Orden: Bay -> Descarga/Carga -> Primero los STS (para darles prioridad)
    resumen['Orden_Op'] = resumen['Op_Sim'].map({'📥 DISCH': 1, '📤 LOAD': 2, '🔄 OTRO': 3})
    resumen['Orden_STS'] = resumen['STS'].map({'Si': 0, 'No': 1}) # 0 aparece antes que 1
    
    resumen = resumen.sort_values(['Bay', 'Orden_Op', 'Orden_STS'])

    # Columnas por defecto para el editor
    resumen['Grúa'] = "Sin Asignar"
    resumen['Secuencia'] = 0
    
    # Retornamos 'Condicion' en lugar de 'Mov_STS' (la barra antigua)
    return resumen[['Bay', 'Op_Sim', 'Condicion', 'Total_Mov', 'Grúa', 'Secuencia']]

def render_planificador_sts():
    st.title("🧩 Planificador One Sphere (STS)")
    st.markdown("Generación de secuencia maestra con detección automática de **Dig-Outs** y **Restricciones STS**.")

    # 1. Carga de Archivos
    c1, c2 = st.columns(2)
    f_arr = c1.file_uploader("BAPLIE Llegada (.txt)", key="sts_arr")
    f_dep = c2.file_uploader("BAPLIE Salida (.txt)", key="sts_dep")
    
    if f_arr and f_dep:
        if 'df_movimientos_sts' not in st.session_state:
            with st.spinner("🔄 Procesando lógica STS y detectando Dig-Outs..."):
                st.session_state.df_movimientos_sts = generar_movimientos_onesphere(f_arr, f_dep)
                st.session_state.df_resumen_sts = preparar_datos_editor(st.session_state.df_movimientos_sts)

        df_full = st.session_state.df_movimientos_sts
        
        # 2. Métricas Generales
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Movimientos", len(df_full))
        col3.metric("⚠️ Movimientos STS", len(df_full[df_full['STS']=='Si']))
        col5.metric("Reestibas Totales", len(df_full[df_full['Tipo_Movimiento'].str.contains('REESTIBA', na=False)]))

        st.divider()

        # 3. Editor de Planificación (Working List)
        c_config, c_editor = st.columns([1, 2])
        
        with c_config:
            st.subheader("⚙️ Grúas")
            n_gruas = st.number_input("N° Grúas", 1, 4, 2)
            lista_gruas = [f"G{i+1}" for i in range(n_gruas)]
            config_gruas = {}
            df_asignacion_previa = st.session_state.get("editor_sts")
            if not isinstance(df_asignacion_previa, pd.DataFrame):
                df_asignacion_previa = st.session_state.df_resumen_sts
            movimientos_por_grua = {}
            if {'Grúa', 'Total_Mov'}.issubset(df_asignacion_previa.columns):
                asignados_previos = df_asignacion_previa[
                    df_asignacion_previa['Grúa'].isin(lista_gruas)
                ]
                movimientos_por_grua = asignados_previos.groupby('Grúa')['Total_Mov'].sum().to_dict()
            colores_config_gruas = obtener_colores_gruas(lista_gruas)
            
            for g in lista_gruas:
                mostrar_leyenda_grua(g, colores_config_gruas[g], destino=c_config)
                with st.expander(f"Config {g}", expanded=False):
                    # --- NUEVO: Selector de Tipo de Grúa ---
                    tipo_grua = st.selectbox(f"Tipo {g}", ["STS", "MHC"], key=f"tipo_{g}")
                    
                    hora = st.time_input(f"Inicio", time(8,0), key=f"h_{g}")
                    inicio_grua = datetime.combine(datetime.today(), hora)
                    gmph, gmph_schedule = configurar_gmph_por_grua(
                        g, 22, 20, 40, f"gmph_{g}", inicio_grua,
                        movimientos_por_grua.get(g, 0)
                    )
                    
                    # Guardamos el tipo en el diccionario de configuración
                    config_gruas[g] = {
                        "inicio": inicio_grua,
                        "gmph": gmph,
                        "gmph_schedule": gmph_schedule,
                        "tipo": tipo_grua  # <--- Guardado aquí
                    }
            # Botón de Descarga Excel (Igual que tu script original)
            st.write("---")
            if st.button("📥 Descargar Excel Maestro"):
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_full.to_excel(writer, sheet_name='MASTER_SEQUENCE', index=False)
                    # Resumen
                    pivot = pd.pivot_table(df_full, index=['Bay','STS'], columns='Tipo_Movimiento', values='ID_Contenedor', aggfunc='count', fill_value=0).reset_index()
                    pivot.to_excel(writer, sheet_name='Resumen_STS', index=False)
                
                st.download_button(
                    label="Guardar Excel",
                    data=buffer.getvalue(),
                    file_name="Master_Plan_Secuencia_SVTI_Restrict.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        with c_editor:
            st.subheader("📋 Asignación de Bays")
            st.info("Se han separado las filas por condición STS (Rojo) vs Standard (Verde).")
            
            df_editado = st.data_editor(
                st.session_state.df_resumen_sts,
                column_config={
                    "Bay": st.column_config.TextColumn("Bay", disabled=True, width="small"),
                    "Op_Sim": st.column_config.TextColumn("Operación", disabled=True, width="small"),
                    
                    # --- CAMBIO AQUÍ: Nueva columna de Condición ---
                    "Condicion": st.column_config.TextColumn("Tipo Bloque", disabled=True, width="medium"),
                    
                    # --- CAMBIO AQUÍ: Quitamos Mov_STS y dejamos solo Total ---
                    "Total_Mov": st.column_config.NumberColumn("Cant. Cont.", disabled=True),
                    
                    "Grúa": st.column_config.SelectboxColumn("Asignar Grúa", options=["Sin Asignar"] + lista_gruas, required=True),
                    "Secuencia": st.column_config.NumberColumn("Orden", min_value=0, step=1)
                },
                use_container_width=True,
                hide_index=True,
                key="editor_sts"
            )
        # 4. Simulación Visual (Gantt)
    # 4. Simulación Visual (Estilo Simulador Completo)
        if st.button("🚀 Calcular Secuencia Visual", type="primary", use_container_width=True):
            errores_asignacion = []
            
            # Recorremos lo que el usuario editó para buscar infracciones
            for i, row in df_editado.iterrows():
                grua = row['Grúa']
                condicion = row['Condicion'] # "🔴 Restringido" o "🟢 Standard"
                
                if grua != "Sin Asignar":
                    # Recuperamos qué tipo de grúa es (STS o MHC) desde la config
                    tipo_actual = config_gruas[grua]['tipo']
                    
                    # REGLA DE ORO: Si es Restringido Y la grúa es MHC -> ERROR
                    if "Solo STS" in condicion and tipo_actual == "MHC":
                        errores_asignacion.append(f"⛔ **Bay {row['Bay']} ({condicion})**: No puedes asignar la grúa **{grua}** porque es **MHC**.")

            # Si encontramos errores, detenemos todo y mostramos la alerta
            if errores_asignacion:
                st.error("Error de Planificación: Se han violado reglas de asignación STS/MHC.")
                for e in errores_asignacion:
                    st.write(e)
                st.stop()            
              # --- A. UNIFICACIÓN Y PREPARACIÓN DE DATOS ---
            # 1. Unimos las tablas (Restringido + Standard) si usaste la lógica de listas separadas
            # Si no usaste la lógica separada, df_editado_final es simplemente df_editado
            df_procesar = df_editado

            asignados = df_procesar[df_procesar['Grúa'] != "Sin Asignar"].sort_values(['Grúa', 'Secuencia'])
            
            if asignados.empty:
                st.warning("⚠️ Debes asignar grúas a los bloques antes de calcular.")
                st.stop()

            # 2. Traducción para el Motor de Cálculo
            df_sim = []
            for _, row in asignados.iterrows():
                # TRUCO: Limpiamos el emoji para que 'calcular_cronograma' detecte "DISCH" o "LOAD"
                # y asigne el color Azul o Naranja automáticamente.
                op_raw = str(row['Op_Sim']).upper()
                tipo_accion = "DISCH" if "DISCH" in op_raw else ("LOAD" if "LOAD" in op_raw else "MOV")
                
                # Detectamos si es crítico para ponerlo en el nombre (tooltip)
                es_critico = "Restringido" in str(row['Condicion']) or "Solo STS" in str(row['Condicion'])
                etiqueta_adicional = " (STS ONLY)" if es_critico else ""

                df_sim.append({
                    'Grúa': row['Grúa'],
                    'Bay': str(row['Bay']), # El simulador espera texto o número
                    'Action Name': f"{tipo_accion}{etiqueta_adicional}", # Esto define el color
                    'Mov': row['Total_Mov'],
                    'Seq': row['Secuencia'],
                    'Total M': None # No necesario si damos Mov explícito
                })
            
            df_para_gantt = pd.DataFrame(df_sim)

            # --- B. CÁLCULO DE TIEMPOS (Motor Matemático) ---
            # Usamos la misma función que el simulador PDF
            try:
                df_cronograma = calcular_cronograma(df_para_gantt, config_gruas)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            
            if df_cronograma.empty:
                st.error("No se pudo calcular el cronograma. Revisa las asignaciones.")
            else:
                # --- C. VISUALIZACIÓN IDENTICA AL SIMULADOR (Dark Mode) ---
                
                # 1. Métricas Rápidas
                fin_max = df_cronograma['Fin'].max()
                inicio_min = df_cronograma['Inicio'].min()
                duracion_total = (fin_max - inicio_min).total_seconds() / 3600
                total_movs = asignados['Total_Mov'].sum()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Fin Operación", fin_max.strftime("%d/%m %H:%M"))
                m2.metric("Duración Total", f"{duracion_total:.1f} Hrs")
                m3.metric("Total Movimientos", int(total_movs))
                
                st.divider()

                # 2. Configuración de Colores (Copia exacta del Simulador)
                mapa_colores = { "Pontón": "#7F8C8D", "Otro": "#BDC3C7" }
                gruas_activas = sorted(df_cronograma['Grúa'].unique())
                colores_por_grua = obtener_colores_gruas(gruas_activas)
                for nombre_g, (color_desc, color_emb) in colores_por_grua.items():
                    mapa_colores[f"{nombre_g} - Descarga"] = color_desc
                    mapa_colores[f"{nombre_g} - Embarque"] = color_emb

                # 3. Construcción del Gráfico (Plotly Graph Objects)
                range_y_start = inicio_min - timedelta(minutes=30)
                range_y_end = fin_max + timedelta(minutes=30)
                
                # Ordenamiento de Bays en el eje X
                try:
                    # Intentamos ordenar numéricamente
                    bays_nums = df_cronograma['Bay Visual'].unique()
                    lista_ordenada = sorted(bays_nums, key=lambda x: int(x) if str(x).isdigit() else 999)
                except:
                    lista_ordenada = sorted(df_cronograma['Bay Visual'].unique())
# 1. Filtramos solo los valores numéricos para calcular min y max
                bays_numericos = [int(b) for b in df_cronograma['Bay Visual'].unique() if str(b).isdigit()]
                
                if bays_numericos:
                    min_bay = min(bays_numericos)
                    max_bay = max(bays_numericos)
                    
                    # Generamos la secuencia completa (02, 06, 10...) usando paso de 4
                    # Esto rellena los huecos vacíos visualmente
                    lista_ordenada = [f"{b:02d}" for b in range(min_bay, max_bay + 4, 4)]
                else:
                    # Fallback por si acaso no hay bays numéricos
                    lista_ordenada = sorted(df_cronograma['Bay Visual'].unique())

                fig = go.Figure()

                # Dibujar Barras
                for categoria, color in mapa_colores.items():
                    df_cat = df_cronograma[df_cronograma['Categoría Visual'] == categoria]
                    if df_cat.empty: continue
                    for zona, df_zona in df_cat.groupby('Zona', dropna=False):
                        duracion_ms = (df_zona['Fin'] - df_zona['Inicio']).dt.total_seconds() * 1000
                        es_cubierta = zona == "Cubierta"
                        patron = {"shape": "/" if es_cubierta else ""}
                        if es_cubierta:
                            patron.update({
                                "fgcolor": color,
                                "bgcolor": GANTT_BG_COLOR,
                                "solidity": 0.75,
                            })
                        fig.add_trace(go.Bar(
                            name=categoria,
                            x=df_zona['Bay Visual'],
                            y=duracion_ms,
                            base=df_zona['Inicio'],
                            orientation='v',
                            marker=dict(
                                color=color,
                                line=dict(color='black', width=1),
                                pattern=patron,
                            ),
                            opacity=1.0,
                            width=0.95,
                            customdata=df_zona[['Bay Real', 'Movimientos', 'Zona', 'Inicio', 'Fin', 'Duración Neta']],
                            hovertemplate=generar_hovertemplates_gantt(df_zona['Bay Real']),
                            showlegend=not es_cubierta,
                        ))

                # 4. Dibujar Descansos (Breaks - Franjas Negras)
                dias_unicos = pd.to_datetime(df_cronograma['Inicio']).dt.date.unique()
                ultimo_dia = df_cronograma['Fin'].max().date()
                if ultimo_dia not in dias_unicos:
                    import numpy as np
                    dias_unicos = np.append(dias_unicos, ultimo_dia)

                for dia in dias_unicos:
                    for h_ini, m_ini, h_fin, m_fin, label in BREAKS_CONFIG:
                        t_inicio = datetime.combine(dia, time(h_ini, m_ini))
                        t_fin = datetime.combine(dia, time(h_fin, m_fin))
                        
                        # Ajuste por si el break cruza medianoche
                        if t_fin < t_inicio: t_fin += timedelta(days=1)

                        fig.add_hrect(
                            y0=t_inicio, y1=t_fin, 
                            fillcolor="rgba(50, 50, 50, 1.0)", 
                            line_width=0, layer="above",
                            annotation_text=label, 
                            annotation_position="inside left",
                            annotation_y=t_inicio + (t_fin - t_inicio) / 2,
                            annotation_font_color="white", 
                            annotation_font_size=10
                        )

                # 5. Layout Oscuro (Idéntico al Simulador)
                fig.update_layout(
                    height=800, 
                    title="Planificación Maestra STS (Visualización Simulador)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='white'), bgcolor='rgba(0,0,0,0)'),
                    dragmode='pan', hovermode='closest', barmode='overlay', bargap=0.02,
                    plot_bgcolor=GANTT_BG_COLOR, paper_bgcolor=GANTT_BG_COLOR, font=dict(color='white'),
                    yaxis=dict(
                        type='date', 
                        title="Hora", 
                        tickformat="%H:%M", 
                        tickfont=dict(color='white'), 
                        autorange="reversed", # De arriba hacia abajo (mañana -> noche)
                        range=[range_y_start, range_y_end], 
                        gridcolor='#3a506b'
                    ),
                    xaxis=dict(
                        title="Bay", 
                        type='category', 
                        categoryorder='array', 
                        categoryarray=lista_ordenada, 
                        side="top", 
                        tickfont=dict(color='white'), 
                        gridcolor='#000000', 
                        fixedrange=True
                    ),
                    margin=dict(l=60, r=20, t=100, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Ver detalle de tiempos calculados"):
                    st.dataframe(df_cronograma)

# --- 5. FUNCIONES DEL MÓDULO 3: OPERACIONES (One Sphere & Logic) ---

@st.cache_data
def parse_baplie_advanced(file_content):
    content = file_content.decode("utf-8", errors="ignore")
    segments = content.replace('\n', '').split("'")
    
    container_list = []
    current_container = {}
    vessel_name = "Unknown"
    in_group = False
    
    for seg in segments:
        seg = seg.strip()
        parts = seg.split('+')
        tag = parts[0]
        
        if tag == 'TDT':
            try:
                if len(parts) > 8: vessel_name = parts[8].split(':')[-1]
            except: pass

        if tag == 'LOC' and len(parts)>1 and parts[1] == '147':
            if current_container: container_list.append(current_container)
            stow = parts[2].split(':')[0]
            bay = int(stow[0:3]) if len(stow) >= 6 else 0
            row = int(stow[3:5]) if len(stow) >= 6 else 0
            tier = int(stow[5:7]) if len(stow) >= 6 else 0
                
            current_container = {
                'Buque': vessel_name, 'ID': '', 'Stowage': stow,
                'Bay': bay, 'Row': row, 'Tier': tier,
                'POL': '', 'POD': '', 'Peso': 0.0
            }
            in_group = True
        elif in_group:
            if tag == 'LOC':
                if parts[1] == '9': current_container['POL'] = parts[2].split(':')[0]
                elif parts[1] == '11': current_container['POD'] = parts[2].split(':')[0]
            elif tag == 'EQD' and len(parts)>2: current_container['ID'] = parts[2]
            elif tag == 'MEA' and len(parts)>3 and ':' in parts[3]:
                try: current_container['Peso'] = float(parts[3].split(':')[1])
                except: pass

    if current_container: container_list.append(current_container)
    return pd.DataFrame(container_list), vessel_name

def generar_dataframe_operativo(df_arr, df_dep, puerto_obj='CLSVE'):
    df_arr['Tipo_Op'] = 'Tránsito'
    df_arr.loc[df_arr['POD'] == puerto_obj, 'Tipo_Op'] = 'Descarga'
    
    embarque = df_dep[df_dep['POL'] == puerto_obj].copy()
    embarque['Tipo_Op'] = 'Embarque'
    
    transit_arr = df_arr[df_arr['Tipo_Op'] == 'Tránsito']
    transit_dep = df_dep[df_dep['POL'] != puerto_obj]
    
    merged = pd.merge(transit_arr[['ID', 'Stowage']], transit_dep[['ID', 'Stowage']], on='ID', suffixes=('_In', '_Out'))
    ids_reestiba = merged[merged['Stowage_In'] != merged['Stowage_Out']]['ID'].tolist()
    
    df_arr.loc[df_arr['ID'].isin(ids_reestiba), 'Tipo_Op'] = 'Reestiba'
    
    df_final = pd.concat([df_arr, embarque], ignore_index=True)
    return df_final

def aplicar_logica_one_sphere(df, nombre_buque):
    if "ONE SPHERE" not in nombre_buque.upper():
        df['Color_Visual'] = df['Tipo_Op'] 
        return df
    
    def regla_color(row):
        t, r = row['Tier'], row['Row']
        es_rojo = False
        es_impar = (r % 2 != 0)
        
        if es_impar:
            if t > 50 and r >= 13: es_rojo = True
            elif t <= 50 and (7 <= r <= 17): es_rojo = True
        
        return "⚠️ Restricción OneSphere" if es_rojo else row['Tipo_Op']

    df['Color_Visual'] = df.apply(regla_color, axis=1)
    return df

def create_3d_plot_operacion(df, vessel_name):
    def calcular_y_visual(row):
        if row == 0: return 0
        return row if row % 2 != 0 else -row

    df['Eje_Y_Visual'] = df['Row'].apply(calcular_y_visual)
    
    color_map = {
        'Descarga': '#3498db', 'Embarque': '#2ecc71', 'Reestiba': '#f39c12',
        'Tránsito': '#95a5a6', '⚠️ Restricción OneSphere': '#e74c3c'
    }

    fig = px.scatter_3d(
        df, x='Bay', y='Eje_Y_Visual', z='Tier',
        color='Color_Visual', color_discrete_map=color_map,
        hover_data=['ID', 'Row', 'Tier', 'Stowage', 'Tipo_Op'],
        title=f"Operación 3D: {vessel_name}",
        labels={'Eje_Y_Visual': 'Manga (Row)', 'Bay': 'Eslora (Bay)', 'Tier': 'Altura (Tier)'},
        opacity=0.9, height=750
    )
    
    fig.update_traces(marker=dict(size=6, line=dict(width=1, color='black')))
    fig.update_layout(scene=dict(aspectmode='data'))
    
    max_row = df['Row'].max()
    tick_vals = [i for i in range(-max_row, max_row + 1, 2)] 
    tick_text = [str(abs(i)) for i in tick_vals] 
    fig.update_layout(scene=dict(yaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, title='Manga (Row ISO)')))

    return fig

# --- 6. RENDERIZADORES DE PÁGINAS ---

def render_menu():
    st.title("⚓ Portal de Operaciones SVTI")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h3>🏗️ Simulador</h3><p>Gantt desde PDF</p></div>', unsafe_allow_html=True)
        if st.button("Ir al Simulador", key="b1", use_container_width=True): navigate_to('simulador')
    with c2:
        st.markdown('<div class="card"><h3>🚢 Desarrollo de Working List</p></div>', unsafe_allow_html=True)
        if st.button("Ir al Visualizador", key="b2", use_container_width=True): navigate_to('planificador_sts')
    with c3:
        st.markdown('<div class="card"><h3>📊 Visualización Operación</h3><p>Descarga, Carga y Reglas One Sphere</p></div>', unsafe_allow_html=True)
        if st.button("Ir a Operaciones", key="b3", use_container_width=True): navigate_to('operacion')

def mostrar_tabla_ejecutiva(df, resaltar_ultima=False):
    """Renderiza un DataFrame como tabla HTML de estilo ejecutivo (solo lectura)."""
    encabezados = "".join(
        f"<th class='{'num' if pd.api.types.is_numeric_dtype(df[c]) else ''}'>{html.escape(str(c))}</th>"
        for c in df.columns
    )
    filas = []
    for i, (_, fila) in enumerate(df.iterrows()):
        clase = "total" if resaltar_ultima and i == len(df) - 1 else ""
        celdas = "".join(
            f"<td class='{'num' if pd.api.types.is_number(v) else ''}'>"
            f"{html.escape(str(v))}</td>" for v in fila
        )
        filas.append(f"<tr class='{clase}'>{celdas}</tr>")
    st.markdown(
        """<style>
        table.tbl-exec{width:100%;border-collapse:collapse;margin:10px 0 22px;font-size:0.92rem;
            font-family:'Roboto',sans-serif}
        table.tbl-exec th{background:transparent;color:#9FB3D1;text-align:left;padding:10px 14px;
            font-weight:500;letter-spacing:.12em;text-transform:uppercase;font-size:0.7rem;
            border-bottom:2px solid #C9A227}
        table.tbl-exec td{padding:11px 14px;border-bottom:1px solid rgba(159,179,209,.18);color:#E6EDF7;
            background:transparent}
        table.tbl-exec tbody tr:hover td{background:rgba(201,162,39,.07)}
        table.tbl-exec td.num{text-align:right;font-variant-numeric:tabular-nums}
        table.tbl-exec th.num{text-align:right}
        table.tbl-exec tr.total td{color:#C9A227;font-weight:700;border-top:1px solid #C9A227;
            border-bottom:none;background:rgba(201,162,39,.06)}
        .etc-banner{display:flex;flex-direction:column;gap:4px;margin:6px 0 20px;padding:16px 22px;
            border-left:4px solid #C9A227;background:linear-gradient(90deg,rgba(201,162,39,.12),rgba(0,0,0,0));
            border-radius:0 8px 8px 0}
        .etc-banner span{color:#9FB3D1;letter-spacing:.14em;font-size:0.68rem}
        .etc-banner b{color:#fff;font-size:1.3rem;letter-spacing:.03em;font-weight:500}
        </style>"""
        f"<table class='tbl-exec'><thead><tr>{encabezados}</tr></thead><tbody>{''.join(filas)}</tbody></table>",
        unsafe_allow_html=True,
    )

ALTO_PLOT_GANTT = 540
SLOT_PUENTE, SLOT_CHIMENEA = " ", "  "

def svg_buque(categorias, estribor_muelle):
    """Buque ilustrado con volumen (SVG, vista lateral) con una pila por columna de bay."""
    from math import pi, sin
    ancho, cubierta, fondo, flotacion, y_top, alto = 1000, 180, 222, 208, 40, 186
    n = len(categorias)
    cw = ancho / n
    hw = cw * 0.46
    X = (lambda x: x) if estribor_muelle else (lambda x: ancho - x)
    deriva = -1 if estribor_muelle else 1
    ch, dx, dy = 11, 5, 4

    ruta = (f"M{X(0)},{cubierta} L{X(ancho)},{cubierta} Q{X(ancho - 5)},{cubierta + 26} {X(ancho - 52)},{fondo} "
            f"L{X(16)},{fondo} L{X(0)},{cubierta + 6} Z")
    s = [f"""<defs>
<linearGradient id="hull" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#31507F"/><stop offset=".5" stop-color="#16294A"/><stop offset="1" stop-color="#070D1A"/></linearGradient>
<linearGradient id="white" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#D8E0EA"/></linearGradient>
<linearGradient id="glass" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#9ED8FF"/><stop offset="1" stop-color="#3C8FD6"/></linearGradient>
<linearGradient id="fun" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#E24B3D"/><stop offset="1" stop-color="#A52A20"/></linearGradient>
<clipPath id="hc"><path d="{ruta}"/></clipPath>
<filter id="blur"><feGaussianBlur stdDeviation="3"/></filter></defs>"""]

    # Casco sin ojos de buey: obra viva roja, cinta blanca y borde de cubierta
    s.append(f'<path d="{ruta}" fill="url(#hull)"/><g clip-path="url(#hc)">'
             f'<rect x="0" y="{flotacion}" width="{ancho}" height="30" fill="#A82D2D"/>'
             f'<rect x="0" y="{flotacion - 3}" width="{ancho}" height="3" fill="#F4F6F8"/>'
             f'<rect x="0" y="{cubierta}" width="{ancho}" height="3" fill="#C9D6EA"/>'
             f'<rect x="0" y="{cubierta + 3}" width="{ancho}" height="8" fill="#fff" opacity=".06"/></g>'
             f'<path d="{ruta}" fill="none" stroke="#9FB3D1" stroke-width="1" opacity=".6"/>'
             f'<polygon points="{X(ancho - 62)},{cubierta} {X(ancho - 62)},{cubierta - 16} {X(ancho - 14)},{cubierta - 16} '
             f'{X(ancho)},{cubierta - 8} {X(ancho)},{cubierta}" fill="#22385E" stroke="#9FB3D1"/>'
             f'<circle cx="{X(ancho - 24)}" cy="{cubierta + 14}" r="5" fill="#0A1222" stroke="#9FB3D1"/>')

    # Contenedores: dos colores con cara superior y lateral
    carga = [i for i, c in enumerate(categorias) if c not in (SLOT_PUENTE, SLOT_CHIMENEA)]
    sw = cw * 0.80
    for k, i in enumerate(carga):
        cx = (i + 0.5) * cw
        x0, x1 = cx - sw / 2, cx + sw / 2
        for t in range(2 + int(round(2.2 * sin(pi * (k + 0.5) / len(carga))))):
            y = cubierta - (t + 1) * ch
            frente, lado, tapa = (("#2D7FC1", "#1F5F99", "#6FB4E8") if (k + t) % 2 == 0
                                  else ("#E8743B", "#BF5624", "#F6A877"))
            s.append(f'<rect x="{x0:.1f}" y="{y}" width="{sw:.1f}" height="{ch - 1}" fill="{frente}" stroke="#0b1220" stroke-width=".8"/>'
                     f'<polygon points="{x0:.1f},{y} {x0 + dx:.1f},{y - dy} {x1 + dx:.1f},{y - dy} {x1:.1f},{y}" fill="{tapa}" stroke="#0b1220" stroke-width=".6"/>'
                     f'<polygon points="{x1:.1f},{y} {x1 + dx:.1f},{y - dy} {x1 + dx:.1f},{y + ch - 1 - dy} {x1:.1f},{y + ch - 1}" fill="{lado}" stroke="#0b1220" stroke-width=".6"/>')

    for i, categoria in enumerate(categorias):
        cx = (i + 0.5) * cw
        if categoria == SLOT_PUENTE:
            # Puente en T: torre delgada de alojamiento y cubierta de mando ancha arriba
            ht = hw * 0.40
            for d in range(5):
                y = cubierta - (d + 1) * 11
                s.append(f'<rect x="{cx - ht:.1f}" y="{y}" width="{2 * ht:.1f}" height="11" fill="url(#white)" stroke="#8FA0B8" stroke-width=".8"/>'
                         f'<polygon points="{cx + ht:.1f},{y} {cx + ht + 5:.1f},{y - 4} {cx + ht + 5:.1f},{y + 7} {cx + ht:.1f},{y + 11}" fill="#B9C4D3" stroke="#8FA0B8" stroke-width=".6"/>')
                for k in (-1, 1):
                    s.append(f'<rect x="{cx + k * ht * 0.5 - ht * 0.22:.1f}" y="{y + 4}" width="{ht * 0.44:.1f}" height="4" fill="#FFD36B"/>')
            y0 = cubierta - 55
            s.append(f'<rect x="{cx - hw * 0.95:.1f}" y="{y0 - 4}" width="{hw * 1.9:.1f}" height="4" rx="1" fill="#4B5C74"/>'
                     f'<polygon points="{cx - hw * 0.9:.1f},{y0 - 4} {cx + hw * 0.9:.1f},{y0 - 4} {cx + hw:.1f},{y0 - 18} {cx - hw:.1f},{y0 - 18}" fill="#0F2038" stroke="#9FB3D1" stroke-width=".8"/>'
                     f'<polygon points="{cx - hw * 0.85:.1f},{y0 - 7} {cx + hw * 0.85:.1f},{y0 - 7} {cx + hw * 0.94:.1f},{y0 - 15} {cx - hw * 0.94:.1f},{y0 - 15}" fill="url(#glass)"/>'
                     f'<rect x="{cx - hw * 1.02:.1f}" y="{y0 - 21}" width="{hw * 2.04:.1f}" height="3" rx="1" fill="#E7ECF3"/>'
                     f'<line x1="{cx}" y1="{y0 - 21}" x2="{cx}" y2="{y0 - 46}" stroke="#DDE3EA" stroke-width="1.6"/>'
                     f'<rect x="{cx - hw * 0.5:.1f}" y="{y0 - 36}" width="{hw:.1f}" height="2" fill="#DDE3EA"/>'
                     f'<ellipse cx="{cx}" cy="{y0 - 50}" rx="{hw * 0.26:.1f}" ry="5" fill="#F4F6F8" stroke="#9FB3D1"/>')
        elif categoria == SLOT_CHIMENEA:
            tope = cubierta - 78
            s.append(f'<rect x="{cx - hw * 0.9:.1f}" y="{cubierta - 20}" width="{hw * 1.8:.1f}" height="20" fill="url(#white)" stroke="#9FB3D1" stroke-width=".8"/>'
                     f'<polygon points="{cx - hw * 0.8:.1f},{cubierta - 20} {cx + hw * 0.8:.1f},{cubierta - 20} {cx + hw * 0.5:.1f},{tope + 6} {cx - hw * 0.5:.1f},{tope}" fill="url(#fun)" stroke="#6E1D16" stroke-width=".8"/>'
                     f'<polygon points="{cx - hw * 0.66:.1f},{cubierta - 46} {cx + hw * 0.66:.1f},{cubierta - 46} {cx + hw * 0.6:.1f},{cubierta - 56} {cx - hw * 0.6:.1f},{cubierta - 56}" fill="#F4F6F8"/>'
                     f'<circle cx="{cx}" cy="{cubierta - 51}" r="{min(hw * 0.22, 5):.1f}" fill="#12263F"/>'
                     f'<polygon points="{cx - hw * 0.5:.1f},{tope} {cx + hw * 0.5:.1f},{tope + 6} {cx + hw * 0.5:.1f},{tope + 14} {cx - hw * 0.5:.1f},{tope + 8}" fill="#0B1220"/>')
            for k in range(4):
                s.append(f'<ellipse cx="{cx + deriva * (k + 1) * hw * 0.5:.1f}" cy="{tope - 6 - k * 9}" rx="{hw * (0.3 + 0.12 * k):.1f}" ry="{6 + k * 3}" fill="#CBD5E1" opacity="{0.42 - 0.09 * k:.2f}" filter="url(#blur)"/>')

    ola = f"M0,{flotacion + 6}"
    for k in range(40):
        ola += f" Q{(k + 0.5) * ancho / 40:.1f},{flotacion + 6 + (3 if k % 2 == 0 else -3)} {(k + 1) * ancho / 40:.1f},{flotacion + 6}"
    s.append(f'<path d="{ola}" fill="none" stroke="#4FC3F7" stroke-opacity=".85" stroke-width="2"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {y_top} {ancho} {alto}" '
            f'preserveAspectRatio="none">{"".join(s)}</svg>')

def agregar_buque_al_grafico(fig, lista_ordenada, estribor_muelle, alto_plot=ALTO_PLOT_GANTT):
    """Dibuja el buque sobre el eje de bays; puente y chimenea ocupan su propia columna y bajan como barra gris."""
    import base64
    n = len(lista_ordenada)
    px = lambda v: 1 + v / alto_plot
    alto_img = 186
    fuente = "data:image/svg+xml;base64," + base64.b64encode(svg_buque(lista_ordenada, estribor_muelle).encode()).decode()
    fig.add_layout_image(source=fuente, xref="paper", yref="paper", x=0, y=px(36 + alto_img),
                         sizex=1, sizey=alto_img / alto_plot, xanchor="left", yanchor="top",
                         sizing="stretch", layer="above")

    if estribor_muelle:
        proa_x, popa_x, proa_ha, popa_ha = 0.93, 0.015, "right", "left"
    else:
        proa_x, popa_x, proa_ha, popa_ha = 0.07, 0.985, "left", "right"
    for texto, x, ha in (("PROA", proa_x, proa_ha), ("POPA", popa_x, popa_ha)):
        fig.add_annotation(text=texto, x=x, y=px(61), xref="paper", yref="paper", showarrow=False,
                           xanchor=ha, font=dict(color="#C9D6EA", size=10))

    for slot, texto, alto_marca in ((SLOT_PUENTE, "PUENTE", 214), (SLOT_CHIMENEA, "CHIMENEA", 228)):
        if slot not in lista_ordenada:
            continue
        c = (lista_ordenada.index(slot) + 0.5) / n
        fig.add_annotation(text=texto, x=c, y=px(alto_marca), xref="paper", yref="paper", showarrow=False,
                           xanchor="center", font=dict(color="white", size=10))
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=c - 0.46 / n, x1=c + 0.46 / n, y0=px(40), y1=0,
                      fillcolor="rgba(160,160,160,0.35)", line=dict(color="rgba(190,190,190,0.7)", width=1),
                      layer="above")

def construir_figura_gantt(
    df_cronograma, mapa_colores, lista_ordenada, estribor_muelle,
    intervalo_minutos, range_y_start, range_y_end,
):
    fig = go.Figure()

    for categoria, color in mapa_colores.items():
        df_cat = df_cronograma[df_cronograma['Categoría Visual'] == categoria]
        if df_cat.empty: continue
        for zona, df_zona in df_cat.groupby('Zona', dropna=False):
            duracion_ms = (df_zona['Fin'] - df_zona['Inicio']).dt.total_seconds() * 1000
            es_cubierta = zona == "Cubierta"
            patron = {"shape": "/" if es_cubierta else ""}
            if es_cubierta:
                patron.update({
                    "fgcolor": color,
                    "bgcolor": GANTT_BG_COLOR,
                    "solidity": 0.90,
                })
            fig.add_trace(go.Bar(
                name=categoria,
                x=df_zona['Bay Visual'], y=duracion_ms, base=df_zona['Inicio'],
                orientation='v',
                marker=dict(
                    color=color,
                    line=dict(color='black', width=1),
                    pattern=patron,
                ),
                opacity=1.0, width=0.95,
                customdata=df_zona[['Bay Real', 'Movimientos', 'Zona', 'Inicio', 'Fin', 'Duración Neta']],
                hovertemplate=generar_hovertemplates_gantt(df_zona['Bay Real']),
                showlegend=not es_cubierta,
            ))

    dias_unicos = pd.to_datetime(df_cronograma['Inicio']).dt.date.unique()
    if not df_cronograma.empty:
        ultimo_dia = df_cronograma['Fin'].max().date()
        if ultimo_dia not in dias_unicos:
            import numpy as np
            dias_unicos = np.append(dias_unicos, ultimo_dia)

    for dia in dias_unicos:
        for h_ini, m_ini, h_fin, m_fin, label in BREAKS_CONFIG:
            t_inicio = datetime.combine(dia, time(h_ini, m_ini))
            t_fin = datetime.combine(dia, time(h_fin, m_fin))
            fig.add_hrect(
                y0=t_inicio, y1=t_fin, fillcolor="rgba(50, 50, 50, 1.0)", line_width=0, layer="above",
                annotation_text=label, annotation_position="inside left",
                annotation_y=t_inicio + (t_fin - t_inicio) / 2,
                annotation_font_color="white", annotation_font_size=10
            )

    for dia in pd.date_range(range_y_start.date(), range_y_end.date(), freq="D").date:
        for hora_hito in (time(6, 30), time(8, 0), time(15, 30), time(23, 0)):
            fig.add_hline(
                y=datetime.combine(dia, hora_hito),
                line_color="white",
                line_dash="solid",
                line_width=1.5,
            )

    agregar_buque_al_grafico(fig, lista_ordenada, estribor_muelle)
    # Traza invisible: obliga a Plotly a mostrar todas las columnas, incluso las sin datos
    fig.add_trace(go.Scatter(x=lista_ordenada, y=[range_y_start] * len(lista_ordenada), mode="markers",
                             marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        height=900, legend=dict(orientation="h", yanchor="bottom", y=1 + 295 / ALTO_PLOT_GANTT, xanchor="right", x=1, font=dict(color='white'), bgcolor='rgba(0,0,0,0)'),
        dragmode='pan', hovermode='closest', barmode='overlay', bargap=0.02,
        plot_bgcolor=GANTT_BG_COLOR, paper_bgcolor=GANTT_BG_COLOR, font=dict(color='white'),
        yaxis=dict(type='date', title="Hora", tickformat="%H:%M", tickfont=dict(color='white'), dtick=intervalo_minutos*60*1000, autorange="reversed", range=[range_y_start, range_y_end], gridcolor='#3a506b'),
        xaxis=dict(title=None, type='category', categoryorder='array', categoryarray=lista_ordenada, side="top", tickfont=dict(color='white'), gridcolor='#000000', fixedrange=True),
        margin=dict(l=60, r=20, t=900 - 20 - ALTO_PLOT_GANTT, b=20)
    )
    return fig

METODOS_OPTIMIZACION = {
    "Reparación rápida": {"tiempo_limite": 10, "activar_alns": False, "porcentaje_vecindad": 0.10},
    "Híbrido recomendado": {"tiempo_limite": 20, "activar_alns": True, "porcentaje_vecindad": 0.10},
    "Intensivo": {"tiempo_limite": 60, "activar_alns": True, "porcentaje_vecindad": 0.15},
}

def exportar_colas_excel(colas):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for grua, cola in colas.items():
            nombre_hoja = re.sub(r'[\[\]:*?/\\]', '_', f"Cola {grua}")[:31]
            cola.to_excel(writer, sheet_name=nombre_hoja, index=False)
    return buffer.getvalue()

def mostrar_resumen_optimizacion(propuesta, tiempo_total, interf_iniciales, interf_finales):
    a = propuesta.attrs

    def mm(clave):
        v = a.get(clave)
        return "—" if v is None else f"{v:.0f} min"

    final, semilla = a.get("makespan_final_min"), a.get("makespan_semilla_min")
    primera = a.get("makespan_primera_factible_min")
    interf_semilla = a.get("interferencias_semilla", interf_iniciales)
    cruces_semilla = a.get("cruces_semilla")
    semilla_factible = a.get("semilla_factible")
    if semilla_factible is None:
        semilla_factible = not interf_semilla and not cruces_semilla
    with st.expander("Resumen de la optimización", expanded=True):
        st.write(f"Resultado: **{estado_solucion(a)}** · {'✅ factible' if a.get('factible') else '❌ no factible'}")
        filas = [
            ("Makespan de la semilla", mm("makespan_semilla_min")),
            ("Estado de la semilla", "factible" if semilla_factible else "no factible"),
            ("Interferencias / cruces iniciales", f"{interf_semilla} / {'—' if cruces_semilla is None else cruces_semilla}"),
            ("Primera solución factible", mm("makespan_primera_factible_min")),
            ("Antes del ALNS", mm("makespan_antes_alns_min")),
            ("Mejor candidata del ALNS", mm("makespan_mejor_candidata_alns_min")),
            ("Makespan final", mm("makespan_final_min")),
        ]
        if semilla_factible:
            if final is not None and semilla:
                filas.append(("Mejora respecto de la semilla", f"{semilla - final:+.0f} min ({(semilla - final) / semilla * 100:+.1f} %)"))
        elif primera is not None and semilla is not None:
            filas.append(("Variación necesaria para alcanzar factibilidad", f"{primera - semilla:+.0f} min"))
        if primera is not None and final is not None and primera:
            filas.append((
                "Mejora desde la primera solución factible",
                f"{primera - final:+.0f} min ({(primera - final) / primera * 100:+.1f} %)",
            ))
        filas += [
            ("Interferencias / cruces finales", f"{interf_finales} / {a.get('cruces_finales', '—')}"),
            ("Movimientos conservados", "Sí" if a.get("movimientos_conservados") else "No"),
            ("Grúas disponibles / utilizadas", f"{a.get('gruas_disponibles', '—')} / {a.get('gruas_utilizadas', '—')}"),
            ("Iteraciones ALNS", a.get("iteraciones_alns", 0)),
            ("Candidatas diferentes", a.get("candidatas_distintas", 0)),
            ("Operador ganador", a.get("operador_destruccion_ganador") or "—"),
            ("Tiempo total", "—" if tiempo_total is None else f"{tiempo_total:.1f} s"),
        ]
        cp = resumen_cp_local(a.get("estadisticas_cp_local"))
        filas.append(("CP-SAT local", cp if isinstance(cp, str) else f"{cp['ejecuciones']} ejecuciones, {cp['mejoran_incumbente']} mejoran"))
        st.table(pd.DataFrame(filas, columns=["Indicador", "Valor"]).astype(str).set_index("Indicador"))
        referencia = a.get("makespan_antes_alns_min")
        if referencia is None:
            referencia = primera
        if a.get("iteraciones_alns") and referencia is not None and final is not None and final >= referencia - 1.0:
            st.info(
                "El ALNS exploró alternativas factibles, pero no encontró una solución mejor dentro del tiempo disponible."
            )

def mostrar_analisis_interferencias(
    df_cronograma, config_gruas, mapa_colores, lista_ordenada, estribor_muelle,
    intervalo_minutos, range_y_start, range_y_end, config_zoom,
):
    st.divider()
    st.subheader("⚠️ Interferencias entre grúas")
    holgura = st.number_input(
        "Holgura de seguridad (min)", 0, 120, 15, 5, key="holgura_interferencias",
        help=(
            "Se alerta cuando dos grúas están en el mismo bay o en bays contiguos y sus tareas "
            "coinciden o quedan separadas por menos que esta holgura. Cubierta y bodega cuentan como el mismo bay."
        ),
    )
    firma_actual = hash((
        repr(sorted((g, repr(c)) for g, c in config_gruas.items())),
        df_cronograma[["Grúa", "Bay Real", "Acción", "Inicio", "Fin"]].to_json(),
    ))
    orden_gruas = sorted(orden_original_gruas(df_cronograma).items(), key=lambda par: par[1])
    st.caption(
        "Orden de grúas de proa a popa (según el bay de su primera tarea): "
        + " → ".join(grua for grua, _ in orden_gruas)
        + ". La propuesta nunca invierte este orden y mantiene al menos 8 bays visuales entre grúas que trabajan a la vez."
    )
    st.caption("Configura primero los turnos y rendimientos de cada grúa; luego presiona el botón para analizar.")
    with st.expander("Configuración de optimización"):
        metodo_opt = st.radio(
            "Método", list(METODOS_OPTIMIZACION), index=1, key="metodo_optimizacion", horizontal=True,
            help="Reparación rápida: sin ALNS. Híbrido: mejora local + ALNS + intensificación final. Intensivo: lo mismo con más tiempo y vecindad mayor.",
        )
        base_metodo = METODOS_OPTIMIZACION[metodo_opt]
        c1, c2 = st.columns(2)
        tiempo_opt = c1.selectbox(
            "Tiempo límite (s)", [10, 20, 30, 60], index=[10, 20, 30, 60].index(base_metodo["tiempo_limite"]),
            key=f"tiempo_opt_{metodo_opt}",
        )
        permitir_reasig = c1.checkbox("Permitir reasignación", True, key="opt_reasignacion")
        permitir_div = c2.checkbox("Permitir división", True, key="opt_division")
        usar_alns = c2.checkbox("Activar ALNS", base_metodo["activar_alns"], key=f"opt_alns_{metodo_opt}")
        usar_cp = c2.checkbox("Usar CP-SAT local", True, key="opt_cpsat_local")
        with st.expander("Configuración avanzada"):
            semilla_opt = st.number_input("Random seed", 0, 99999, 42, 1, key="opt_seed")
            modo_repro = st.checkbox(
                "Modo reproducible (1 worker)", False, key="opt_reproducible",
                help="Usa un solo hilo y la misma semilla; el resultado puede variar levemente si cambia la carga del equipo, porque el tiempo límite es real.",
            )
            usar_inferido = st.checkbox(
                "Usar orden físico inferido de la semilla", True, key="opt_orden_inferido",
                help="Orden de grúas de proa a popa que se mantiene fijo para evitar cruces. No fija la primera tarea ni el primer bay.",
            )
            orden_fisico, error_orden = None, None
            if not usar_inferido:
                nombres_gruas = [g for g, _ in orden_gruas]
                orden_fisico = st.multiselect(
                    "Orden de las grúas de proa a popa", nombres_gruas, default=nombres_gruas, key="opt_orden_manual",
                )
                if sorted(orden_fisico) != sorted(nombres_gruas):
                    error_orden = "Indique todas las grúas, cada una una sola vez."
                    st.error(error_orden)
    params_opt = {
        "tiempo_limite": tiempo_opt, "random_seed": int(semilla_opt), "activar_alns": usar_alns,
        "permitir_reasignacion": permitir_reasig, "permitir_division": permitir_div, "usar_cpsat_local": usar_cp,
        "porcentaje_vecindad": base_metodo["porcentaje_vecindad"], "activar_mejora_local": True,
        "activar_mejora_local_inicial": True, "activar_intensificacion_final": usar_alns,
        "modo_reproducible": modo_repro, "orden_fisico_gruas": orden_fisico if not usar_inferido else None,
    }
    if st.button("🔍 Analizar interferencias", type="primary", key="btn_analizar_interferencias"):
        resultado = {"firma": firma_actual, "holgura": holgura}
        resultado["interferencias"] = detectar_interferencias(df_cronograma, holgura)
        resultado["propuesta"] = None
        resultado["error"] = None
        if not resultado["interferencias"].empty and error_orden:
            resultado["error"] = error_orden
        elif not resultado["interferencias"].empty:
            try:
                with st.spinner(f"Optimizando la secuencia (puede tardar hasta ~{params_opt['tiempo_limite']} s)..."):
                    t_inicio_opt = time_module.monotonic()
                    resultado["propuesta"] = proponer_secuencia(
                        df_cronograma, config_gruas, calcular_intervalo_tarea, holgura, **params_opt
                    )
                    resultado["tiempo_total"] = time_module.monotonic() - t_inicio_opt
            except ValueError as e:
                resultado["error"] = str(e)
        st.session_state["analisis_interferencias"] = resultado

    resultado = st.session_state.get("analisis_interferencias")
    if resultado is None:
        return
    if resultado["firma"] != firma_actual:
        st.warning(
            "Los resultados están desactualizados: cambió la configuración de las grúas "
            "(turnos, rendimientos, inicio o nombres). Presiona «Analizar interferencias» para actualizarlos."
        )
    holgura = resultado["holgura"]
    interferencias = resultado["interferencias"]
    if interferencias.empty:
        st.success(f"No hay grúas en el mismo bay ni en bays contiguos dentro de la holgura de {holgura} min.")
        return

    st.warning(f"Se detectaron {len(interferencias)} posible(s) interferencia(s) entre grúas.")
    tabla_interferencias = interferencias.assign(
        Desde=interferencias["Desde"].dt.strftime("%d/%m %H:%M"),
        Hasta=interferencias["Hasta"].dt.strftime("%d/%m %H:%M"),
    )
    st.dataframe(tabla_interferencias, use_container_width=True, hide_index=True)

    st.subheader("🔁 Propuesta sugerida")
    if resultado["error"]:
        st.error(resultado["error"])
        return
    propuesta = resultado["propuesta"]

    restantes = len(detectar_interferencias(propuesta, holgura))
    comparacion = comparar_cronogramas(df_cronograma, propuesta)
    variacion_total = comparacion["Variación (min)"].iloc[-1]
    if restantes:
        st.warning(f"La propuesta aún tiene {restantes} interferencia(s) que no pudo resolver.")
    else:
        st.success("La propuesta no deja grúas en el mismo bay ni en bays contiguos dentro de la holgura.")
    if variacion_total and variacion_total > 0:
        st.info(f"Para evitar las interferencias, la operación completa termina {variacion_total:.0f} min después.")
    elif variacion_total is not None and variacion_total <= 0:
        st.info(f"La operación completa termina {abs(variacion_total):.0f} min antes o a la misma hora.")
    st.dataframe(comparacion, use_container_width=True, hide_index=True)
    if propuesta.attrs.get("metodo"):
        st.caption(f"Método elegido: {propuesta.attrs['metodo']}")
    mostrar_resumen_optimizacion(propuesta, resultado.get("tiempo_total"), len(interferencias), restantes)
    st.caption(
        "La propuesta puede mover y partir tareas entre grúas, respetando el orden original dentro de cada bay "
        "(descarga antes que embarque, cubierta antes que bodega) y los pontones en su grúa. "
        "Las grúas no se cruzan: conservan el orden de bays que tenían al inicio del plan original "
        "        y, mientras trabajan, quedan al menos 8 bays visuales separadas; una grúa que espera, se desplaza o ya terminó no restringe a las demás. "
                "La propuesta puede rehacer la secuencia desde cero y se elige, entre varios métodos "
                "(incluida una optimización CP-SAT que minimiza el término de la operación), la que termina antes. "
        "No considera el tipo de grúa (STS/MHC) ni restricciones de estiba."
    )

    inicio_propuesta = min(range_y_start, propuesta["Inicio"].min() - timedelta(minutes=30))
    fin_propuesta = max(range_y_end, propuesta["Fin"].max() + timedelta(minutes=30))
    figura_propuesta = construir_figura_gantt(
        propuesta, mapa_colores, lista_ordenada, estribor_muelle,
        intervalo_minutos, inicio_propuesta, fin_propuesta,
    )
    st.plotly_chart(figura_propuesta, use_container_width=True, config=config_zoom, key="gantt_propuesta")

    st.subheader("📋 Cola de trabajo por grúa (propuesta)")
    colas = construir_colas_trabajo(propuesta)
    for grua, cola in colas.items():
        with st.expander(f"{grua} · {len(cola)} tarea(s) · {int(cola['Movimientos'].sum())} movimientos"):
            st.dataframe(cola, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Descargar colas de trabajo (Excel)",
        data=exportar_colas_excel(colas),
        file_name="Colas_trabajo_propuesta.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

def render_simulador():
    st.title("🏗️ Simulador de Planificación Naviera - SVTI")
    
    with st.sidebar:
        st.header("1. Carga de Datos")
        uploaded_file = st.file_uploader("Subir PDF de Secuencia", type=["pdf", "xlsx", "csv"])

    config_gruas = {}
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.pdf'):
            df_raw, vessel_name = procesar_pdf(uploaded_file)
        else:
            df_raw, vessel_name = procesar_excel(uploaded_file)
        if vessel_name: st.info(f"**Buque:** {vessel_name}")

        gmph_default = 22.0
        excel_path = "Consolidado Naves 2025.xlsx"
        
        try:
            if (not vessel_name) and os.path.exists(excel_path):
                df_naves = pd.read_excel(excel_path, sheet_name="Naves")
                with pdfplumber.open(uploaded_file) as pdf_tmp:
                    first_text = pdf_tmp.pages[0].extract_text() or ""
                for candidate in df_naves['Nave'].astype(str).unique():
                    if candidate and candidate.strip().lower() in first_text.lower():
                        vessel_name = candidate.strip(); break
            if vessel_name and os.path.exists(excel_path):
                df_naves = pd.read_excel(excel_path, sheet_name="Naves")
                mask = df_naves['Nave'].astype(str).str.contains(vessel_name, case=False, na=False)
                vals = pd.to_numeric(df_naves.loc[mask, 'Rend.Cuadrilla Neto (ROR)'], errors='coerce')
                if not vals.dropna().empty: gmph_default = float(vals.mean())
        except: pass

        if df_raw is not None and 'Mov' in df_raw.columns:
            st.sidebar.divider()
            lista_gruas = sorted(df_raw['Grúa'].unique())
            movimientos_por_grua = estimar_movimientos_por_grua(df_raw)
            colores_config_gruas = obtener_colores_gruas(lista_gruas)
            renombres = {}
            for grua in lista_gruas:
                nombre_sel = st.sidebar.selectbox(
                    f"Nombre de la grúa {grua}", [grua, *NOMBRES_GRUA], key=f"n_{grua}",
                    help="STS 1 y STS 2 son grúas STS; G5, G6, G9 y G10 son MHC."
                )
                renombres[grua] = nombre_sel
                mostrar_leyenda_grua(
                    nombre_sel, colores_config_gruas[grua], destino=st.sidebar
                )
                with st.sidebar.expander(f"⚙️ {nombre_sel}", expanded=True):
                    c1, c2 = st.columns(2)
                    d = c1.date_input("Fecha", datetime.today(), key=f"d_{grua}")
                    t = c2.time_input("Hora", time(8,0), key=f"t_{grua}")
                    inicio_grua = datetime.combine(d, t)
                    gmph_input, gmph_schedule = configurar_gmph_por_grua(
                        grua, gmph_default, 1.0, 200.0, f"g_{grua}", inicio_grua,
                        movimientos_por_grua.get(grua, 0)
                    )
                    config_gruas[grua] = {
                        "inicio": inicio_grua,
                        "gmph": gmph_input,
                        "gmph_schedule": gmph_schedule,
                    }

            if len(set(renombres.values())) < len(renombres):
                st.error("Hay grúas con el mismo nombre; asigna un nombre distinto a cada una.")
                st.stop()
            colores_por_nombre_grua = {
                renombres[grua]: colores_config_gruas[grua]
                for grua in lista_gruas
            }
            df_raw = df_raw.assign(**{"Grúa": df_raw["Grúa"].map(renombres)})
            config_gruas = {renombres[g]: c for g, c in config_gruas.items()}
            lista_gruas = sorted(renombres.values())

            try:
                df_cronograma = calcular_cronograma(df_raw, config_gruas)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            
            if not df_cronograma.empty:
                fin_max = df_cronograma['Fin'].max()
                inicio_min = df_cronograma['Inicio'].min()
                total_movs = df_cronograma['Acción'].str.extract(r'\((\d+)\)')[0].astype(float).sum()
                duracion_total = (fin_max - inicio_min).total_seconds() / 3600
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Fin Operación", fin_max.strftime("%d/%m %H:%M"))
                c2.metric("Duración", f"{duracion_total:.1f} Hrs")
                c3.metric("Movimientos", int(total_movs))
                
                st.divider()
                col_titulo, col_intervalo, col_orden = st.columns([2, 1, 1.4])
                with col_titulo: st.subheader("Línea temporal de Working List")
                with col_intervalo: intervalo_minutos = st.number_input("Grid Tiempo (min)", 10, 120, 60, 15)
                with col_orden:
                    orden_pref = st.selectbox(
                        "Orden de Bays",
                        ["Estribor muelle (mayor a menor)", "Babor muelle (menor a mayor)"], index=0,
                    )
                with st.expander("Ubicación de puente de mando y chimenea (opcional)"):
                    bay_ini, bay_fin = int(df_cronograma["Bay Visual"].min()), int(df_cronograma["Bay Visual"].max())
                    bays_ui = list(range(max(2, bay_ini - 16), bay_fin + 24 + 1, 4))
                    opciones_bay = {"No indicar": None}
                    for a, b in zip(bays_ui, bays_ui[1:]):
                        opciones_bay[f"Entre bay {a:02d} y {b:02d}"] = (a, b)
                    cp, cc = st.columns(2)
                    ubic_puente = opciones_bay[cp.selectbox("¿Entre qué bays está el puente de mando?", list(opciones_bay))]
                    ubic_chimenea = opciones_bay[cc.selectbox("¿Entre qué bays está la chimenea?", list(opciones_bay))]
                st.caption("En BAY, D = cubierta y H = bodega. Achurado diagonal = cubierta; las barras sólidas son bodega o zona no identificada. Pasa el cursor para ver la zona detectada.")
                sin_zona = (df_cronograma['Zona'] == "Sin identificar").sum()
                if sin_zona:
                    st.warning(
                        f"{sin_zona} tarea(s) no tienen una zona identificable en el texto extraído. "
                        "Se muestran sólidas; confirma que el PDF incluya una etiqueta como DECK/CUBIERTA o HOLD/BODEGA."
                    )
                mapa_colores = { "Pontón": "#7F8C8D", "Otro": "#BDC3C7" }
                for nombre_g, (color_desc, color_emb) in colores_por_nombre_grua.items():
                    mapa_colores[f"{nombre_g} - Descarga"] = color_desc
                    mapa_colores[f"{nombre_g} - Embarque"] = color_emb

                min_bay = int(df_cronograma["Bay Visual"].min())
                max_bay = int(df_cronograma["Bay Visual"].max())
                for ubic in (ubic_puente, ubic_chimenea):
                    if ubic:
                        min_bay, max_bay = min(min_bay, ubic[0]), max(max_bay, ubic[1])
                bays_completas = [f"{b:02d}" for b in range(min_bay, max_bay + 4, 4)]
                estribor_muelle = orden_pref.startswith("Estribor")
                for ubic, slot in ((ubic_puente, SLOT_PUENTE), (ubic_chimenea, SLOT_CHIMENEA)):
                    if ubic:
                        bays_completas.insert(bays_completas.index(f"{ubic[0]:02d}") + 1, slot)
                lista_ordenada = bays_completas[::-1] if estribor_muelle else bays_completas

                range_y_start = inicio_min - timedelta(minutes=30)
                range_y_end = fin_max + timedelta(minutes=30)

                fig = construir_figura_gantt(
                    df_cronograma, mapa_colores, lista_ordenada, estribor_muelle,
                    intervalo_minutos, range_y_start, range_y_end,
                )
                config_zoom = {'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d', 'lasso2d']}
                st.plotly_chart(fig, use_container_width=True, config=config_zoom)

                mostrar_analisis_interferencias(
                    df_cronograma, config_gruas, mapa_colores, lista_ordenada, estribor_muelle,
                    intervalo_minutos, range_y_start, range_y_end, config_zoom,
                )

                st.subheader("TENTATIVE WORKING PROGRAM")
                tabla_programa, resumen_programa = construir_programa_trabajo(df_cronograma)
                nombre_buque = vessel_name or os.path.splitext(uploaded_file.name)[0]
                clave_programa = f"{uploaded_file.name}_{abs(hash(uploaded_file.getvalue()))}"
                total_programa = int(tabla_programa["Total"].iloc[-1])
                horas_op = max(duracion_total, 1e-9)
                horas_grua = (pd.to_datetime(df_cronograma["Fin"]) - pd.to_datetime(df_cronograma["Inicio"])).dt.total_seconds().sum() / 3600
                encabezado_programa = pd.DataFrame([{
                    "Vessel name": nombre_buque,
                    "Discharge": resumen_programa["Discharge"],
                    "Loading": resumen_programa["Load"],
                    "Restow": resumen_programa["Restow"],
                    "Pontoons": resumen_programa["Pontoons"],
                    "Rendimiento (mov/h)": f"{total_programa / horas_op:.0f}",
                    "Crane density": f"{horas_grua / horas_op:.2f}",
                }])
                mostrar_tabla_ejecutiva(encabezado_programa)

                mostrar_tabla_ejecutiva(tabla_programa, resaltar_ultima=True)

                tabla_turnos = construir_tabla_turnos_trabajados(df_cronograma)
                cantidad_turnos = len(tabla_turnos)
                col_turnos, col_gangs = st.columns([1, 3])
                with col_turnos:
                    mostrar_tabla_ejecutiva(pd.DataFrame({"Working Shifts": [cantidad_turnos]}))
                with col_gangs:
                    mostrar_tabla_ejecutiva(tabla_turnos)

                st.markdown(
                    f"<div class='etc-banner'><span>ESTIMATED TIME OF COMPLETION</span>"
                    f"<b>ETC {fin_max:%d/%m %H:%M} HRS TBC WP.</b></div>",
                    unsafe_allow_html=True,
                )

                st.sidebar.divider()
                st.sidebar.download_button(
                    "📄 Descargar reporte PDF",
                    data=generar_reporte_pdf(df_cronograma, tabla_programa, nombre_buque),
                    file_name=f"Reporte_Ejecutivo_{nombre_buque}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else: st.warning("No hay datos calculados.")
        else: st.error("Error al procesar PDF.")
    else: st.info("Sube el PDF en el panel lateral.")



def render_operacion():
    st.title("📊 Visualización de Operación (SVTI)")
    
    with st.sidebar:
        st.header("1. Archivos BAPLIE")
        f_arr = st.file_uploader("1. Llegada (Arrival)", type=['txt', 'edi'], key="f_arr")
        f_dep = st.file_uploader("2. Salida (Loading)", type=['txt', 'edi'], key="f_dep")
        
        st.divider()
        st.header("2. Opciones")
        activar_one_sphere = st.checkbox("Forzar Regla 'One Sphere'", value=True, help="Pinta de rojo las zonas estructurales restringidas.")

    if f_arr and f_dep:
        df_arr_raw, v_name_arr = parse_baplie_advanced(f_arr.getvalue())
        df_dep_raw, v_name_dep = parse_baplie_advanced(f_dep.getvalue())
        
        df_op = generar_dataframe_operativo(df_arr_raw, df_dep_raw, puerto_obj='CLSVE')
        
        c1, c2, c3 = st.columns(3)
        n_desc = len(df_op[df_op['Tipo_Op']=='Descarga'])
        n_emb = len(df_op[df_op['Tipo_Op']=='Embarque'])
        n_ree = len(df_op[df_op['Tipo_Op']=='Reestiba'])
        
        c1.metric("📉 A Descargar", n_desc)
        c2.metric("📈 A Embarcar", n_emb)
        c3.metric("🔄 Reestibas", n_ree*2)
        
        ver_solo_movs = st.checkbox("Ocultar Tránsito (Ver solo lo que se mueve)", value=False)
        if ver_solo_movs:
            df_viz = df_op[df_op['Tipo_Op'] != 'Tránsito'].copy()
        else:
            df_viz = df_op.copy()
            
        nombre_buque_final = v_name_arr if v_name_arr != "Unknown" else "ONE SPHERE"
        if activar_one_sphere:
            if "ONE SPHERE" not in nombre_buque_final.upper():
                nombre_buque_final = "ONE SPHERE (Forzado)" 
            
            df_viz = aplicar_logica_one_sphere(df_viz, nombre_buque_final)
            
            n_rojos = len(df_viz[df_viz['Color_Visual'] == '⚠️ Restricción OneSphere'])
            if n_rojos > 0:
                st.warning(f"⚠️ ¡Atención! Se detectaron **{n_rojos} contenedores** en zonas restringidas del One Sphere.")

        fig = create_3d_plot_operacion(df_viz, nombre_buque_final)
        st.plotly_chart(fig, use_container_width=True)

        # --- NUEVA LÓGICA: TABLA RESUMEN POR BAY ---
        st.divider()
        st.subheader("📋 Resumen de Restricciones por Bay")

        if activar_one_sphere and 'Color_Visual' in df_viz.columns:
            # Crear columna auxiliar binaria para sumar fácil
            df_viz['Es_Restringido'] = (df_viz['Color_Visual'] == '⚠️ Restricción OneSphere').astype(int)
            
            # Agrupar por Bay y Action (Tipo_Op)
            resumen = df_viz.groupby(['Bay', 'Tipo_Op']).agg(
                TOTAL_MOV=('ID', 'count'),
                MOV_RESTRINGIDOS=('Es_Restringido', 'sum')
            ).reset_index()

            # Renombrar columnas para la visualización final
            resumen = resumen.rename(columns={'Bay': 'BAY', 'Tipo_Op': 'ACTION', 'TOTAL_MOV': 'TOTAL MOV', 'MOV_RESTRINGIDOS': 'MOV RESTRINGIDOS'})

            # Filtrar para mostrar solo aquellos que tienen restricciones > 0 (los más críticos)
            # O mostrar todos ordenados. Mostraremos ordenados por peligrosidad.
            resumen_restringidos = resumen[resumen['MOV RESTRINGIDOS'] > 0].sort_values(
                by=['MOV RESTRINGIDOS', 'BAY'], ascending=[False, True]
            )

            if not resumen_restringidos.empty:
                st.dataframe(
                    resumen_restringidos,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "BAY": st.column_config.NumberColumn("Bay", format="%d"),
                        "MOV RESTRINGIDOS": st.column_config.ProgressColumn(
                            "Restringidos",
                            format="%d",
                            min_value=0,
                            max_value=int(resumen['TOTAL MOV'].max()),
                            help="Cantidad de movimientos en zona roja"
                        )
                    }
                )
            else:
                st.success("✅ No se detectaron movimientos en zonas restringidas.")
        
    else:
        st.info("👈 Por favor carga ambos archivos (Llegada y Salida) en la barra lateral para calcular la operación.")

# --- 7. CONTROLADOR DE NAVEGACIÓN ---

if st.session_state.page != 'home':
    if st.sidebar.button("🏠 Inicio"): navigate_to('home')

if st.session_state.page == 'home': render_menu()
elif st.session_state.page == 'simulador': render_simulador()
elif st.session_state.page == 'planificador_sts': render_planificador_sts()
elif st.session_state.page == 'operacion': render_operacion()
