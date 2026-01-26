import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pdfplumber
import re
from datetime import datetime, timedelta, time
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulador Naviero SVTI", layout="wide")
st.title("🚢 Simulador de Planificación Naviera - SVTI")

# --- CONFIGURACIÓN DE DESCANSOS (GLOBAL) ---
BREAKS_CONFIG = [
    (2, 20, 3, 0, "COLACIÓN T3"),
    (6, 30, 8, 0, "CAMBIO TURNO"),
    (11, 20, 12, 0, "COLACIÓN T1"),
    (18, 50, 19, 30, "COLACIÓN T2")
]

# --- FUNCIONES AUXILIARES ---

def obtener_bay_visual(bay_real):
    try:
        match = re.search(r"(\d+)", str(bay_real))
        if match:
            b = float(match.group(1))
            k = round((b - 2) / 4)
            bay_visual = 2 + (4 * k)
            return int(max(2, bay_visual))
        return 0
    except:
        return 0

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
                    if m:
                        vessel_name = m.group(1).strip()
                
                grua_match = re.search(r"Crane No\.:\s*([\w\d]+)", texto)
                nombre_grua = grua_match.group(1) if grua_match else f"Grúa_{i+1}"

                tabla = pagina.extract_table({
                    "vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 4
                })
                
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
    except Exception as e:
        return None, None
    
    if datos_totales:
        return pd.concat(datos_totales, ignore_index=True), vessel_name
    return None, vessel_name

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
            if salto_hasta and salto_hasta > siguiente_corte:
                cursor = salto_hasta
    return cursor

def extraer_numero(texto):
    """Extrae el primer número flotante encontrado en un texto sucio."""
    match = re.search(r'(\d+)', str(texto))
    return float(match.group(1)) if match else None

def calcular_cronograma(df, config_gruas):
    cronograma = []
    
    # Aseguramos que 'Seq' sea numérico para ordenar correctamente los acumulados
    if 'Seq' in df.columns:
        df['Seq'] = pd.to_numeric(df['Seq'], errors='coerce')
        df = df.sort_values(by=['Grúa', 'Seq'])

    for nombre_grua, grupo_df in df.groupby("Grúa"):
        
        if nombre_grua in config_gruas:
            params = config_gruas[nombre_grua]
            tiempo_actual = params['inicio']
            gmph = params.get('gmph', 20)
        else:
            tiempo_actual = datetime.now()
            gmph = 20
        
        # Variable para llevar el conteo del acumulado anterior
        last_total_m = 0
        
        for index, row in grupo_df.iterrows():
            # 1. Recuperación de textos crudos para identificación
            raw_action = str(row.get('Action Name', '')).upper()
            raw_bay = str(row.get('Bay', '')).upper()
            
            # --- LÓGICA MATEMÁTICA (DELTA TOTAL) ---
            # Intentamos leer el Total M actual
            current_total_m = extraer_numero(row.get('Total M', ''))
            
            movimientos = 0
            
            if current_total_m is not None:
                # Calculamos por diferencia
                delta = current_total_m - last_total_m
                
                if delta >= 0:
                    # Caso Ideal: La resta es lógica
                    movimientos = delta
                    last_total_m = current_total_m
                else:
                    # Caso Raro: El total bajó (error de OCR en Total M).
                    # Usamos la columna 'Mov' como respaldo.
                    mov_col = extraer_numero(row.get('Mov', ''))
                    movimientos = mov_col if mov_col is not None else 0
                    last_total_m += movimientos # Corregimos nuestro acumulado interno
            else:
                # Caso Fallo Total: No se pudo leer 'Total M'.
                # Usamos la columna 'Mov'.
                mov_col = extraer_numero(row.get('Mov', ''))
                movimientos = mov_col if mov_col is not None else 0
                last_total_m += movimientos
            
            # ----------------------------------------
            
            # Lógica de categorías y texto (Mantenemos tu corrección anterior de filas sucias)
            texto_fila_completo = f"{raw_bay} {raw_action}"
            
            bay_visual_num = obtener_bay_visual(raw_bay)
            if bay_visual_num == 0 and "PONTOON" not in texto_fila_completo: 
                continue

            bay_visual_str = f"{bay_visual_num:02d}" 
            
            duracion_minutos = 0
            categoria_visual = "Otro"
            
            if "PONTOON" in texto_fila_completo or "OPEN" in texto_fila_completo or "CLOSE" in texto_fila_completo:
                duracion_minutos = 15 
                categoria_visual = "Pontón"
                if bay_visual_num == 0: bay_visual_str = "General" 

            elif movimientos > 0:
                duracion_minutos = (movimientos / gmph) * 60
                
                # Palabras clave ampliadas según tus hallazgos
                kws_descarga = ["DISCH", "DSCH", "BLISCH", "DISH", "DDI", "DHI", "SCH", "DCH"]
                kws_embarque = ["LOAD", "HOAD", "DOAD", "LOD", "LOA", "O AD", " HL", " DL", "AD TWI"]
                
                es_descarga = any(kw in texto_fila_completo for kw in kws_descarga)
                es_embarque = any(kw in texto_fila_completo for kw in kws_embarque)
                
                if es_descarga: 
                    categoria_visual = f"{nombre_grua} - Descarga"
                elif es_embarque: 
                    categoria_visual = f"{nombre_grua} - Embarque"

            if duracion_minutos > 0:
                tiempo_fin_real = calcular_fin_con_descansos(tiempo_actual, duracion_minutos)
                
                accion_display = raw_action.replace('\n', ' ')
                if not accion_display: accion_display = "Detectado por contexto"

                cronograma.append({
                    "Grúa": nombre_grua,
                    "Bay Real": raw_bay, 
                    "Bay Visual": bay_visual_str,
                    "Acción": f"{accion_display} ({int(movimientos)})",
                    "Inicio": tiempo_actual,
                    "Fin": tiempo_fin_real,
                    "Duración Neta": round(duracion_minutos, 1),
                    "Categoría Visual": categoria_visual
                })
                tiempo_actual = tiempo_fin_real

    return pd.DataFrame(cronograma)

# --- INTERFAZ PRINCIPAL ---

with st.sidebar:
    st.header("1. Carga de Datos")
    uploaded_file = st.file_uploader("Subir PDF de Secuencia", type="pdf")

config_gruas = {}

if uploaded_file is not None:
    df_raw, vessel_name = procesar_pdf(uploaded_file)

    if vessel_name:
        st.info(f"**Buque:** {vessel_name}")

    gmph_default = 25.0
    excel_path = "Consolidado Naves 2025.xlsx"
    try:
        if (not vessel_name) and os.path.exists(excel_path):
            try:
                df_naves = pd.read_excel(excel_path, sheet_name="Naves")
            except Exception:
                df_naves = None

            if df_naves is not None and 'Nave' in df_naves.columns:
                try:
                    with pdfplumber.open(uploaded_file) as pdf_tmp:
                        first_text = pdf_tmp.pages[0].extract_text() or ""
                except Exception:
                    first_text = ""

                for candidate in df_naves['Nave'].astype(str).unique():
                    if candidate and candidate.strip():
                        if candidate.lower().strip() in first_text.lower():
                            vessel_name = candidate.strip()
                            break

        if vessel_name and os.path.exists(excel_path):
            df_naves = pd.read_excel(excel_path, sheet_name="Naves")
            if 'Nave' in df_naves.columns and 'Rend.Cuadrilla Neto (ROR)' in df_naves.columns:
                mask = df_naves['Nave'].astype(str).str.contains(vessel_name, case=False, na=False)
                vals = pd.to_numeric(df_naves.loc[mask, 'Rend.Cuadrilla Neto (ROR)'], errors='coerce')
                if not vals.dropna().empty:
                    gmph_default = float(vals.mean())
    except Exception:
        gmph_default = 25.0

    if df_raw is not None and 'Mov' in df_raw.columns:
        
        st.sidebar.divider()
        st.sidebar.header("2. Configuración por Grúa")
        
        lista_gruas = sorted(df_raw['Grúa'].unique())
        
        for grua in lista_gruas:
            with st.sidebar.expander(f"⚙️ {grua}", expanded=True):
                col1, col2 = st.columns(2)
                def_date = datetime.today()
                def_time = datetime.strptime("08:00", "%H:%M").time()
                f_inicio = col1.date_input(f"Fecha Ini", value=def_date, key=f"d_{grua}")
                h_inicio = col2.time_input(f"Hora Ini", value=def_time, key=f"t_{grua}")
                gmph_input = st.number_input(
                    f"GMPH",
                    min_value=1.0,
                    max_value=200.0,
                    value=float(round(gmph_default, 1)),
                    step=0.1,
                    key=f"g_{grua}"
                )
                
                inicio_dt = datetime.combine(f_inicio, h_inicio)
                config_gruas[grua] = {"inicio": inicio_dt, "gmph": gmph_input}

        df_cronograma = calcular_cronograma(df_raw, config_gruas)
        
        if not df_cronograma.empty:
            fin_max = df_cronograma['Fin'].max()
            inicio_min = df_cronograma['Inicio'].min()
            raw_movs = df_raw['Mov'].astype(str).str.split().str[0]
            # Movimientos reales usados en el gráfico
            total_movs = (
                df_cronograma['Acción']
                .str.extract(r'\((\d+)\)')[0]
                .astype(float)
                .sum()
)

            duracion_total = (fin_max - inicio_min).total_seconds() / 3600

            def contar_turnos(tramo_inicio, tramo_fin):
                turno_defs = [ (8,0,15,30), (15,30,23,0), (23,0,6,30) ]
                cuenta = 0
                dia_inicio = tramo_inicio.date() - timedelta(days=1)
                dia_fin = tramo_fin.date() + timedelta(days=1)
                dia = dia_inicio
                while dia <= dia_fin:
                    for h_ini, m_ini, h_fin, m_fin in turno_defs:
                        t_ini = datetime.combine(dia, time(h_ini, m_ini))
                        t_fin = datetime.combine(dia, time(h_fin, m_fin))
                        if t_fin <= t_ini:
                            t_fin += timedelta(days=1)
                        inicio_solap = max(tramo_inicio, t_ini)
                        fin_solap = min(tramo_fin, t_fin)
                        if inicio_solap < fin_solap:
                            cuenta += 1
                    dia += timedelta(days=1)
                return cuenta

            num_turnos = contar_turnos(inicio_min, fin_max)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fin de Operación", fin_max.strftime("%d/%m %H:%M"))
            c2.metric("Duración Total (inc. Pausas)", f"{duracion_total:.1f} Hrs")
            c3.metric("Movimientos Totales", int(total_movs))
            c4.metric("Turnos Trabajados", int(num_turnos))
            
            st.divider()
            
            col_titulo, col_intervalo, col_orden = st.columns([2, 1, 1])
            with col_titulo:
                st.subheader("Línea temporal de Working List")
            with col_intervalo:
                intervalo_minutos = st.number_input("Grid Tiempo (min)", 10, 120, 60, 15)
            with col_orden:
                orden_pref = st.selectbox("Orden de Bays", ["Mayor a Menor", "Menor a Mayor"], index=0)
            
            pares_colores = [
                ("#D35400", "#F39C12"), 
                ("#2980B9", "#5DADE2"), 
                ("#27AE60", "#58D68D"), 
                ("#8E44AD", "#AF7AC5"), 
                ("#C0392B", "#E74C3C"), 
            ]
            mapa_colores = { "Pontón": "#7F8C8D", "Otro": "#BDC3C7" }
            gruas_unicas = sorted(df_cronograma['Grúa'].unique())
            for i, nombre_g in enumerate(gruas_unicas):
                color_desc, color_emb = pares_colores[i % len(pares_colores)]
                mapa_colores[f"{nombre_g} - Descarga"] = color_desc
                mapa_colores[f"{nombre_g} - Embarque"] = color_emb

            min_bay = int(df_cronograma["Bay Visual"].min())
            max_bay = int(df_cronograma["Bay Visual"].max())
            bays_completas = [f"{b:02d}" for b in range(min_bay, max_bay + 4, 4)]
            
            if "Mayor a Menor" in orden_pref:
                lista_ordenada = sorted(bays_completas, reverse=True) 
            else:
                lista_ordenada = bays_completas

            min_inicio = df_cronograma['Inicio'].min()
            max_fin = df_cronograma['Fin'].max()
            range_y_start = min_inicio - timedelta(minutes=30)
            range_y_end = max_fin + timedelta(minutes=30)

            fig = go.Figure()

            for categoria, color in mapa_colores.items():
                df_cat = df_cronograma[df_cronograma['Categoría Visual'] == categoria]
                if df_cat.empty: continue
                
                duracion_ms = (df_cat['Fin'] - df_cat['Inicio']).dt.total_seconds() * 1000

                fig.add_trace(go.Bar(
                    name=categoria,
                    x=df_cat['Bay Visual'],   
                    y=duracion_ms,
                    base=df_cat['Inicio'],
                    orientation='v',
                    marker_color=color,
                    marker_line_color='black',
                    marker_line_width=1,
                    opacity=1.0, 
                    width=0.95, 
                    customdata=df_cat[['Grúa', 'Acción', 'Inicio', 'Fin', 'Duración Neta']],
                    hovertemplate="<b>Bay %{x}</b><br>%{customdata[1]}<br>In: %{base|%H:%M}<br>Out: %{customdata[3]|%H:%M}<br>Trabajo Neto: %{customdata[4]} min<extra></extra>"
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
                        y0=t_inicio, 
                        y1=t_fin,
                        fillcolor="rgba(50, 50, 50, 1.0)", 
                        line_width=0,
                        layer="above", 
                        annotation_text=label,
                        annotation_position="top left",
                        annotation_font_color="white",
                        annotation_font_size=10
                    )

            fig.update_layout(
                height=900,
                legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="right", x=1, font=dict(color='white', size=12), bgcolor='rgba(0,0,0,0)'),
                dragmode='pan', hovermode='closest',
                barmode='overlay', bargap=0.02,
                plot_bgcolor='#1b263b', paper_bgcolor='#1b263b', font=dict(color='white'),

                yaxis=dict(
                    type='date',
                    title=dict(text="Hora", font=dict(color='white', size=16, weight="bold")),
                    tickformat="%H:%M", tickfont=dict(color='white', size=12),
                    dtick=intervalo_minutos * 60 * 1000,
                    autorange="reversed", range=[range_y_start, range_y_end],
                    fixedrange=False, showgrid=True, gridcolor='#3a506b', gridwidth=1,
                    layer="below traces"
                ),
                xaxis=dict(
                    title=dict(text="Bay (Serie Par)", font=dict(color='white', size=16, weight="bold")),
                    type='category', 
                    categoryorder='array', 
                    categoryarray=lista_ordenada,
                    side="top", tickfont=dict(color='white', size=14, weight="bold"),
                    showgrid=True, gridcolor='#000000', ticks="outside", tickcolor='#000000',
                    fixedrange=True 
                ),
                margin=dict(l=60, r=20, t=120, b=20),
            )

            configuracion_zoom = {
                'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False,
                'modeBarButtonsToRemove': ['select2d', 'lasso2d']
            }

            st.plotly_chart(fig, use_container_width=True, config=configuracion_zoom)
            
            with st.expander("Ver detalle de datos"):
                df_mostrar = df_cronograma.copy()
                df_mostrar['Inicio'] = df_mostrar['Inicio'].dt.strftime('%d/%m/%Y %H:%M')
                df_mostrar['Fin'] = df_mostrar['Fin'].dt.strftime('%d/%m/%Y %H:%M')
                st.dataframe(df_mostrar)
        else:
            st.warning("No hay datos calculados.")
    else:
        st.error("Error al procesar el PDF o falta columna 'Mov'.")
else:
    st.info("Sube el PDF para visualizar.")
