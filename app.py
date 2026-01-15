import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pdfplumber
import re
from datetime import datetime, timedelta, time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulador Naviero SVTI", layout="wide")
st.title("🚢 Simulador de Planificación Naviera - SVTI")

# --- CONFIGURACIÓN DE DESCANSOS (GLOBAL) ---
# Formato: (Hora_Inicio, Minuto_Inicio, Hora_Fin, Minuto_Fin, Etiqueta)
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
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i, pagina in enumerate(pdf.pages):
                texto = pagina.extract_text()
                grua_match = re.search(r"Crane No\.:\s*(\w+)", texto)
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
        return None
    
    if datos_totales:
        return pd.concat(datos_totales, ignore_index=True)
    return None

# --- NUEVA LÓGICA: CALCULAR FIN SALTANDO DESCANSOS ---
def calcular_fin_con_descansos(inicio_dt, duracion_minutos):
    """
    Avanza el reloj consumiendo 'duracion_minutos'. Si choca con un break,
    pausa el consumo y salta al final del break.
    """
    cursor = inicio_dt
    minutos_pendientes = duracion_minutos
    
    # Loop para avanzar mientras quede trabajo pendiente
    while minutos_pendientes > 0.001:
        
        # 1. Identificar el próximo break relevante para el día del cursor
        dia_cursor = cursor.date()
        siguiente_corte = None
        salto_hasta = None
        
        # Revisamos los breaks configurados
        for h_ini, m_ini, h_fin, m_fin, _ in BREAKS_CONFIG:
            # Crear timestamps para el break en el día actual del cursor
            b_inicio = datetime.combine(dia_cursor, time(h_ini, m_ini))
            b_fin = datetime.combine(dia_cursor, time(h_fin, m_fin))
            
            # Ajuste por si el break cruza medianoche (aunque en tu config no pasa, es buena práctica)
            if b_fin < b_inicio: 
                b_fin += timedelta(days=1)

            # CASO A: Estamos DENTRO de un break -> Saltar inmediatamente al final
            if b_inicio <= cursor < b_fin:
                cursor = b_fin
                siguiente_corte = None # Ya saltamos, reiniciamos búsqueda en siguiente iteración
                break 
            
            # CASO B: El break está en el futuro
            if b_inicio > cursor:
                # Nos quedamos con el break más cercano
                if siguiente_corte is None or b_inicio < siguiente_corte:
                    siguiente_corte = b_inicio
                    salto_hasta = b_fin

        # Si no hay más breaks hoy, mirar si hay mañana (solo si la tarea es muy larga)
        if siguiente_corte is None:
             # Límite técnico: fin del día actual + un poco
             siguiente_corte = datetime.combine(dia_cursor + timedelta(days=1), time(0,0))
             salto_hasta = siguiente_corte # Solo avanzamos hasta medianoche y reevaluamos
        
        # 2. Calcular cuánto podemos trabajar antes del próximo corte
        tiempo_hasta_corte = (siguiente_corte - cursor).total_seconds() / 60
        
        if tiempo_hasta_corte >= minutos_pendientes:
            # Alcanzamos a terminar antes del break
            cursor += timedelta(minutes=minutos_pendientes)
            minutos_pendientes = 0
        else:
            # Chocamos con el break
            # Trabajamos lo que se pudo
            cursor = siguiente_corte
            minutos_pendientes -= tiempo_hasta_corte
            
            # Si el corte era un break real, saltamos su duración
            if salto_hasta and salto_hasta > siguiente_corte:
                cursor = salto_hasta
                
    return cursor

def calcular_cronograma(df, config_gruas):
    cronograma = []
    
    if 'Seq' in df.columns:
        df['Seq'] = pd.to_numeric(df['Seq'], errors='coerce')
        df = df.sort_values(by=['Grúa', 'Seq'])

    for nombre_grua, grupo_df in df.groupby("Grúa"):
        
        if nombre_grua in config_gruas:
            params = config_gruas[nombre_grua]
            tiempo_actual = params['inicio']
            mph = params['mph']
        else:
            tiempo_actual = datetime.now()
            mph = 20

        es_grua_1 = "1" in str(nombre_grua)
        
        for index, row in grupo_df.iterrows():
            mov_val = str(row.get('Mov', 0)).split()[0]
            try: movimientos = float(mov_val)
            except: movimientos = 0
                
            accion = str(row.get('Action Name', '')).upper()
            bay_real_texto = str(row.get('Bay', ''))
            
            bay_visual_num = obtener_bay_visual(bay_real_texto)
            if bay_visual_num == 0 and "PONTOON" not in accion: 
                continue

            bay_visual_str = f"{bay_visual_num:02d}" 
            
            duracion_minutos = 0
            categoria_visual = "Otro"
            
            if "PONTOON" in accion or "OPEN" in accion or "CLOSE" in accion:
                duracion_minutos = 15 
                categoria_visual = "Pontón"
                if bay_visual_num == 0: bay_visual_str = "General" 

            elif movimientos > 0:
                duracion_minutos = (movimientos / mph) * 60
                
                if "DISCH" in accion: 
                    categoria_visual = "STS1 - Descarga" if es_grua_1 else "STS2 - Descarga"
                elif "LOAD" in accion: 
                    categoria_visual = "STS1 - Embarque" if es_grua_1 else "STS2 - Embarque"

            if duracion_minutos > 0:
                # --- APLICACIÓN DE LA LÓGICA DE DESCANSOS ---
                # Calculamos el fin real considerando las pausas
                tiempo_fin_real = calcular_fin_con_descansos(tiempo_actual, duracion_minutos)
                
                # Para la visualización, la duración bruta es la diferencia total
                duracion_visual_ms = (tiempo_fin_real - tiempo_actual).total_seconds() * 1000
                
                cronograma.append({
                    "Grúa": nombre_grua,
                    "Bay Real": bay_real_texto, 
                    "Bay Visual": bay_visual_str,
                    "Acción": f"{accion} ({int(movimientos)})",
                    "Inicio": tiempo_actual,
                    "Fin": tiempo_fin_real,
                    "Duración Neta": round(duracion_minutos, 1),
                    "Categoría Visual": categoria_visual
                })
                # El siguiente movimiento parte donde terminó este realmente
                tiempo_actual = tiempo_fin_real

    return pd.DataFrame(cronograma)

# --- INTERFAZ PRINCIPAL ---

with st.sidebar:
    st.header("1. Carga de Datos")
    uploaded_file = st.file_uploader("Subir PDF de Secuencia", type="pdf")

config_gruas = {}

if uploaded_file is not None:
    df_raw = procesar_pdf(uploaded_file)
    
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
                mph_input = st.number_input(f"MPH", min_value=10, max_value=60, value=25, key=f"m_{grua}")
                
                inicio_dt = datetime.combine(f_inicio, h_inicio)
                config_gruas[grua] = {"inicio": inicio_dt, "mph": mph_input}

        df_cronograma = calcular_cronograma(df_raw, config_gruas)
        
        if not df_cronograma.empty:
            fin_max = df_cronograma['Fin'].max()
            inicio_min = df_cronograma['Inicio'].min()
            raw_movs = df_raw['Mov'].astype(str).str.split().str[0]
            total_movs = pd.to_numeric(raw_movs, errors='coerce').sum()
            duracion_total = (fin_max - inicio_min).total_seconds() / 3600

            c1, c2, c3 = st.columns(3)
            c1.metric("Fin de Operación", fin_max.strftime("%d/%m %H:%M"))
            c2.metric("Duración Total (inc. Pausas)", f"{duracion_total:.1f} Hrs")
            c3.metric("Movimientos Totales", int(total_movs))
            
            st.divider()
            
            col_titulo, col_intervalo, col_orden = st.columns([2, 1, 1])
            with col_titulo:
                st.subheader("Línea temporal de Working List")
            with col_intervalo:
                intervalo_minutos = st.number_input("Grid Tiempo (min)", 10, 120, 10, 15)
            with col_orden:
                orden_pref = st.selectbox("Orden de Bays", ["Mayor a Menor (38 ➡ 02)", "Menor a Mayor (02 ➡ 38)"], index=0)
            
            mapa_colores = {
                "STS1 - Descarga": "#D35400", "STS1 - Embarque": "#F39C12",
                "STS2 - Descarga": "#2980B9", "STS2 - Embarque": "#5DADE2",
                "Pontón": "#7F8C8D", "Otro": "#BDC3C7"
            }

            bays_unicas = sorted(df_cronograma["Bay Visual"].unique())
            if "Mayor a Menor" in orden_pref:
                lista_ordenada = sorted(bays_unicas, reverse=True) 
            else:
                lista_ordenada = bays_unicas 

            min_inicio = df_cronograma['Inicio'].min()
            max_fin = df_cronograma['Fin'].max()
            range_y_start = min_inicio - timedelta(minutes=30)
            range_y_end = max_fin + timedelta(minutes=30)

            fig = go.Figure()

            # 1. Dibujar Barras de Tareas (Ahora son más largas porque incluyen el hueco del break)
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

            # 2. Dibujar Bloques de Colación (SUPERPUESTOS)
            # Esto crea el efecto visual: Naranja -> Gris -> Naranja
            dias_unicos = pd.to_datetime(df_cronograma['Inicio']).dt.date.unique()
            # Asegurar cubrir día final
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
                        fillcolor="rgba(50, 50, 50, 1.0)", # Gris sólido para tapar bien
                        line_width=0,
                        layer="above", # IMPORTANTE: Encima de las barras
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
                    layer="below traces" # Grid detrás
                ),
                xaxis=dict(
                    title=dict(text="Bay (Serie Par)", font=dict(color='white', size=16, weight="bold")),
                    type='category', categoryorder='array', categoryarray=lista_ordenada,
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