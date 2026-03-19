import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pdfplumber
import re
import os
import io
from datetime import datetime, timedelta, time

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
        last_total_m = 0
        
        for index, row in grupo_df.iterrows():
            raw_action, raw_bay = str(row.get('Action Name', '')).upper(), str(row.get('Bay', '')).upper()
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
                duracion_minutos = (movimientos / gmph) * 60
                kws_descarga = ["DISCH", "DSCH", "BLISCH", "DISH", "DDI", "DHI", "SCH", "DCH"]
                kws_embarque = ["LOAD", "HOAD", "DOAD", "LOD", "LOA", "O AD", " HL", " DL", "AD TWI"]
                
                if any(kw in texto_fila for kw in kws_descarga): categoria_visual = f"{nombre_grua} - Descarga"
                elif any(kw in texto_fila for kw in kws_embarque): categoria_visual = f"{nombre_grua} - Embarque"

            if duracion_minutos > 0:
                tiempo_fin = calcular_fin_con_descansos(tiempo_actual, duracion_minutos)
                cronograma.append({
                    "Grúa": nombre_grua, "Bay Real": raw_bay, "Bay Visual": bay_visual_str,
                    "Acción": f"{raw_action.replace(chr(10), ' ') or 'Contexto'} ({int(movimientos)})",
                    "Inicio": tiempo_actual, "Fin": tiempo_fin, "Duración Neta": round(duracion_minutos, 1),
                    "Categoría Visual": categoria_visual
                })
                tiempo_actual = tiempo_fin
    return pd.DataFrame(cronograma)

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
            
            for g in lista_gruas:
                with st.expander(f"Config {g}", expanded=False):
                    # --- NUEVO: Selector de Tipo de Grúa ---
                    tipo_grua = st.selectbox(f"Tipo {g}", ["STS", "MHC"], key=f"tipo_{g}")
                    
                    gmph = st.number_input(f"GMPH", 20, 40, 25, key=f"gmph_{g}")
                    hora = st.time_input(f"Inicio", time(8,0), key=f"h_{g}")
                    
                    # Guardamos el tipo en el diccionario de configuración
                    config_gruas[g] = {
                        "inicio": datetime.combine(datetime.today(), hora), 
                        "gmph": gmph,
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
            try:
                # Intentamos usar la variable unificada si existe (del paso anterior)
                df_procesar = df_editado_final 
            except:
                # Si no existe, usamos la del editor único
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
            df_cronograma = calcular_cronograma(df_para_gantt, config_gruas)
            
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
                pares_colores = [("#D35400", "#F39C12"), ("#2980B9", "#5DADE2"), ("#27AE60", "#58D68D"), ("#8E44AD", "#AF7AC5"), ("#C0392B", "#E74C3C")]
                mapa_colores = { "Pontón": "#7F8C8D", "Otro": "#BDC3C7" }
                
                # Asignar colores dinámicos según las grúas presentes
                gruas_activas = sorted(df_cronograma['Grúa'].unique())
                for i, nombre_g in enumerate(gruas_activas):
                    color_desc, color_emb = pares_colores[i % len(pares_colores)]
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
                        hovertemplate="<b>Bay %{x}</b><br>%{customdata[1]}<br>In: %{base|%H:%M}<br>Out: %{customdata[3]|%H:%M}<br>Neto: %{customdata[4]} min<extra></extra>"
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
                            annotation_position="top left", 
                            annotation_font_color="white", 
                            annotation_font_size=10
                        )

                # 5. Layout Oscuro (Idéntico al Simulador)
                fig.update_layout(
                    height=800, 
                    title="Planificación Maestra STS (Visualización Simulador)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='white'), bgcolor='rgba(0,0,0,0)'),
                    dragmode='pan', hovermode='closest', barmode='overlay', bargap=0.02,
                    plot_bgcolor='#1b263b', paper_bgcolor='#1b263b', font=dict(color='white'),
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

        gmph_default = 25.0
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
            for grua in lista_gruas:
                with st.sidebar.expander(f"⚙️ {grua}", expanded=True):
                    c1, c2 = st.columns(2)
                    d = c1.date_input("Fecha", datetime.today(), key=f"d_{grua}")
                    t = c2.time_input("Hora", time(8,0), key=f"t_{grua}")
                    gmph_input = st.number_input("GMPH", 1.0, 200.0, float(gmph_default), 0.1, key=f"g_{grua}")
                    config_gruas[grua] = {"inicio": datetime.combine(d, t), "gmph": gmph_input}

            df_cronograma = calcular_cronograma(df_raw, config_gruas)
            
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
                col_titulo, col_intervalo, col_orden = st.columns([2, 1, 1])
                with col_titulo: st.subheader("Línea temporal de Working List")
                with col_intervalo: intervalo_minutos = st.number_input("Grid Tiempo (min)", 10, 120, 60, 15)
                with col_orden: orden_pref = st.selectbox("Orden de Bays", ["Mayor a Menor", "Menor a Mayor"], index=0)
                
                pares_colores = [("#D35400", "#F39C12"), ("#2980B9", "#5DADE2"), ("#27AE60", "#58D68D"), ("#8E44AD", "#AF7AC5"), ("#C0392B", "#E74C3C")]
                mapa_colores = { "Pontón": "#7F8C8D", "Otro": "#BDC3C7" }
                for i, nombre_g in enumerate(lista_gruas):
                    color_desc, color_emb = pares_colores[i % len(pares_colores)]
                    mapa_colores[f"{nombre_g} - Descarga"] = color_desc
                    mapa_colores[f"{nombre_g} - Embarque"] = color_emb

                min_bay = int(df_cronograma["Bay Visual"].min())
                max_bay = int(df_cronograma["Bay Visual"].max())
                bays_completas = [f"{b:02d}" for b in range(min_bay, max_bay + 4, 4)]
                lista_ordenada = sorted(bays_completas, reverse=(orden_pref == "Mayor a Menor"))

                range_y_start = inicio_min - timedelta(minutes=30)
                range_y_end = fin_max + timedelta(minutes=30)

                fig = go.Figure()

                for categoria, color in mapa_colores.items():
                    df_cat = df_cronograma[df_cronograma['Categoría Visual'] == categoria]
                    if df_cat.empty: continue
                    duracion_ms = (df_cat['Fin'] - df_cat['Inicio']).dt.total_seconds() * 1000
                    fig.add_trace(go.Bar(
                        name=categoria, x=df_cat['Bay Visual'], y=duracion_ms, base=df_cat['Inicio'],
                        orientation='v', marker_color=color, marker_line_color='black', marker_line_width=1,
                        opacity=1.0, width=0.95,
                        customdata=df_cat[['Grúa', 'Acción', 'Inicio', 'Fin', 'Duración Neta']],
                        hovertemplate="<b>Bay %{x}</b><br>%{customdata[1]}<br>In: %{base|%H:%M}<br>Out: %{customdata[3]|%H:%M}<br>Neto: %{customdata[4]} min<extra></extra>"
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
                            annotation_text=label, annotation_position="top left", annotation_font_color="white", annotation_font_size=10
                        )

                fig.update_layout(
                    height=900, legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="right", x=1, font=dict(color='white'), bgcolor='rgba(0,0,0,0)'),
                    dragmode='pan', hovermode='closest', barmode='overlay', bargap=0.02,
                    plot_bgcolor='#1b263b', paper_bgcolor='#1b263b', font=dict(color='white'),
                    yaxis=dict(type='date', title="Hora", tickformat="%H:%M", tickfont=dict(color='white'), dtick=intervalo_minutos*60*1000, autorange="reversed", range=[range_y_start, range_y_end], gridcolor='#3a506b'),
                    xaxis=dict(title="Bay (Serie Par)", type='category', categoryorder='array', categoryarray=lista_ordenada, side="top", tickfont=dict(color='white'), gridcolor='#000000', fixedrange=True),
                    margin=dict(l=60, r=20, t=120, b=20)
                )
                
                config_zoom = {'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d', 'lasso2d']}
                st.plotly_chart(fig, use_container_width=True, config=config_zoom)
                
                with st.expander("Ver detalle"):
                    st.dataframe(df_cronograma)
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




