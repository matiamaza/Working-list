# Portal Operaciones SVTI

Portal Streamlit para planificar operaciones de grúas en puerto: simulador de colas de trabajo, planificador STS
y análisis de interferencias entre grúas con propuesta de re-secuenciación.

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt   # solo para correr pruebas
```

## Ejecución local

```powershell
streamlit run app.py
```

Pruebas: `python -m pytest tests -q` (la suite completa tarda unos 6 minutos).

## Despliegue en Streamlit

Suba el repositorio a GitHub y cree la app en Streamlit Community Cloud apuntando a `app.py`. Las dependencias se
toman de `requirements.txt` (incluye `ortools`). No se requieren secretos para el análisis de interferencias.

## Reparación y optimización híbrida

En «Interferencias entre grúas» → «Configuración de optimización» se elige el método:

| Método | Qué hace | Tiempo |
|---|---|---|
| Reparación rápida | Candidatas iniciales, CP-SAT global, reparación y mejora local. Sin ALNS. | 10 s |
| Híbrido recomendado | Lo anterior + ALNS + intensificación final. | 20 s |
| Intensivo | Igual que el híbrido con vecindad mayor. | 60 s |

El flujo híbrido es: candidatas iniciales → primera solución factible → mejora local → ALNS → intensificación final.
El resultado es siempre el mejor de todas las fases: si el ALNS no mejora, se devuelve el incumbente anterior.
El resumen indica el estado: *Solución factible mejorada*, *Solución factible mantenida*, *Solución reparada*
(la semilla tenía interferencias o cruces y se resolvieron) o *Solución parcialmente reparada*.

## Significado de la semilla

La semilla se conserva como referencia, pero no restringe horarios, primera tarea, primer bay ni asignación.
El orden físico sí permanece fijo para evitar cruces.

De la semilla solo se heredan las precedencias dentro de cada bay (descarga antes que embarque, cubierta antes que
bodega) y los pontones en su grúa.

## Orden físico

Por defecto se infiere de la semilla (orden de proa a popa según el bay de la primera tarea de cada grúa). En
«Configuración avanzada» se puede desactivar e indicar el orden manualmente; deben aparecer todas las grúas, una sola
vez. El orden impide cruces, pero no fija qué tarea o bay trabaja primero cada grúa.

## Reasignación y división

- *Permitir reasignación*: una tarea puede pasar a otra grúa.
- *Permitir división*: una tarea puede partirse en fragmentos que ejecutan grúas distintas. Todos los
  fragmentos terminan antes de que comience la tarea sucesora.

Los movimientos totales siempre se conservan.

## Parámetros recomendados

`tiempo_limite=20`, `random_seed=42`, `workers=4`, `activar_alns=True`, `porcentaje_vecindad=0.10`,
`permitir_reasignacion=True`, `permitir_division=True`, `usar_simulated_annealing=True`, `usar_cpsat_local=True`,
`activar_mejora_local_inicial=True`, `activar_intensificacion_final=True`, `orden_fisico_gruas=None`.

## Modo reproducible

`proponer_secuencia(..., modo_reproducible=True)` fuerza `workers=1` y usa la semilla aleatoria fija. El modo paralelo
(`modo_reproducible=False`) es el predeterminado. Como el tiempo límite es real, el resultado puede variar si cambia la
carga del equipo. `propuesta.attrs` registra `random_seed`, `workers`, `tiempo_limite`, `modo_reproducible`,
`version_ortools`, `fecha_ejecucion`, `firma_entrada` y el estado y la mejor cota del CP-SAT global.

## Benchmarks

Sintético: `python benchmarks/benchmark_alns.py [tiempo_s]`.

Real (sin guardar datos sensibles en el repositorio):

```powershell
python benchmarks/benchmark_real.py ruta\archivo.xlsx --config gruas.benchmark.local.json --tiempo 20
```

Requiere el archivo de secuencia (formato de la app) y un JSON local con inicio y GMPH de cada grúa, por ejemplo:

```json
{"G1": {"inicio": "2025-01-10T08:00", "gmph": 25}, "G2": {"inicio": "2025-01-10T08:00", "gmph": 25}}
```

Corre las seeds 42, 43 y 44 y escribe un CSV en `benchmarks/results/`. Git ignora `benchmarks/data/`,
`benchmarks/results/` y `*.benchmark.local.json`.

## Limitaciones conocidas

- No considera el tipo de grúa (STS/MHC) ni restricciones de estiba.
- El ALNS rara vez mejora el incumbente en instancias pequeñas; su valor en instancias reales está por validar.
- El CP-SAT global con varios workers no es determinista; use el modo reproducible para comparar corridas.
- El CP-SAT local es una guía aproximada: el cronograma real se reconstruye con las reglas exactas.
