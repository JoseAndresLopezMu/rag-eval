# rag-eval: especificación

Estado: v1 revisada
Fecha: 2026-11-17

## 1. Problema

He construido varios sistemas RAG (el más completo es `chatbot-rag`, con BM25 + pgvector + RRF + cross-encoder + MMR), pero no tengo forma de medir si funcionan bien ni de demostrar que una técnica mejora a otra. Hoy la única validación es probar unas cuantas preguntas a mano.

`rag-eval` es una herramienta de línea de comandos que evalúa un sistema RAG contra un conjunto de preguntas de referencia y produce un informe con métricas de recuperación y de generación, comparable entre ejecuciones.

## 2. Objetivos

1. Medir la **recuperación**: si el sistema encuentra los fragmentos que contienen la respuesta.
2. Medir la **generación**: si la respuesta es fiel al contexto recuperado y si es correcta respecto a la respuesta de referencia.
3. **Comparar** dos ejecuciones (por ejemplo, búsqueda semántica frente a híbrida) diciendo si la diferencia es real o puede ser ruido.
4. Saber **cuánto fiarse del juez LLM**, midiendo su acuerdo con etiquetas hechas a mano.
5. Resultados **reproducibles**: misma entrada, mismo informe.

## 3. Fuera de alcance (v1)

- Generar preguntas sintéticas con un LLM. El conjunto de referencia lo escribo yo.
- Interfaz web. Solo CLI e informes en JSON y Markdown.
- Soporte para varios proveedores de LLM. Solo Groq y un juez falso para tests.
- Evaluar conversaciones con varios turnos. Solo preguntas aisladas.
- Métricas de coste o latencia más allá de registrar el tiempo de respuesta.

## 4. Conceptos y formatos

### 4.1 Conjunto de referencia (`dataset.jsonl`)

Una pregunta por línea:

```json
{
  "id": "q001",
  "question": "¿Qué trabajos incluye el mantenimiento de la nube privada de disco?",
  "reference_answer": "Mantenimiento de la nube, gestión de cuentas y permisos de usuario, integración CIFS, recuperación de datos y soporte in situ.",
  "answerable": true,
  "evidence": [
    {
      "source": "pliego_tecnico_madrimasd_S_2026_003.pdf",
      "page": 7,
      "quote": "Creación y mantenimiento de cuentas de usuario para la nube"
    },
    {
      "source": "pliego_tecnico_madrimasd_S_2026_003.pdf",
      "page": 7,
      "quote": "Servicio de recuperación de datos"
    }
  ]
}
```

- `evidence` lista los fragmentos del documento original que justifican la respuesta. Puede haber varios.
- `answerable: false` marca preguntas cuya respuesta no está en los documentos. Sirven para comprobar que el sistema se abstiene en lugar de inventar. No tienen `evidence`.

### 4.2 Ejecución (`run.jsonl`)

Lo que devolvió el sistema evaluado para cada pregunta:

```json
{
  "id": "q001",
  "answer": "El pliego exige ISO 27001 y ...",
  "contexts": [
    {"rank": 1, "source": "pliego_tecnico_madrimasd_S_2026_003.pdf", "page": 7, "text": "..."}
  ],
  "latency_ms": 2140
}
```

Separar "obtener respuestas" de "evaluarlas" permite evaluar sin volver a llamar al sistema y hace que los tests no necesiten red.

### 4.3 Cuándo un fragmento recuperado es relevante

Cada sistema trocea los documentos a su manera, así que no se pueden usar identificadores de fragmento. Regla propuesta:

> Un contexto recuperado cubre una evidencia si tienen el mismo `source` y `text` contiene, como un tramo de tokens seguidos, al menos el 60 % de los tokens normalizados de `quote`.

Normalizar significa pasar a minúsculas, quitar acentos y signos de puntuación y separar por espacios. El umbral del 60 % tolera que el troceado corte la cita por la mitad: el fragmento contendrá el principio o el final de la cita como un tramo seguido. Es configurable, se guarda en el informe y se prueba con fragmentos reales del pliego.

**Por qué tramo seguido (revisión del paso 3).** La primera versión de esta regla contaba tokens sueltos, en cualquier orden. Al probarla con el texto real del pliego, troceado en 50 fragmentos de 800 caracteres, daba por relevantes muchos fragmentos que no contienen la cita, porque comparten palabras frecuentes ("de", "la", "servicio", "datos"):

| Cita | Falsos positivos con tokens sueltos | Con tramo seguido |
|---|---|---|
| será preciso, por parte del adjudicatario, implementar las siguientes tareas | 3 | 0 |
| Gestión de certificados y protocolos seguros | 3 | 1 |
| Mantenimiento de la nube privada de disco | 11 | 1 |
| Servicio de recuperación de datos | 33 | 0 |
| La administración de sistemas de servidores disponibles en la Fundación Madri+d | 22 | 0 |

Con tokens sueltos, la métrica de recuperación saldría inflada y un mal sistema parecería bueno. Los pocos falsos positivos que quedan con el tramo seguido aparecen en citas cortas (6–7 palabras), así que **conviene que las citas tengan al menos 8 palabras**.

Alternativas descartadas:
- Coincidencia por página: es demasiado gruesa, porque una página contiene muchos fragmentos que no responden a la pregunta.
- Cita exacta completa: no da falsos positivos, pero da por fallado un fragmento que tiene la cita cortada por el borde del troceado.
- Similitud por embeddings: añade un modelo y una fuente de error más a la propia medida.

## 5. Métricas

### 5.1 Recuperación (solo preguntas con `answerable: true`)

Por pregunta, con `k` configurable (5 por defecto):

| Métrica | Definición |
|---|---|
| `hit@k` | 1 si alguno de los k primeros contextos cubre alguna evidencia |
| `recall@k` | fracción de evidencias cubiertas por los k primeros contextos |
| `precision@k` | fracción de los contextos devueltos (como mucho k) que cubren alguna evidencia. Se divide por los devueltos, no por k, para no penalizar a un sistema que devuelve menos contextos. |
| `mrr` | 1 / posición del primer contexto relevante (0 si no hay ninguno) |

### 5.2 Generación (con juez LLM)

| Métrica | Cómo se calcula |
|---|---|
| `faithfulness` | El juez divide la respuesta en afirmaciones y marca cada una como apoyada o no por los contextos. Puntuación = apoyadas / total. |
| `correctness` | El juez decide si la respuesta es correcta (1) o incorrecta (0) respecto a `reference_answer`. Es correcta si contiene la información esencial de la referencia y no la contradice. |
| `abstention` | Solo para `answerable: false`: 1 si el sistema dice que no tiene la información y 0 si responde algo. |

El juez debe devolver JSON validado con Pydantic e incluir una justificación breve, que se guarda en el informe para poder auditar cada nota.

Casos concretados en el paso 5:
- `faithfulness` se mide contra **todos** los contextos devueltos, no solo los k primeros, porque el generador los vio todos.
- Una respuesta sin afirmaciones factuales (por ejemplo, "No tengo esa información") no recibe nota de `faithfulness`: la métrica no aplica, y contarla como 1 inflaría la media. Su juicio se guarda igualmente.
- `faithfulness` también se calcula en preguntas sin respuesta: si el sistema se inventa algo, debe penalizarse.

### 5.3 Agregados e intervalos

- Media de cada métrica sobre todas las preguntas aplicables.
- Intervalo de confianza del 95 % por bootstrap (2000 remuestreos, semilla fija).
- En una comparación entre dos ejecuciones se usa un bootstrap pareado sobre las diferencias por pregunta. Si el intervalo de la diferencia incluye 0, el informe la marca como "no concluyente".

Con unas 30 preguntas, muchas diferencias pequeñas no serán concluyentes. Prefiero que el informe lo diga a presentar como mejora algo que es ruido.

## 6. Juez LLM

- Interfaz `Judge` con dos implementaciones: `GroqJudge` (real) y `FakeJudge` (respuestas fijas para tests).
- Temperatura 0 y modelo fijado en la configuración.
- Los prompts están versionados (`PROMPT_VERSION`) y su versión se guarda en el informe.
- Caché en disco con clave `hash(modelo, versión del prompt, entradas)`, para no pagar dos veces por lo mismo y poder reproducir un informe.
- Modo JSON de la API. Si la respuesta no cumple el esquema, el modelo tiene un único reintento con el error de validación delante. Los errores 429 y 5xx se reintentan con espera exponencial, respetando `retry-after`.
- El juez es opcional (`rag-eval evaluate --judge`), para que las métricas de recuperación funcionen sin clave de API. El modelo es configurable (`--judge-model`), y dos informes con jueces distintos no se pueden comparar.
- **Riesgo de autopreferencia:** si el juez es el mismo modelo que generó las respuestas, puede favorecerlas. La calibración del paso 7 permite medirlo.
- **Modelo por defecto: `openai/gpt-oss-120b`.** Al probar con la API real (2026-11-22) se vio que Groq ya no ofrece `llama-3.3-70b-versatile`, el modelo previsto y el que usa chatbot-rag por defecto. Cuando chatbot-rag cambie de modelo en el paso 6, conviene que el generador no sea `gpt-oss-120b`, para mantener el juez y el generador separados.

### 6.1 Calibración del juez

Un juez LLM también se equivoca. Para saber cuánto:

1. Etiqueto a mano unas 20 respuestas (`faithfulness` y `correctness`) en `labels.jsonl`.
2. `rag-eval calibrate` ejecuta el juez sobre las mismas respuestas y calcula el acuerdo (exactitud y kappa de Cohen).
3. El README mostrará ese acuerdo junto a los resultados, para que se sepa cuánto pesan las notas del juez.

## 7. Sistemas evaluados (adaptadores)

| Adaptador | Qué hace |
|---|---|
| `jsonl` | Lee una ejecución ya guardada. Es el núcleo y lo que usan los tests. |
| `chatbot-rag` | Llama a la API de `chatbot-rag` y genera un `run.jsonl`. |

`/api/chat` de chatbot-rag devuelve `answer` y `sources` (archivo y página), pero no el texto de los fragmentos, que hace falta para `faithfulness`. Se consideraron dos opciones:

- A. El adaptador llama también a `/api/search` con la misma pregunta para obtener los contextos.
- B. Añadir a chatbot-rag un parámetro opcional `include_contexts` para que `/api/chat` devuelva los fragmentos que usó para generar la respuesta.

**Decisión: B.** Con A se evaluaría la fidelidad contra unos contextos que pueden no ser los que vio el LLM: `/api/chat` reformula la pregunta según el historial antes de recuperar, y el reranking puede variar. Sería una métrica que mide algo distinto de lo que dice medir, sin avisar. El cambio en chatbot-rag es pequeño y compatible hacia atrás: sin el parámetro, la respuesta no cambia. Se hará en su propio repositorio, con su propio commit, en el paso 6.

Para comparar búsqueda semántica frente a híbrida, chatbot-rag necesitará un parámetro que elija el modo de recuperación. Ese cambio queda fuera de este repositorio y se decidirá cuando llegue.

### 7.1 Contrato con chatbot-rag (paso 6)

Petición del adaptador a `POST {url}/chat`:

```json
{"input": "¿Incluye el servicio la recuperación de datos?", "include_contexts": true}
```

Respuesta esperada. `answer` y `sources` ya existen; `contexts` es lo nuevo y solo aparece si se pide:

```json
{
  "answer": "...",
  "sources": [{"fileName": "...", "pageNumber": 7}],
  "contexts": [{"content": "texto del fragmento", "fileName": "...", "pageNumber": 7}]
}
```

`contexts` son los fragmentos, en el orden en que se pasaron al LLM. El adaptador falla si la respuesta no trae `contexts`, en vez de suponer que está vacío: así una versión de chatbot-rag sin el cambio no produce una evaluación de fidelidad sin sentido.

### 7.2 Estado de chatbot-rag en producción (2026-11-24)

Al preparar el adaptador se comprobó la API desplegada (`https://chatbot-rag-src.vercel.app/api`). Las dos rutas fallan:

- `/api/search`: `tenant/user postgres.… not found`. El proyecto de Supabase ya no está disponible.
- `/api/chat`: `expired_api_key`. La clave de Groq configurada en Vercel ha caducado. Además, el modelo por defecto (`llama-3.3-70b-versatile`) ya no existe en Groq.

Para evaluar hace falta: restaurar o recrear la base de datos (esquema `db/schema.sql` y `npm run seed`), configurar en Vercel una clave de Groq válida y un modelo disponible, y desplegar el cambio de 7.1. Como generador se propone `qwen/qwen3.8-27b`: responde sin texto de razonamiento mezclado y es de otra familia que el juez (`openai/gpt-oss-120b`).

**Restaurado (2026-11-28).** El proyecto de Supabase estaba pausado, no borrado: se reanudó sin pérdida de datos. En Vercel se configuraron una clave de Groq nueva y `GROQ_MODEL=qwen/qwen3.8-27b`. El proyecto de Vercel, que se había desplegado con la CLI, se conectó al repositorio de GitHub (con acceso de la app de Vercel limitado a ese repositorio), y el cambio de 7.1 se desplegó automáticamente desde `main`. Sin `include_contexts`, la respuesta de `/api/chat` no cambia.

Observaciones al restaurarlo:
- chatbot-rag asigna página 1 a todos los fragmentos, también a los del PDF. No afecta a la evaluación, porque la relevancia depende del documento y la cita (4.3), no de la página.
- Los DOCX se convierten a Markdown antes de trocearlos (`# 2. Cumplimiento…`, `**Expediente…**`). La normalización de 4.3 elimina esos símbolos, así que las citas copiadas del texto del DOCX siguen coincidiendo.

## 8. CLI

```
rag-eval collect --target chatbot-rag --dataset data/dataset.jsonl --out runs/hybrid.jsonl
rag-eval evaluate --dataset data/dataset.jsonl --run runs/hybrid.jsonl --out reports/hybrid.json
rag-eval compare reports/semantic.json reports/hybrid.json --out reports/comparison.md
rag-eval calibrate --labels data/labels.jsonl
rag-eval check-quotes --dataset data/dataset.jsonl --sources-dir documents/
```

El informe JSON incluye: métricas por pregunta, agregados con intervalos, configuración (k, umbral, modelo del juez, versión del prompt), hash del dataset y commit de git.

## 9. Stack

- Python 3.12, gestionado con `uv`.
- `pydantic` para formatos y para validar la salida del juez.
- `typer` para la CLI.
- `httpx` para las llamadas HTTP, tanto a Groq como a chatbot-rag. Sin SDKs de LLM: la API de Groq es compatible con OpenAI y basta con una petición.
- `pytest`, `ruff` y `mypy` en CI.

Sin numpy ni pandas: el bootstrap y las métricas son sencillos de escribir y así se pueden testear sin dependencias pesadas.

## 10. Plan de trabajo

Cada paso termina con tests en verde, se revisa antes de subirse y se anota en `docs/bitacora.md`.

| Paso | Contenido |
|---|---|
| 1 | Especificación, `CLAUDE.md` y bitácora (sin código) |
| 2 | Estructura del proyecto, modelos Pydantic de dataset y ejecución, CI con GitHub Actions |
| 3 | Regla de relevancia y métricas de recuperación |
| 4 | Informe, agregados, bootstrap y comando `compare` |
| 5 | Interfaz del juez, `FakeJudge`, `GroqJudge`, caché y métricas de generación |
| 6 | Parámetro `include_contexts` en chatbot-rag, adaptador y conjunto de referencia real (lo escribo yo) |
| 7 | Calibración del juez con etiquetas manuales |
| 8 | Primera evaluación real de chatbot-rag y README con "Cómo lo construí" |

## 11. Riesgos

- **Dataset pequeño:** 30 preguntas dan intervalos anchos. Se asume y se muestra.
- **Sesgo del juez:** mitigado con la calibración (6.1) y guardando sus justificaciones.
- **Citas mal copiadas:** si `quote` no coincide con el texto extraído del PDF, la métrica de recuperación da 0 sin culpa del sistema. `rag-eval check-quotes` comprueba que cada cita aparece literalmente en su documento fuente (PDF mediante su texto extraído con `pdftotext`; DOCX, TXT y MD directamente) y que tiene al menos 8 palabras.
- **Coste de API:** la caché evita repetir llamadas, y 30 preguntas con Groq tienen un coste muy bajo.

## 12. Decisiones tomadas en la revisión

1. **Umbral de cobertura: 60 %**, configurable y registrado en el informe (sección 4.3). En el paso 3 se cambió de tokens sueltos a tramo seguido, tras medir los falsos positivos con el pliego real.
2. **Adaptador: opción B** (sección 7), para que `faithfulness` se mida contra los contextos que vio el LLM y no contra una aproximación.
3. **`correctness` binaria.** Una escala de tres niveles parece más fina, pero el nivel "parcial" es donde más discrepan entre sí los anotadores humanos y los jueces LLM. Con unas 20 etiquetas de calibración, una decisión binaria da un acuerdo juez-humano más estable y más fácil de interpretar. Si el análisis de errores muestra muchas respuestas a medias, se revisará.
