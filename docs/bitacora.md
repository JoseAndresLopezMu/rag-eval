# Bitácora

Registro de cada paso: qué hizo el agente, qué revisé y qué decidí. Es la fuente de la sección "Cómo lo construí" del README.

## Paso 0 - Elección del proyecto (2026-11-17)

**Agente:** propuso tres ideas (analizador de licitaciones, evaluador de RAG y alertas NDVI) y recomendó el analizador de licitaciones.

**Revisión:** pregunté qué aportaba cada idea frente a lo que ya tengo publicado. Al compararlo, el analizador de licitaciones resultó ser casi lo mismo que `invoice-processor` (extraer datos de un PDF con un LLM) con otro tipo de documento, y la idea NDVI repetía temática con dos proyectos existentes.

**Decisión:** evaluador de RAG, porque es lo único que no tengo: mis proyectos construyen sistemas, pero ninguno mide si funcionan. Código en inglés y documentación en español.

## Paso 1 - Especificación y CLAUDE.md (2026-11-17)

**Agente:** redactó `docs/SPEC.md` y `CLAUDE.md`. Antes, revisó la API de chatbot-rag y vio que `/api/chat` no devuelve el texto de los contextos, solo archivo y página. Lo dejó como decisión pendiente (SPEC, sección 7).

**Revisión:** dejé abiertas tres preguntas (umbral de cobertura, cómo obtener los contextos de chatbot-rag y escala de `correctness`) y pedí al agente que eligiera la opción más profesional en cada una, justificándola en la SPEC.

**Decisión:** acepté sus propuestas, que quedan razonadas en la SPEC (sección 12):
- Umbral del 60 %.
- Modificar chatbot-rag para que devuelva los contextos reales en vez de aproximarlos con una segunda llamada.
- `correctness` binaria.

Lo más relevante fue el cambio de criterio en el adaptador. En el borrador, el agente proponía la opción más cómoda (A), que habría medido la fidelidad contra contextos distintos a los que vio el LLM.

## Paso 2 - Estructura, formatos y CI (2026-11-18)

**Agente:**
- Creó el proyecto con uv, los modelos Pydantic de dataset y ejecución, el lector de JSONL, el punto de entrada de la CLI y el workflow de GitHub Actions (lint, formato, tipos y tests). Hay 27 tests y ninguno usa red.
- Antes de fijar los formatos revisó cómo trocea chatbot-rag los documentos: los `.docx` no tienen páginas y todos sus fragmentos llevan página 1. Por eso `page` es opcional; la SPEC ya hacía depender la relevancia solo de `source` y la cita.
- Los modelos rechazan campos desconocidos, de modo que una errata como `refrence_answer` falla al cargar en vez de perderse sin avisar.
- Ruff marcó un patrón de test mal escapado. Al corregirlo, el agente sustituyó también un `getattr` con `noqa` que había usado para leer el `id`, poniendo en su lugar una clase base `Record` con tipado correcto.

**Revisión:** revisé los cambios y no pedí modificaciones.

**Decisión:** acepté `page` opcional y el rechazo de campos desconocidos. La subida queda condicionada a que CI pase en GitHub.

## Paso 3 - Regla de relevancia y métricas de recuperación (2026-11-20)

**Agente:**
- Implementó `hit@k`, `recall@k`, `precision@k` y `mrr`. La regla de relevancia se inyecta como función, así que las métricas se prueban sin depender de ella.
- Antes de implementar la regla, la probó con el texto real del pliego troceado en 50 fragmentos. La regla de la SPEC (60 % de los tokens, en cualquier orden) daba hasta 33 falsos positivos por cita. Exigir un tramo seguido los bajaba a 0–1. Según `CLAUDE.md`, un cambio de la SPEC requiere decisión del autor, así que paró y preguntó.
- Al revisar el pliego vio que el ejemplo de la SPEC (una cita sobre ISO 27001) no existía en el documento: lo había inventado como ilustración en el paso 1. Lo sustituyó por citas reales.
- Un test falló porque el agente contó a mano 18 tokens donde había 19. Lo comprobó con el propio normalizador antes de corregir el test, no el código.
- Después de subir el paso, el agente detectó que el texto de la portada usado en un test, presentado como "extracto real del pliego", tenía la segunda mitad completada de memoria. Lo sustituyó por el texto literal. El resultado del test no cambiaba, pero el comentario afirmaba algo falso. Es el mismo error que el ejemplo de ISO 27001: rellenar con algo verosímil en vez de comprobarlo. Para evitarlo, se añadió una regla a `CLAUDE.md`.

**Revisión:** el agente me presentó tres opciones (tokens sueltos, tramo seguido, cita exacta) con la tabla de falsos positivos y un ejemplo real: la portada del pliego, que con tokens sueltos cubría el 80 % de la cita "Servicio de recuperación de datos" sin contenerla.

**Decisión:** elegí el tramo seguido con umbral del 60 %. La cita exacta se descartó porque fallaría con citas cortadas por el borde de un fragmento. Queda como norma escribir citas de al menos 8 palabras. `precision@k` se divide por los contextos devueltos, no por k.

**Aprendizaje:** una regla que parecía razonable en papel era claramente mala con datos reales. Probar la especificación contra el documento real antes de escribir el código evitó una métrica inflada.

## Paso 4 - Informe, intervalos de confianza y comparación (2026-11-21)

**Agente:**
- Implementó el bootstrap (intervalo del 95 % con 2000 remuestreos y semilla fija) y el bootstrap pareado para comparar ejecuciones, sin numpy. Un test demuestra por qué hace falta el emparejamiento: con preguntas de dificultad muy variable, los intervalos por separado se solapan, pero el pareado detecta una mejora constante de 0,1.
- Creó el informe JSON (puntuaciones por pregunta, medias con intervalo, configuración, hash del dataset y de la ejecución, versión y commit) y los comandos `evaluate` y `compare`. La comparación se niega a comparar informes de datasets o configuraciones distintas.
- Las puntuaciones por pregunta se guardan como un diccionario de métricas, para que las de generación del paso 5 entren sin cambiar el formato.
- Tres tests fallaron por errores del agente al escribirlos, no del código:
  - supuso que dos semillas distintas siempre dan intervalos distintos, lo que no es cierto con 5 valores;
  - usó `pytest.approx` con un dataclass;
  - olvidó que el umbral del 60 % es inclusivo (3 de 5 palabras = 0,6).
  En los tres casos comprobó el comportamiento real antes de corregir el test.
- Probó la herramienta de punta a punta con 8 citas literales del pliego y dos ejecuciones sintéticas que no se publican en el repo. El informe dio un acierto más de los que había colocado el generador. Al investigarlo vio que un fragmento de relleno empezaba a mitad de la cita (7 de 10 palabras): era un acierto legítimo según la regla, y el error estaba en la suposición del generador.

**Revisión:** vi el informe y la tabla de comparación del ejemplo, y pedí continuar sin cambios.

**Decisión:** acepté el diseño tal como estaba, incluido marcar como `inconclusive` cualquier diferencia cuyo intervalo incluya el 0. En el ejemplo, con 8 preguntas, solo la mejora del MRR resultó concluyente: un dataset pequeño no permite afirmar mejoras pequeñas.

## Paso 5 - Juez LLM y métricas de generación (2026-11-22)

**Agente:**
- Separó el juez en capas: prompts versionados, un protocolo `Judge` con `ask(prompt, esquema)`, un `FakeJudge` para tests, un `CachedJudge` en disco y un `GroqJudge` sobre HTTP. Las métricas no saben qué juez usan, así que se prueban sin red.
- `GroqJudge` usa temperatura 0 y modo JSON. Si la respuesta no cumple el esquema, el modelo tiene un único reintento con el error delante. Los errores 429 y 5xx se reintentan con espera exponencial, respetando `retry-after`. Los 4xx no se reintentan. Todo se prueba con `httpx.MockTransport`.
- Concretó en la SPEC tres casos que no estaban definidos: la fidelidad se mide contra todos los contextos devueltos, una respuesta sin afirmaciones no recibe nota de fidelidad (contarla como 1 inflaría la media) y la fidelidad también se calcula en preguntas sin respuesta.
- Antes de fijar el modelo por defecto revisó chatbot-rag: genera con `llama-3.3-70b-versatile`. Usar el mismo modelo como juez tiene riesgo de autopreferencia. Lo dejó documentado en la SPEC como decisión pendiente.
- Errores propios detectados por ruff y mypy: un patrón de test sin escapar y un import que mypy no aceptaba. Al preparar la prueba manual, un heredoc anidado mal escrito no llegó a crear los ficheros, y el primer extracto del pliego se cortaba antes de la frase que la prueba necesitaba.
- El juez real con Groq no se podía probar sin una clave. El agente intentó buscar una en el entorno, pero el sistema de permisos lo bloqueó (correctamente) y pidió la clave al autor.

**Prueba real:** con una clave temporal, el agente listó los modelos de Groq y descubrió que `llama-3.3-70b-versatile` ya no existe. Era el modelo por defecto del juez y el que usa chatbot-rag si no se configura otro. Sin la prueba real, el juez habría fallado en la primera ejecución.

Prueba de humo con `openai/gpt-oss-120b` y 3 preguntas con texto literal del pliego:
- Respuesta fiel: 5 de 5 afirmaciones respaldadas, y correcta.
- Pregunta sin respuesta: abstención detectada. Al no tener afirmaciones, no recibe nota de fidelidad.
- Respuesta con una afirmación inventada a propósito ("copias de seguridad diarias con retención de 90 días"): el juez la marca como no respaldada, con fidelidad 0,5. La respuesta sí sale correcta, porque contesta bien a la pregunta. Esto muestra por qué hacen falta las dos métricas.
- Al repetir la ejecución: 0 llamadas a la API, 6 desde caché y un informe idéntico.

**Revisión:** no encontraba la clave de Groq. El agente explicó cómo crear una nueva y la generé con caducidad de 7 días para esta prueba.

**Decisión:** el modelo del juez pasa a ser `openai/gpt-oss-120b`. Lo eligió el agente porque es el más capaz de los disponibles y no coincide con el generador. Queda pendiente para el paso 6: chatbot-rag tiene que cambiar de modelo, y conviene que no use el mismo que el juez.

## Paso 6a - Adaptador de chatbot-rag y validación de citas (2026-11-24)

**Agente:**
- Antes de escribir el adaptador comprobó la API desplegada de chatbot-rag. Está caída por dos motivos: la base de datos de Supabase ya no existe (`tenant/user ... not found`) y la clave de Groq de Vercel ha caducado. Además, el modelo que usa por defecto ya no existe. Quedó documentado en la SPEC (7.2).
- Intentó levantar chatbot-rag en local para no depender de producción, pero no hay Node.js ni Docker en el equipo.
- Probó modelos de Groq para sustituir al generador retirado. Propone `qwen/qwen3.8-27b`: responde sin texto de razonamiento mezclado y es de otra familia que el juez.
- Fijó en la SPEC (7.1) el contrato exacto con chatbot-rag e implementó el adaptador y el comando `collect`. Si la respuesta no trae `contexts`, falla con un mensaje que apunta al despliegue, en vez de evaluar con contextos vacíos.
- Implementó `check-quotes`, que estaba previsto en los riesgos de la SPEC. Comprueba que cada cita es literal en su documento y tiene al menos 8 palabras. Lee DOCX con la librería estándar (sin dependencias nuevas) y PDF a través de su texto extraído. Se probó con el pliego y las tres ofertas reales.
- Errores propios:
  - Cuarto fallo de un test por contar palabras a mano (12 y no 11). Desde ahora el cálculo esperado se escribe en el test.
  - Ruff detectó una función definida dentro de un bucle que capturaba sus variables.
  - El extracto del pliego de un test saltaba una frase sin indicarlo. Se corrigió por la regla de extractos literales del `CLAUDE.md`.
  - Dos ediciones de la SPEC fallaron sin que lo notara, y el commit salió con un mensaje que no correspondía a su contenido. Se rehízo antes de subirlo.

**Revisión:**
<!-- Pendiente -->

**Decisión:**
<!-- Pendiente: cómo restaurar chatbot-rag (Supabase y Vercel) y quién escribe el borrador de las ~30 preguntas -->

## Paso 6b - chatbot-rag de vuelta en producción (2026-11-28)

**Agente:**
- Guió la restauración paso a paso. Antes de conectar Vercel con GitHub comprobó que GitHub tenía el mismo código que producción: la ruta `/api/process-document`, que el último commit había renombrado, ya existía desplegada, y la antigua `/api/ingest` daba 404. Así, conectar no sustituiría producción por otra versión.
- Implementó `include_contexts` en chatbot-rag y cambió el modelo por defecto. Sin Node.js en el equipo no podía validar el JavaScript antes de un push que despliega directamente en producción, así que descargó el binario oficial de Node (con la suma SHA-256 verificada) en una carpeta temporal, solo para `node --check`.
- El primer push no se desplegó: el proyecto de Vercel no estaba conectado a GitHub, aunque lo parecía. Lo detectó porque producción seguía sin `contexts` y GitHub no mostraba ningún estado de Vercel para el commit.
- Prueba de punta a punta con 3 preguntas contra la API real (`collect` y `evaluate`): q01 recupera la evidencia, pero en la posición 5 de 5; q03 no la recupera y el modelo se abstiene; la pregunta sin respuesta se abstiene correctamente. Indica que los fallos están en la recuperación, no en la generación, pero con 3 preguntas no se puede concluir nada.
- El juez no se pudo ejecutar porque la clave temporal ya no era válida (`invalid_api_key`).

**Revisión y decisiones:** la restauración la hice yo en los paneles, porque el agente no tiene acceso a mis cuentas:
- Reanudé Supabase.
- En Vercel, puse una clave de Groq nueva y `GROQ_MODEL`, y conecté el proyecto con GitHub.
- En GitHub, limité la app de Vercel a acceder solo a `chatbot-rag`, en lugar de a todos mis repositorios.
- Al ejecutar yo la evaluación con juez, pegué por error el texto de ejemplo del comando (`gsk_aquí_tu_clave_completa`) como clave. rag-eval se rompía con un traceback de `httpx` (`UnicodeEncodeError` por la "í") en vez de dar un mensaje claro. Se añadió primero el test que lo reproduce y luego el arreglo: la clave se valida al crear el juez, los espacios de alrededor del copiado se ignoran y el mensaje de error nunca muestra la clave. **Aprendizaje:** las instrucciones con huecos para rellenar invitan a este error, así que es mejor pedir la clave de forma interactiva (`read -s`).
