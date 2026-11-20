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

**Revisión:** el agente me presentó tres opciones (tokens sueltos, tramo seguido, cita exacta) con la tabla de falsos positivos y un ejemplo real: la portada del pliego, que con tokens sueltos cubría el 80 % de la cita "Servicio de recuperación de datos" sin contenerla.

**Decisión:** elegí el tramo seguido con umbral del 60 %. La cita exacta se descartó porque fallaría con citas cortadas por el borde de un fragmento. Queda como norma escribir citas de al menos 8 palabras. `precision@k` se divide por los contextos devueltos, no por k.

**Aprendizaje:** una regla que parecía razonable en papel era claramente mala con datos reales. Probar la especificación contra el documento real antes de escribir el código evitó una métrica inflada.
