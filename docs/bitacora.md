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
