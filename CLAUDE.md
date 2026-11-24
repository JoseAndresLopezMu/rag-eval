# CLAUDE.md

Instrucciones para el agente que trabaja en este repositorio.

## Qué es

`rag-eval` evalúa sistemas RAG: métricas de recuperación, métricas de generación con un juez LLM y comparación entre ejecuciones con intervalos de confianza. La especificación completa está en `docs/SPEC.md`. Léela antes de proponer cambios y no implementes nada que no esté ahí sin preguntar.

## Arquitectura

```
src/rag_eval/
  models.py        # Formatos Pydantic: Record, DatasetItem, Evidence, RunItem, Context, Report
  relevance.py     # Regla de cobertura cita/contexto (SPEC 4.3)
  metrics/
    retrieval.py   # hit@k, recall@k, precision@k, mrr
    generation.py  # faithfulness, correctness, abstention (usan un Judge)
  stats.py         # Agregados y bootstrap (simple y pareado)
  quotes.py        # Comprueba que las citas del dataset son literales
  judge/
    base.py        # Protocolo Judge y modelos de salida
    groq.py        # GroqJudge (httpx contra la API compatible con OpenAI)
    fake.py        # FakeJudge para tests
    cache.py       # Caché en disco por hash(modelo, versión de prompt, entradas)
    prompts.py     # Prompts versionados (PROMPT_VERSION)
  adapters/
    jsonl.py       # Lee dataset y ejecuciones desde JSONL, con errores por línea
    chatbot_rag.py # Llama a la API de chatbot-rag (contrato en SPEC 7.1)
    sources.py     # Lee el texto de los documentos fuente (TXT, MD, DOCX)
  report.py        # Construye el informe JSON y el Markdown de comparación
  cli.py           # Typer: collect, evaluate, compare, calibrate
tests/
data/              # Dataset y etiquetas manuales
```

Flujo: `check-quotes` (validar el dataset), `collect` (adaptador → run.jsonl), `evaluate` (dataset + run → informe), `compare` (dos informes → Markdown).

Las métricas son funciones puras: reciben modelos y devuelven números. Toda la E/S (ficheros, red) vive en `adapters/`, `judge/groq.py` y `cli.py`.

## Comandos

```bash
uv sync                       # instalar dependencias
uv run pytest                 # tests
uv run ruff check .           # lint
uv run ruff format .          # formato
uv run mypy src tests         # tipos (modo estricto)
uv run rag-eval --help        # CLI
uv run rag-eval evaluate --dataset d.jsonl --run r.jsonl --out report.json [--judge]
uv run rag-eval compare a.json b.json
uv run rag-eval collect --dataset d.jsonl --url https://host/api --out r.jsonl
uv run rag-eval check-quotes --dataset d.jsonl --sources-dir documentos/
```

`--judge` llama a Groq (necesita `GROQ_API_KEY`) y guarda las respuestas en `.rag_eval_cache/`. Los tests nunca lo usan.

Antes de dar un paso por terminado deben pasar los cuatro: pytest, ruff check, ruff format --check y mypy. Son los mismos que ejecuta CI (`.github/workflows/ci.yml`) en cada push a `main` y en cada pull request.

## Convenciones

- Código, nombres, tests y mensajes de commit en inglés. Documentación (`docs/`, README) en español.
- Type hints en todo el código de `src/`.
- Comentarios solo cuando explican un porqué que no se deduce del código. Nada de comentarios que repiten lo que hace la línea siguiente ni docstrings de relleno.
- Sin emojis en código, commits ni documentación.
- Errores claros y tempranos: si el dataset tiene un campo mal, falla al cargar con el id de la pregunta y el campo, no más tarde.

## Tests

- Los tests no hacen llamadas de red. El juez se sustituye por `FakeJudge` y las APIs por `httpx.MockTransport`.
- Cada métrica tiene tests con casos calculados a mano, incluidos los bordes: sin contextos, sin evidencias, k mayor que el número de contextos o empates.
- Si corriges un bug, añade primero un test que lo reproduzca.
- Un texto presentado como extracto de un documento real (en tests, ejemplos o la SPEC) se copia literalmente del documento. Nunca se completa ni se parafrasea de memoria.

## Commits

- Pequeños y de un solo propósito, con el formato Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `refactor:`, `chore:`.
- No añadas líneas de coautoría ni de herramientas generadoras en los mensajes.
- No hagas `git push` ni crees el repositorio remoto: lo hace el autor después de revisar.

## Límites

- No modifiques `data/dataset.jsonl` ni `data/labels.jsonl` salvo petición explícita: son las etiquetas humanas contra las que se mide todo.
- No añadas dependencias que no estén en la SPEC (sección 9) sin preguntar.
- No cambies la definición de una métrica para que un test pase. Si la SPEC está mal, dilo y espera una decisión.
- Las claves de API se leen de variables de entorno (`GROQ_API_KEY`). Nunca en código, tests ni ficheros versionados.
