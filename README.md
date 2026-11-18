# rag-eval

Herramienta para evaluar sistemas RAG: métricas de recuperación, métricas de generación con un juez LLM calibrado y comparación entre ejecuciones con intervalos de confianza.

En desarrollo. La especificación está en [docs/SPEC.md](docs/SPEC.md) y el registro del proceso en [docs/bitacora.md](docs/bitacora.md).

## Desarrollo

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```
