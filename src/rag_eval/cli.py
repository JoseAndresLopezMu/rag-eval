from typing import Annotated

import typer

from rag_eval import __version__

app = typer.Typer(no_args_is_help=True, help="Evaluate RAG systems against a reference dataset.")


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"rag-eval {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_print_version, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    pass
