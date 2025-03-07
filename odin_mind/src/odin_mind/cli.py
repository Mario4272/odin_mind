"""Console script for odin_mind."""
import odin_mind

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for odin_mind."""
    console.print("Replace this message by putting your code into "
               "odin_mind.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
