from __future__ import annotations

import typer

from ts3d.cli import eval as _eval
from ts3d.cli import train as _train

app = typer.Typer(
    name="ts3d",
    help="3D candidate cluster verification for traffic sign FP reduction.",
    add_completion=False,
    no_args_is_help=True,
)

app.command("train")(_train.main)
app.command("eval")(_eval.main)


if __name__ == "__main__":
    app()
