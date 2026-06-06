import marimo

__generated_with = "0.23.6"
app = marimo.App()

with app.setup:
    # setup cell
    import subprocess as sp
    from pathlib import Path
    import marimo as mo


@app.function
def fenced_block(text: str, language: str = "bash") -> str:
    """
    Return a Markdown fenced code block, choosing a fence long enough
    not to conflict with the block contents.
    """
    fence = "```"
    while fence in text:
        fence += "`"

    return f"{fence}{language}\n{text.rstrip()}\n{fence}"


@app.function
def sh(
    command: str,
    cwd: str | Path = ".",
    language: str = "bash",
) -> mo.Html:
    result = sp.run(
        ["bash", "-lc", command],
        cwd=Path(cwd),
        capture_output=True,
        text=True,
        check=False,
    )

    parts = []

    if result.stdout:
        parts.append("**stdout**")
        parts.append(fenced_block(result.stdout, language))

    if result.stderr:
        parts.append("**stderr**")
        parts.append(fenced_block(result.stderr, language))

    if result.returncode != 0:
        parts.append(f"**exit status:** `{result.returncode}`")

    if not parts:
        parts.append("Command produced no output.")

    return mo.md("\n\n".join(parts))


@app.cell
def _(app):
    if __name__ == "__main__":
        app.run()
    return


if __name__ == "__main__":
    app.run()
