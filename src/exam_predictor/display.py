"""Show tables, text, charts and pictures: nicely in a notebook, as plain text in a terminal."""
import matplotlib.pyplot as plt
import pandas as pd


def in_notebook():
    """True inside Jupyter or Colab."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except Exception:
        return False


def show(table, max_colwidth=90):
    """Show a table (or any object)."""
    if in_notebook():
        from IPython.display import display
        with pd.option_context("display.max_colwidth", max_colwidth):
            display(table)
    elif hasattr(table, "to_string"):
        print(table.to_string(max_colwidth=max_colwidth))
    else:
        print(table)


def show_markdown(text):
    if in_notebook():
        from IPython.display import Markdown, display
        display(Markdown(text))
    else:
        print(text)


def show_image(path, width=420):
    if in_notebook():
        from IPython.display import Image, display
        display(Image(str(path), width=width))
    else:
        print(f"[picture: {path}]")


def finish_figure(fig, path):
    """Save a chart, then show it in a notebook (or just close it in a terminal)."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if in_notebook():
        plt.show()
    else:
        plt.close(fig)
