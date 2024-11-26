import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np



def format_as_percentage(val):
    return f'{val* 100:.1f}'


def format_as_float(val):
    if val is None:
        return ""
    return f'{val:.2f}'


def table2latex(
    table: pd.DataFrame,
    label: str,
    caption: str,
    fmtter: callable=format_as_percentage,
    normalize=(0,1),
    cmap="Greens",
):    
    def color_cell(val, cmap, norm):
        color = cmap(norm(val))
        return f'\\cellcolor[HTML]{{{mcolors.rgb2hex(color)[1:]}}}{{{fmtter(val)}}}'
    
    # Normalize the data and create a colormap
    norm = plt.Normalize(*normalize)
    cmap = plt.get_cmap(cmap)
    
    table_colored = table.applymap(lambda x: color_cell(x, cmap, norm))

    return table_colored.to_latex(
        caption=caption,
        label=f"tab:{label}",
        header=[c.replace("_", " ") for c in table.columns],
    )
