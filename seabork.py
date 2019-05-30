import numpy as np
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from seaborn.distributions import _freedman_diaconis_bins as n_bins_calc
from scipy.special import erf
from scipy.stats import skew, skewtest, normaltest
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.smoothers_lowess import lowess

from bokeh.plotting import figure
from bokeh.models import LinearColorMapper, ColumnDataSource, ColorBar, FixedTicker, NumeralTickFormatter, Span
from bokeh.palettes import RdBu, Blues
from bokeh.models.tools import HoverTool


def correlation(df):

    # Create Matrix Data
    _corr_matrix = df.corr().round(3)
    factors = list(df.columns)

    mask = np.zeros_like(_corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    corr_matrix = _corr_matrix.mask(mask).T

    corr = corr_matrix.stack().to_frame('Correlation')
    corr.index.rename(['y', 'x'], inplace=True)
    corr.reset_index(inplace=True)

    # Create Figure
    colors = RdBu[11][::-1]
    mapper = LinearColorMapper(palette=colors, low=-1, high=1)
    source = ColumnDataSource(corr)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                     ticker=FixedTicker(ticks=np.linspace(-1,1,5)),
                     label_standoff=6, border_line_color=None, location=(0, 0))

    p = figure(title="Correlation Matrix",
               x_range=factors[1:],
               y_range=factors[::-1][1:],
               plot_width=800, plot_height=400,
               tools="save",
               y_axis_location = 'right',
               x_axis_location = 'above'
              )
    rect = p.rect(x='x', y='y',
           width=0.95, height=0.95,
           source=source,
           fill_color={'field':'Correlation', 'transform':mapper},
           name='rect'
          )

    text_props = {"source": source, "text_align": "center", "text_baseline": "middle"}
    r = p.text(x='x', y="y",
        text="Correlation",
        text_font_size="20pt",
        source=source,
        text_align="center",
        text_baseline="middle")
    r.glyph.text_font_style="bold"
    p.add_layout(color_bar, 'right')
    p.outline_line_color = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.visible = False


    TOOLTIPS = """
        <div style="width:250px">
            <div>
                <span style="font-size: 17px; font-weight: bold;">@Correlation{0.00}</span>
            </div>
            <div>
                <span style="font-weight: bold">@x</span>
            </div>
            <br>
            <div>
                <span style="font-weight: bold">@y</span>
            </div>
        </div>
                """
    hover = HoverTool(tooltips=TOOLTIPS, renderers=[rect])
    p.add_tools(hover)
    return p

def residuals_time(df):
    #Dataframe with datetime index and residuals as column "e"
    colors = Blues[9]+Blues[9][::-1]
    mapper = LinearColorMapper(palette=colors, low=df['e'].min(), high=df['e'].max())
    cds = ColumnDataSource(df)
    tools = "save, box_select, reset, box_zoom"
    p = figure(x_axis_type='datetime', tools=tools, plot_width=1200, plot_height=300)
    p.vbar(x="Date", top="e", width=timedelta(days=29), alpha=0.7, source=cds, fill_color={'field':'e', 'transform':mapper})

    hover=HoverTool(tooltips=[("Date","@Date{%Y-%b}"), ('Residual','@e{0,0}')],
    formatters={"Date":"datetime"}, mode='vline')
    return p

def kde(data, kernel='gau', bw='scott', gridsize=None, cut=3, clip=(-np.inf, np.inf),
                                cumulative=False):
    """Compute a univariate kernel density estimate using statsmodels."""
    fft = kernel == "gau"
    kde = KDEUnivariate(data)
    kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip)
    if cumulative:
        grid, y = kde.support, kde.cdf
    else:
        grid, y = kde.support, kde.density
    # Make sure the density is nonnegative
    y = np.amax(np.c_[np.zeros_like(y), y], axis=1)
    return grid, y


def distplot(e):
    x, y = kde(e)

    bins = n_bins_calc(e)
    hist, edges = np.histogram(e, density=True, bins=bins)

    p = figure(tools="save, box_select, reset",
        title='Residuals Distribution',
        plot_width=800, plot_height=400)
    p.patch(x, y, alpha=0.8, line_width=3, fill_alpha=0.2)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color=None, alpha=0.3)
    p.segment(x0=e, x1=e, y0=0, y1=np.mean(y)/4, line_width=2, alpha=0.8)
    p.yaxis.visible=False
    p.title.align = 'center'
    return p
