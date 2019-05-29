import seabork as sbk
import pandas as pd
import numpy as np
from bokeh.plotting import show

# If using Jupyter, uncomment these lines to output in notebook
#from bokeh.io import output_notebook
#output_notebook()

# Setup some example data
periods = 50
df1 = pd.DataFrame(np.random.rand(periods,6))
df1.columns =  ['a', 'b', 'c', 'd', 'e', 'f']

# Correlation Matrix
show(sbk.matrix_figure(df))

# Residuals over Time
df2 = pd.DataFrame({'e': np.random.randn(periods)}, 
                   index=pd.date_range(start='2015-01-01', 
                                       periods=periods, 
                                       freq='MS', 
                                       name='Date')
                  )

show(sbk.residuals_time(df2))

# Distribution Plot
show(sbk.distplot(df['e']))
