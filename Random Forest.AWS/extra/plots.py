import bokeh.charts
import bokeh.charts.utils
import bokeh.io 
import bokeh.models
import bokeh.palettes
import bokeh.plotting
# from bokeh.charts import color, marker
from bokeh.charts import Line, Bar, output_file, show
from bokeh.sampledata.autompg import autompg as df
from bokeh.plotting import *
# from bokeh.plotting import figure, curplot
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html

import numpy as np


# Define a function that will return an HTML snippet.
def build_plot():

    # Set the output for our plot.

    output_file('plot.html', title='Plot')

    # Create some data for our plot.

    x_data = np.arange(1, 101)
    y_data = np.random.randint(0, 101, 100)

    # Create a line plot from our data.

    Line(x_data, y_data)

    # Create an HTML snippet of our plot.
    # snippet = Line(x_data, y_data)
    snippet = file_html(embed_base_url='../static/js/', embed_save_loc='./static/js')
    # snippet = bokeh.embed_base_url(embed_base_url='../static/js/', embed_save_loc='./static/js')
    # snippet = curplot().create_html_snippet(embed_base_url='../static/js/', embed_save_loc='./static/js')

    # Return the snippet we want to place in our page.

    return snippet
