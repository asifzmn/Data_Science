import random
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from AQ.ObsoleteMethods import cmapPlotly

if __name__ == '__main__':

    for x in range(5):
        fig = go.Figure()
        df = pd.DataFrame([[3000, 6000, 9000, 12000, 15000], [.3, .6, .9, 1.2, 1.5], [.035, .045, .050, .055, .065],
                           ['sunset', 'Purpor', 'magenta', 'BuPu', 'dense']],
                          index=['count', 'mean', 'std', 'color']).T

        for i, row in df.iloc[::-1].iterrows():
            r, ran = np.random.normal(row['mean'], row['std'], row['count']), random.randint(-10, 10)
            theta = np.random.uniform(random.randint(-15, ran) * np.pi / 15, random.randint(ran, 15) * np.pi / 15,
                                      row['count'])
            x, y = r * np.cos(theta), r * np.sin(theta)

            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers',
                marker=dict(color=(np.power(x, 2) + np.power(y, 2)), colorscale=row['color'], line_width=0,
                            reversescale=False)
                # marker=dict(color=(np.power(x, 2) + np.power(y, 2)), colorscale=random.choice(cmapPlotly), line_width=1,reversescale=True)
            ))

        fig.update_layout(height=1000, width=1000, template='ggplot2', xaxis=dict(range=[-2, 2]),
                          yaxis=dict(range=[-2, 2]))
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        fig.show()
