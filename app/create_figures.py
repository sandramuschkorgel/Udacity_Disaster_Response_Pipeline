import pandas as pd
import plotly.graph_objs as go


def return_figures(df):
    """
    Return three plotly visualizations.

    ARGUMENTS:
    df - cleaned dataframe

    RETURN VALUE:
    figures - list containing the  three plotly visualizations
    """

    # Figure 1
    graph_one = []

    plot_1 = df.groupby('genre').message.count().sort_values(ascending=False)

    graph_one.append(
        go.Bar(
            x = plot_1.index,
            y = plot_1
        )
    )

    layout_one = dict(title='Distribution of Message Genres',  
                      xaxis=dict(title='Genre'),
                      yaxis=dict(title='Count'))
    
    # Figure 2
    graph_two = []

    plot_2 = df[df.columns[4:]].sum().sort_values(ascending=False)

    graph_two.append(
        go.Bar(
            x = plot_2.index,
            y = plot_2
        )
    )

    layout_two = dict(title='Number of Messages per Category',
                      yaxis=dict(title='Count'))

    # Figure 3
    graph_three = []

    df_sub = pd.DataFrame()
    df_sub['disaster_category_count'] = df[df.columns[4:]].sum(axis=1)
    plot_3 = df_sub['disaster_category_count'].value_counts()

    graph_three.append(
        go.Bar(
            x = plot_3.index,
            y = plot_3
        )
    )

    layout_three = dict(title='Messages with Overlapping Categories',
                        xaxis=dict(title='Number of categories assigned to a message',
                                   tickmode='linear',
                                   tick0=0,
                                   dtick=1),
                        yaxis=dict(title='Count'))

    # Create list and append all figures
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))

    return figures
