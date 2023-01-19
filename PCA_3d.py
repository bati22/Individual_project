import plotly.graph_objects as go


import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from generate_similar_words import *
import plotly.express as px

def display_pca_scatterplot_2D(model, user_input=None, words=None, label=None, color_map=None, topn=20, sample=10):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [word for word in model.vocab]

    word_vectors = np.array([model[w] for w in words])

    two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0

    for i in range(len(user_input)):
        trace = go.Scatter(
            x=two_dim[count:count + topn, 0],
            y=two_dim[count:count + topn, 1],
            text=words[count:count + topn],
            name=user_input[i],
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )

        data.append(trace)
        count = count + topn

    trace_input = go.Scatter(
        x=two_dim[count:, 0],
        y=two_dim[count:, 1],
        text=words[count:],
        name='input words',
        textposition="top center",
        textfont_size=20,
        mode='markers+text',
        marker={
            'size': 10,
            'opacity': 1,
            'color': 'black'
        }
    )

    data.append(trace_input)

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.write_image("figure.png", engine="kaleido")
    plot_figure.show()
