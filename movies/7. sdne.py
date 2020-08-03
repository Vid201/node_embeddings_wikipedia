from constants import SEEDS
import csv
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
import sys
import tensorflow as tf
import time

sys.path.append('../GraphEmbedding')
from ge import SDNE

edges = pd.read_csv('movies_edges.csv', index_col='id', quoting=csv.QUOTE_ALL)
edges.to_csv('movies_edges_no_index.csv', header=False, index=False, sep=' ')
graph = nx.read_edgelist('movies_edges_no_index.csv', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

for SEED in SEEDS:
    tf.set_random_seed(SEED)
    np.random.seed(SEED)

    for layer in [[128], [256, 128], [512, 128]]:
        for beta in [25, 150, 300]:
            nodes = pd.read_csv('movies_nodes.csv', index_col='id', quoting=csv.QUOTE_ALL)
            movies = pd.read_csv('movies_wikipedia.csv', index_col='id', quoting=csv.QUOTE_ALL)

            start_time = time.time()

            model = SDNE(graph, hidden_size=layer, beta=beta)
            model.train(batch_size=500, epochs=25, verbose=0)

            end_time = time.time()

            embeddings = {int(key): val for key, val in model.get_embeddings().items()}
            nodes.loc[:, 'embedding'] = nodes.index.map(embeddings)
            nodes = nodes[nodes['URL'].isin(movies['URL'].tolist())]
            nodes = nodes.drop(['title'], axis=1)
            movies = movies.merge(nodes, left_on='URL', right_on='URL')

            movies.to_csv(f'./embeddings/movies_embeddings_sdne_{layer}_{beta}_{SEED}.csv', index_label='id',
                          quoting=csv.QUOTE_ALL)

            movies_labels = {label: index for index, label in enumerate(movies.label.unique())}
            movies_colours = movies.label.map(movies_labels)

            X = np.array(movies.embedding.tolist())
            y = movies_colours.to_numpy()

            logistic_classifier = LogisticRegression(multi_class='ovr', random_state=SEED)
            predicted = cross_val_predict(logistic_classifier, X, y, cv=10)
            accuracy = accuracy_score(y, predicted)

            print(
                f'seed: {SEED}, layers: {layer}, beta: {beta}, accuracy: {accuracy}, time: {end_time - start_time}')
