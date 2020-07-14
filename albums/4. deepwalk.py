from constants import SEED
import csv
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
import stellargraph as sg
from stellargraph.data import UniformRandomWalk
import time

nodes = pd.read_csv('albums_nodes.csv', index_col='id', quoting=csv.QUOTE_ALL)
edges = pd.read_csv('albums_edges.csv', index_col='id', quoting=csv.QUOTE_ALL)
graph = sg.StellarGraph(nodes=nodes.drop(['title', 'URL'], axis=1), edges=edges)

for window_size in [5, 10, 15]:
    for length in [40, 60, 80]:
        start_time = time.time()

        random_walker = UniformRandomWalk(graph, seed=SEED)
        random_walks = random_walker.run(nodes=list(graph.nodes()), length=length, n=25, seed=SEED)
        random_walks_str = [[str(n) for n in walk] for walk in random_walks]
        model = Word2Vec(random_walks_str, size=128, window=window_size, workers=8, min_count=0, sg=1, iter=1, hs=1,
                         seed=SEED)

        end_time = time.time()

        albums = pd.read_csv('albums_wikipedia.csv', index_col='id', quoting=csv.QUOTE_ALL)

        node_ids = [int(node_id) for node_id in model.wv.index2word]
        node_embeddings = model.wv.vectors

        albums_embeddings = pd.DataFrame({'node_id': node_ids}).merge(nodes, left_on='node_id', right_on='id').reindex(
            columns=['URL'])
        albums_embeddings['embedding'] = node_embeddings.tolist()
        albums_embeddings = albums_embeddings[albums_embeddings['URL'].isin(albums['URL'].tolist())]
        albums = albums.merge(albums_embeddings, left_on='URL', right_on='URL')

        albums.to_csv(f'./embeddings/albums_embeddings_deepwalk_{length}_{window_size}.csv', index_label='id',
                      quoting=csv.QUOTE_ALL)

        albums_labels = {label: index for index, label in enumerate(albums.label.unique())}
        albums_colours = albums.label.map(albums_labels)

        X = np.array(albums.embedding.tolist())
        y = albums_colours.to_numpy()

        logistic_classifier = LogisticRegression(multi_class='ovr', random_state=SEED)
        predicted = cross_val_predict(logistic_classifier, X, y, cv=10)
        accuracy = accuracy_score(y, predicted)

        print(
            f'window size: {window_size}, length: {length}, accuracy: {accuracy}, time: {end_time - start_time}')
