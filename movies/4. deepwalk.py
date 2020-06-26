import csv
from gensim.models import Word2Vec
from movies.constants import SEED
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import stellargraph as sg
from stellargraph.data import UniformRandomWalk
import time

nodes = pd.read_csv('movies_nodes.csv', index_col='id', quoting=csv.QUOTE_ALL)
edges = pd.read_csv('movies_edges.csv', index_col='id', quoting=csv.QUOTE_ALL)
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

        movies = pd.read_csv('movies_wikipedia.csv', index_col='id', quoting=csv.QUOTE_ALL)

        node_ids = [int(node_id) for node_id in model.wv.index2word]
        node_embeddings = model.wv.vectors

        movies_embeddings = pd.DataFrame({'node_id': node_ids}).merge(nodes, left_on='node_id', right_on='id').reindex(
            columns=['URL'])
        movies_embeddings['embedding'] = node_embeddings.tolist()
        movies_embeddings = movies_embeddings[movies_embeddings['URL'].isin(movies['URL'].tolist())]
        movies = movies.merge(movies_embeddings, left_on='URL', right_on='URL')

        movies.to_csv(f'movies_embeddings_deepwalk_{window_size}_{length}.csv', index_label='id', quoting=csv.QUOTE_ALL)

        movies_labels = {label: index for index, label in enumerate(movies.label.unique())}
        movies_colours = movies.label.map(movies_labels)

        X_train, X_test, y_train, y_test = train_test_split(np.array(movies.embedding.tolist()),
                                                            movies_colours.to_numpy(), train_size=0.75,
                                                            random_state=SEED)
        logistic_classifier = LogisticRegressionCV(cv=10, scoring='accuracy', multi_class='ovr', max_iter=300,
                                                   random_state=SEED)
        logistic_classifier.fit(X_train, y_train)

        y_pred = logistic_classifier.predict(X_test)
        print(
            f'window size: {window_size}, length: {length}, accuracy: {accuracy_score(y_test, y_pred)}, time: {end_time - start_time}')
