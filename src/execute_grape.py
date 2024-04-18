import sys
import glob

import pandas as pd
from grape import Graph
from grape.embedders import Node2VecSkipGramEnsmallen

embed_path = '/scratch/fl1092/COIpaper/expertise/grape/embedding/' # directory to store embedding
GRAPH_NAME = sys.argv[1]
EPOCH = int(sys.argv[2])

PROJDIR = '/scratch/fl1092/COIpaper/' # directory for the COI paper specifically
edge_path = PROJDIR + f'expertise/grape/{GRAPH_NAME}.csv'

print('Graph name: ', GRAPH_NAME, flush=True)
print('Edge file:', glob.glob(edge_path), flush=True)
print('Epoch:', EPOCH, flush=True)


graph = Graph.from_csv(
    # Edges related parameters
    ## The path to the edges list tsv
    edge_path=edge_path,
    ## Set the comma as the separator between values
    edge_list_separator=",",
    ## if there is a header of not
    edge_list_header=True,
    ## The source nodes are in the first nodes
    sources_column_number=0,
    ## The destination nodes are in the second column
    destinations_column_number=1,
    ## The weights are in the third column
    #weights_column_number=2,

    directed=False,
    name=GRAPH_NAME,
    verbose=True,
)

print("Number of nodes:", graph.get_number_of_nodes(), flush=True)
print("Number of edges:", graph.get_number_of_edges(), flush=True)

graph.enable(vector_sources = True, vector_reciprocal_sqrt_degrees = True)

embedding = Node2VecSkipGramEnsmallen(
    epochs=EPOCH,
    max_neighbours = 50,
    enable_cache = False).fit_transform(graph)

print("Done:", embedding.embedding_method_name, flush=True)

central, context = embedding.get_all_node_embedding()

central.to_csv(embed_path + f'{GRAPH_NAME}_{EPOCH}_central.csv')
context.to_csv(embed_path + f'{GRAPH_NAME}_{EPOCH}_contextual.csv')

print("Method:", embedding.embedding_method_name, flush=True)
