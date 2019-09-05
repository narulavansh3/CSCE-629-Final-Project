#%%
import graph
import random
import sys
import djkistra
import heap
import kruskals
import time

#%% [markdown]
## Generate the graphs

#%%
N = 5000
time_array = {'DjKistras_without_heap':{},'DjKistras_without_heap':{}, 'Kruskals':{}}

#generate the graphs            
dense_graph  = graph.Graph.generate_graph(N, 20/N)
sparse_graph = graph.Graph.generate_graph(N, 6/N)

#%%
source = random.randint(0, N-1) 
target = random.randint(0, N-1)
while source == target:
    source = random.randint(0, N-1) 
    target = random.randint(0, N-1)
print("Source: ", source)
print("Target: ", target)

#%% [markdown]
## Djkistra's Algorithm without heap

#%%
start = time.time()
maximum_bandwidth, dad_array = djkistra.djkistra_without_heap(dense_graph, source, target, N)
end = time.time()
print("Maximum bandwidth with Djkistra without heap: ",maximum_bandwidth)
print("Path: ", djkistra.read_path(dad_array, source, target))
print('Time taken: ', end-start)
#%% [markdown]
## Djkistra's Algorithm with heap

#%%
start = time.time()
maximum_bandwidth, dad_array = djkistra.djkistra_with_heap(dense_graph, source, target, N)
end = time.time()
print("Maximum bandwidth with Djkistra with heap: ",maximum_bandwidth)
print("Path: ", djkistra.read_path(dad_array, source, target))
print('Time taken: ', end-start)

#%% [markdown]
## Kruskal's Algorithm

#%%
start = time.time()
maximum_spanning_tree = kruskals.get_maximum_spanning_tree(dense_graph)
end = time.time()
maximum_bandwidth, path = kruskals.get_maximum_bandwidth(maximum_spanning_tree, source, target, N)

print("Maximum bandwidth with Kruskal's algorithm : ",maximum_bandwidth)
print("Path: ", path)
print('Time taken: ', end - start)