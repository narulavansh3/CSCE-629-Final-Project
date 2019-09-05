from queue import Queue
import enum
import graph
import sys

def get_maximum_spanning_tree(input_graph):
    N= len(input_graph.v)
    maximum_spanning_tree = graph.Graph()
    for i in range(0, N):
        maximum_spanning_tree.v.append(graph.Node(i))
    parent_array = [-1] * N
    rank_array = [0] * N
    for edge_source, edge_target, edge_weight in input_graph.sort_and_iterate_edges():
        r1 = find(edge_source, parent_array)
        r2 = find(edge_target, parent_array)
        if(r1 != r2):
            maximum_spanning_tree.connect_nodes(edge_source, edge_target, edge_weight)
            union(r1, r2, rank_array, parent_array)
    return maximum_spanning_tree

def union(rank1, rank2, rank_array, parent_array):
    if(rank_array[rank1] > rank_array[rank2]):
        parent_array[rank2] = rank1
    elif (rank_array[rank1] < rank_array[rank2]):
        parent_array[rank1] = rank2
    else:
        parent_array[rank1] = rank2
        rank_array[rank2] += 1

def find(v, parent_array):
    w = v
    q = Queue()
    while(parent_array[w] != -1):
        q.put(w)
        w = parent_array[w]
    while not q.empty():
        parent_array[q.get()] = w

    return w

class Color(enum.Enum):
    WHITE = 1
    GREY = 2
    BLACK = 3

def apply_dfs(graph, node_number, color_array, path_array, target):
    if (node_number == target):
        return True
    found = False
    color_array[node_number] = Color.GREY
    for edge in graph[node_number].adjacency_list:
        if(color_array[edge.target_number] == Color.WHITE):
            path_array[edge.target_number] = edge.source_number
            found = apply_dfs(graph, edge.target_number, color_array, path_array, target)
            if found:
                break
    color_array[node_number] = Color.BLACK
    return found

def get_maximum_bandwidth(maximum_spanning_tree, source, target, N):
    color_array = [Color.WHITE] * N 
    path_array  = [-1] * N

    apply_dfs(maximum_spanning_tree, source, color_array, path_array, target)

    path = str(target)
    k = target
    maximum_bandwith = sys.maxsize
    while(k != source):
        path = str(path_array[k]) + "->" + path
        maximum_bandwith= min(maximum_bandwith, maximum_spanning_tree.get(k, path_array[k]))
        k = path_array[k]
    return maximum_bandwith, path
