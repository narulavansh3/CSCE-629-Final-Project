
#%%
from enum import Enum
import heap
import sys

class Status(Enum):
    UNSEEN = 1
    FRINGE = 2
    INTREE = 3


def get_fringe_with_max_bw(graph, status_array, bandwidth_array, N):
    index_of_max_bw_fringe = -1
    max_fringe_bw = -sys.maxsize - 1
    for i in range(0, N):
        if((status_array[i] == Status.FRINGE) and (bandwidth_array[i] >= max_fringe_bw)):
            index_of_max_bw_fringe = i
            max_fringe_bw = bandwidth_array[i]
    return index_of_max_bw_fringe

#%%
def djkistra_without_heap(graph, source, target, N):
    if not graph.contains_node(source) or not graph.contains_node(target):
        raise IndexError('Index of node out of scope of graph length')

    fringes_count = 0
    status_array    = [Status.UNSEEN] * N
    bandwidth_array = [sys.maxsize] * N
    dad_array       = [-1] * N
    status_array[source] = Status.INTREE

    for edge in graph[source].adjacency_list:
        status_array[edge.target_number]    = Status.FRINGE
        dad_array[edge.target_number] = source
        bandwidth_array[edge.target_number] = edge.weight
        fringes_count += 1

    maximum_bw = sys.maxsize
    while(fringes_count > 0):
        v = get_fringe_with_max_bw(graph, status_array, bandwidth_array, N)
        status_array[v] = Status.INTREE
        if(v == target):
            maximum_bw = bandwidth_array[target]
            break
        fringes_count -= 1

        for edge in graph[v].adjacency_list:
            w = edge.target_number
            if(status_array[w] == Status.UNSEEN):
                status_array[w] = Status.FRINGE
                fringes_count += 1
                dad_array[w] = v
                bandwidth_array[w] = min(bandwidth_array[v], edge.weight)
            elif((status_array[w] == Status.FRINGE) and (bandwidth_array[w] <= min(bandwidth_array[v], edge.weight))):
                dad_array[w] = v
                bandwidth_array[w] = min(bandwidth_array[v], edge.weight)

    return maximum_bw, dad_array

#%% 
def read_path(array, source, target):
    path = str(target)
    k = target
    while(k != source):
        path = str(array[k]) + "->" + path
        k = array[k]
    return path

#%%
def djkistra_with_heap(graph, source, target, N):
    if not graph.contains_node(source) or not graph.contains_node(target):
        raise IndexError('Index of node out of scope of graph length')
    heap_of_fringes = heap.FringeHeap(N)

    fringes_count = 0
    status_array    = [Status.UNSEEN] * N
    bandwidth_array = [sys.maxsize] * N
    dad_array       = [-1] * N
    status_array[source] = Status.INTREE

    for edge in graph[source].adjacency_list:
        status_array[edge.target_number]    = Status.FRINGE
        bandwidth_array[edge.target_number] = edge.weight
        dad_array[edge.target_number] = source
        heap_of_fringes.push(edge.target_number, bandwidth_array[edge.target_number])
        fringes_count += 1

    maximum_bw = sys.maxsize
    while(fringes_count > 0):
        v,_ = heap_of_fringes.pop()
        status_array[v] = Status.INTREE
        if(v == target):
            maximum_bw = bandwidth_array[target]
            break
        fringes_count -= 1

        for edge in graph[v].adjacency_list:
            w = edge.target_number
            if(status_array[w] == Status.UNSEEN):
                status_array[w] = Status.FRINGE
                fringes_count += 1
                dad_array[w] = v
                bandwidth_array[w] = min(bandwidth_array[v], edge.weight)
                heap_of_fringes.push(w, bandwidth_array[w])
            elif((status_array[w] == Status.FRINGE) and (bandwidth_array[w] < min(bandwidth_array[v], edge.weight))):
                dad_array[w] = v
                bandwidth_array[w] = min(bandwidth_array[v], edge.weight)
                heap_of_fringes.reset_node(w, bandwidth_array[w])
    return maximum_bw, dad_array

