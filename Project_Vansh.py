
#%%
class Node:    
    def __init__(self, name):
        self.name = name
        self.adjacent_list_of_edges = []


#%%
class Edge:  
    def __init__(self,source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight



#%%
import random
def generate_graph(num_nodes, probability):
    graph = []
    n_edges = 0
    for i in range(num_nodes):
        graph.append(Node(str(i)))
    
    #making sure that graph is connected by making a cycle
    for i in range(num_nodes-1):
        initial_node = graph[i]
        target_node = graph[i+1]
        edge_weight = random.randint(1,10000)
        initial_node.adjacent_list_of_edges.append(Edge(initial_node,target_node, edge_weight))
        target_node.adjacent_list_of_edges.append(Edge(target_node, initial_node, edge_weight))
        n_edges+=1
    edge_weight = random.randint(1,10000)
    graph[num_nodes-1].adjacent_list_of_edges.append(Edge(graph[num_nodes-1], graph[0], edge_weight))
    graph[0].adjacent_list_of_edges.append(Edge(graph[0], graph[num_nodes-1], edge_weight))
    n_edges+=1
    
    #randomly adding other nodes in the graph
    for i in range(num_nodes):
        initial_node = graph[i]
        for j in range(i+1,num_nodes):
            if(random.randint(0,num_nodes) < (probability-1/num_nodes) * num_nodes):
                target_node = graph[j]
                edge_weight = random.randint(1,10000)
                initial_node.adjacent_list_of_edges.append(Edge(initial_node, target_node, edge_weight))
                target_node.adjacent_list_of_edges.append(Edge(target_node, initial_node, edge_weight))
                n_edges+=1
        
    return graph, n_edges     


#%%
sparse_graph = generate_graph(5000, float(6/5000))


#%%
dense_graph = generate_graph(5000, 0.20)


#%%
def average_number_of_edges_per_node(graph):
    total_edge = 0
    for i in range(len(graph)):
        total_edge+= len(graph[i].adjacent_list_of_edges)

        
    return(total_edge/len(graph))


#%%
def verify_graph(graph):
    for i in range(len(graph)):
        test =[]
        for j in graph[i].adjacent_list_of_edges:
            if j not in test:
                test.append(j)
            else:
                print('error in', i, 'row', j, 'element')
                
    print('Verified')     
                
            
    


#%%
#

verify_graph(dense_graph[0])


#%%
average_number_of_edges_per_node(sparse_graph[0])

#%%
import math
import sys
class heap:
    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = [-1] * self.max_size
        self.D = [-1]* self.max_size
        self.node_to_heap_index = [-1] * self.max_size
        self.size = 0
        
    def Parent(self, i):
        if i%2 !=0:
            return math.floor((i-1)/2)
        else :
            return math.floor((i-2)/2)
    
    def Left_Child(self, i):
        return (2 * i + 1) if (2* i + 1)< self.size else -1
    
    def Right_Child(self, i):
        return (2 * i + 2) if (2* i + 2)< self.size else -1

    def Swap(self,i,k):

        self.heap[k],self.heap[i] = self.heap[i], self.heap[k]
        self.node_to_heap_index[self.heap[k]], self.node_to_heap_index[self.heap[i]] = self.node_to_heap_index[self.heap[i]], self.node_to_heap_index[self.heap[k]]
    
    
    def Max(self):
        return self.heap[0]
    
    def Fix_Heap(self,i):
        L = self.Left_Child(i)
        R = self.Right_Child(i)
        maximum = i
        if L != -1:
            if L<= self.size-1 and self.D[self.heap[i]]< self.D[self.heap[L]]:
                maximum = L
            else:
                maximum = i
        if R!= -1:

            if R<= self.size-1 and self.D[self.heap[maximum]]< self.D[self.heap[R]]:
                maximum = R
            
        if maximum != i:
            self.Swap(i,maximum)
            self.Fix_Heap(maximum) 
    
    def Extract_Max(self):
        if self.size < 1:
            print('error')
            
        max_element = self.heap[0], self.D[self.heap[0]]
        self.Swap(0, self.size-1)
        self.size= self.size - 1
        self.Fix_Heap(0)
        return max_element
    
    def Insert(self, a, BW):
        if self.size>= self.max_size:
            print('error is here')
        self.heap[self.size] = int(a.name)
        self.D[int(a.name)] = BW
        i = self.size
        self.size = self.size+1

        
        self.node_to_heap_index[int(a.name)] = i
        while(i>0 and (self.D[self.heap[self.Parent(i)]] < self.D[self.heap[i]])):
            self.Swap(i,self.Parent(i))
            i = self.Parent(i)

    def reset_node(self, node_number, value):
        index = self.node_to_heap_index[node_number]
        if(index == -1):
            raise IndexError('Node not in Heap')
        self.D[self.heap[index]] = value
        parent = self.Parent(index)
        while(self.D[self.heap[parent]]< self.D[self.heap[index]]):
            self.Swap(index, parent)
            index = parent
            if(parent == 0):
                break
            parent = parent = self.Parent(index)

        self.Fix_Heap(index)

    def Delete(self, i):
        if i> self.size:
            print('error in there')
        self.D[self.heap[i]] =  sys.maxsize
        while(i>0 and (self.D[self.heap[self.Parent(i)]] < self.D[self.heap[i]])):
            self.Swap(i, self.Parent(i))
            i = self.Parent(i)
        
        self.heap[0] = self.heap[self.size-1]
        self.node_to_heap_index[self.heap[0]] = 0
        self.heap[self.size-1] = -1
        self.size= self.size - 1
        
        self.Fix_Heap(0)

    def print_heap(self, node = 0, tab = ''):
        if(node == -1):
            return
        print(tab,str(self.D[self.heap[node]]))
        self.print_heap(self.Left_Child(node),  tab + '\t')
        self.print_heap(self.Right_Child(node), tab + '\t')





#%%
class Dijkstra:
    def __init__(self, Graph, Source, target):
        self.G = Graph
        self.S = Source
        self.T = target
        

    def apply_With_Heap(self):
        BW = [-1] * len(self.G)
        status = [-1] * len(self.G)
        dad = [-1] * len(self.G)
        self.max_heap = heap(len(self.G))
        for node in self.G:
            status[int(node.name)] = 0  # 0 --> Unseen

        status[int(self.S.name)] = 2  # 2 --> Intree
        BW[int(self.S.name)] = float('inf')
        

        for edge in self.G[int(self.S.name)].adjacent_list_of_edges:
            status[int(edge.target.name)] = 1  # 1 --> Fringe
            BW[int(edge.target.name)] = int(edge.weight)
            self.max_heap.Insert(edge.target, BW[int(edge.target.name)])
            dad[int(edge.target.name)] = int(self.S.name)

        while (self.max_heap.size != 0):
            max_element = self.max_heap.Extract_Max()
            current_max = max_element[0]
            status[current_max] = 2
            if current_max == int(self.T.name):
                return dad, BW[current_max]
            for edge in self.G[current_max].adjacent_list_of_edges:
                w = int(edge.target.name)
                if status[w] == 0:
                    status[w] = 1
                    BW[w] = min(BW[current_max], int(edge.weight))

                    self.max_heap.Insert(edge.target, BW[w])

                    dad[w] = current_max

                elif (status[w] == 1 and BW[w] < min(BW[current_max], int(edge.weight)) ):
                    dad[w] = current_max
                    BW[w] = min(BW[current_max], int(edge.weight))
                    self.max_heap.reset_node(w, BW[w])
                    

    def max_BW_fringe(self, graph, status_array, bandwidth_array, N):
        index_of_max_bw_fringe = -1
        max_fringe_bw = -sys.maxsize - 1
        for i in range(0, N):
            if((status_array[i] == 1) and (bandwidth_array[i] >= max_fringe_bw)):
                index_of_max_bw_fringe = i
                max_fringe_bw = bandwidth_array[i]
        return index_of_max_bw_fringe


    def apply_Without_Heap(self):
        
        fringes_count = 0
        BW = [-1] * len(self.G)
        status = [-1] * len(self.G)
        dad = [-1] * len(self.G)


        for node in self.G:
            status[int(node.name)] = 0  # 0 --> Unseen

        status[int(self.S.name)] = 2  # 2 --> Intree
        BW[int(self.S.name)] = float('inf')

        for edge in self.G[int(self.S.name)].adjacent_list_of_edges:
            BW[int(edge.target.name)] = int(edge.weight)
            status[int(edge.target.name)] = 1  # 1 --> Fringe
            dad[int(edge.target.name)] = int(self.S.name)
            fringes_count += 1

        maximum_bw = sys.maxsize
        while(fringes_count > 0):
            current_max = self.max_BW_fringe(self.G, status, BW, len(self.G))
            status[current_max] = 2
            if(current_max == int(self.T.name)):
                maximum_bw = BW[current_max]
                return maximum_bw, dad

            fringes_count -= 1

            for edge in self.G[current_max].adjacent_list_of_edges:
                w = int(edge.target.name)
                if(status[w] == 0):
                    status[w] = 1
                    fringes_count += 1
                    dad[w] = current_max
                    BW[w] =  min(BW[current_max], int(edge.weight))
                elif(status[w] == 1 and BW[w] < min(BW[current_max], int(edge.weight))):
                    dad[w] = current_max
                    BW[w] = min(BW[current_max], int(edge.weight))
    

#%%
class EdgeHeap:
    def __init__(self, max_size):
        self.size = 0
        self.max_size  = max_size
        self.S = [-1] * max_size
        self.T = [-1] * max_size
        self.D = [-1] * max_size

    def Insert(self, source, target, value):
        if(self.size == self.max_size):
            raise OverflowError('Overflow error: Trying to add element in heap with full capacity')
        self.S[self.size] = source
        self.T[self.size] = target
        self.D[self.size] = value
        self.size += 1

        k = self.size - 1
        while k>0 and self.D[k] > self.D[self.Parent(k)]:
            self.swap(k, self.Parent(k))
            #self.Fix_Heap(k)
            k = self.Parent(k)
            
    def Left_Child(self, i):
        return (2 * i + 1) if (2* i + 1)< self.size else -1
    
    def Right_Child(self, i):
        return (2 * i + 2) if (2* i + 2)< self.size else -1

    def Parent(self, i):
        if  (i>=self.size) or (((i-1)/2) >= self.size):
            raise ValueError('Index out of bounds of the heap')
        return int((i-1)/2)
    
    def extract_max(self):
        element_to_remove = self.S[0], self.T[0], self.D[0]
        self.swap(0, self.size-1)
        self.size -= 1
        self.Fix_Heap(0)
        return element_to_remove

    def swap(self, i, j):
        self.S[i], self.S[j] = self.S[j], self.S[i]
        self.T[i], self.T[j] = self.T[j], self.T[i]
        self.D[i], self.D[j] = self.D[j], self.D[i]

    def Fix_Heap(self, i):
        l = self.Left_Child(i)
        r = self.Right_Child(i)
        maximum = i

        if l != -1:
            if l< self.size and self.D[l] > self.D[i]:
                maximum = l
            else:
                maximum = i

        if r!= -1:
            if r<= self.size and self.D[r] > self.D[maximum]:
                maximum = r
            else:
                maximum = maximum
        
        if maximum != i:
            self.swap(i, maximum)
            self.Fix_Heap(maximum)

        #self.print_heap()
    
    def get_max_element(self):
        if self.size == 0:
            return None, None
        else:
            return self.S[0],self.T[0], self.D[0]

    def print_heap(self, node = 0, tab = ''):
        if(node == -1):
            return
        print(tab,str(self.D[node]))
        self.print_heap(self.Left_Child(node),  tab + '\t')
        self.print_heap(self.Right_Child(node), tab + '\t')

#%% 
from queue import Queue
import sys
class Kruskal:
    def __init__(self, Graph, Source, target, n_edges):
        self.G = Graph
        self.S = Source
        self.T = target
        self.num_of_edges = n_edges

    def sort_and_iterate_edges(self):
        edge_heap = EdgeHeap(self.num_of_edges)
        for i in range(0, len(self.G)):
            for edge in self.G[i].adjacent_list_of_edges:
                if(int(edge.source.name) < int(edge.target.name)):
                    edge_heap.Insert(int(edge.source.name), int(edge.target.name), int(edge.weight))
        for _ in range(0, self.num_of_edges):
            yield edge_heap.extract_max()

    def get_maximum_spanning_tree(self):
        N= len(self.G)
        maximum_spanning_tree = []
        for i in range(0, N):
            maximum_spanning_tree.append(Node(i))
        dad_array = [-1] * N
        rank_array = [0] * N
        for edge_source, edge_target, edge_weight in self.sort_and_iterate_edges():
            r1 = self.find(edge_source, dad_array)
            r2 = self.find(edge_target, dad_array)
            if(r1 != r2):
                maximum_spanning_tree[edge_source].adjacent_list_of_edges.append(Edge(maximum_spanning_tree[edge_source], maximum_spanning_tree[edge_target], int(edge_weight)))
                maximum_spanning_tree[edge_target].adjacent_list_of_edges.append(Edge(maximum_spanning_tree[edge_target], maximum_spanning_tree[edge_source], int(edge_weight)))
                self.union(r1, r2, rank_array, dad_array)
        return maximum_spanning_tree

    def union(self, rank1, rank2, rank_array, dad_array):
        if(rank_array[rank1] > rank_array[rank2]):
            dad_array[rank2] = rank1
        elif (rank_array[rank1] < rank_array[rank2]):
            dad_array[rank1] = rank2
        else:
            dad_array[rank1] = rank2
            rank_array[rank2] += 1

    def find(self, v, dad_array):
        w = v
        q = Queue()
        while(dad_array[w] != -1):
            q.put(w)
            w = dad_array[w]
        while not q.empty():
            dad_array[q.get()] = w

        return w

    def get(self,maximum_spanning_tree, i, j):
        for edge in maximum_spanning_tree[i].adjacent_list_of_edges:
            if(int(edge.target.name) == j):
                return int(edge.weight)
        return -1

    def apply_dfs(self, graph, node_number, color_array, path_array, target):
        if (node_number == target):
            return True
        found = False
        color_array[node_number] = 2
        for edge in graph[node_number].adjacent_list_of_edges:
            if(color_array[int(edge.target.name)] == 1):
                path_array[int(edge.target.name)] = int(edge.source.name)
                found = self.apply_dfs(graph, int(edge.target.name), color_array, path_array, target)
                if found:
                    break
        color_array[node_number] = 3
        return found

    def get_maximum_bandwidth(self, maximum_spanning_tree, source, target, N):
        color_array = [1] * N  # 1 -->White
        path_array  = [-1] * N

        self.apply_dfs(maximum_spanning_tree, source, color_array, path_array, target)

        path = str(target)
        k = target
        maximum_bandwith = sys.maxsize
        while(k != source):
            path = str(path_array[k]) + "->" + path
            maximum_bandwith= min(maximum_bandwith, self.get(maximum_spanning_tree, k, path_array[k]))
            k = path_array[k]
        return maximum_bandwith, path


    def apply_Kruskal(self):

        maximum_spanning_tree = self.get_maximum_spanning_tree()
        max_BW, path = self.get_maximum_bandwidth(maximum_spanning_tree, self.S, self.T, len(self.G))
        return max_BW, path



#%%
def get_random_ST_pair(graph):
    source = graph[random.randint(1,len(graph)-1)]
    target = graph[random.randint(1,len(graph)-1)]
    while (source == target):
        source = graph[random.randint(1,len(graph)-1)]
        target = graph[random.randint(1,len(graph)-1)]
    
    return source,target



#%%
import time
def apply_different_algorithms(graph, n_times):
    for _ in range(n_times):
        source,target = get_random_ST_pair(graph[0])
        MWP = Dijkstra(graph[0], source, target)
        MBW = Kruskal(graph[0], int(source.name), int(target.name), graph[1])
        start_with_heap=time.time()
        value = MWP.apply_With_Heap()
        end_with_heap = time.time()

        start_without_heap = time.time()
        value2 = MWP.apply_Without_Heap()
        end_without_heap = time.time()

        start_kruskal = time.time()
        value3 = MBW.apply_Kruskal()
        end_kruskal = time.time()

        path = str(target.name)
        k = int(target.name)
        while(k != int(source.name)):
                    path = str(value[0][k]) + "->" + path    
                    k = value[0][k]

        print(path)
        print('max BW with heap=', value[1])
        print('time taken with heap=', (end_with_heap-start_with_heap))

        without_heap_path = str(target.name)
        k1 = int(target.name)
        while(k1 != int(source.name)):
                    without_heap_path = str(value2[1][k1]) + "->" + without_heap_path    
                    k1 = value2[1][k1]

        print(without_heap_path)
        print('Max BW without heap =', value2[0])
        print('time taken without heap=', (end_without_heap-start_without_heap))


        print(value3[1])
        print('max BW path using Kruskal=', value3[0])
        print('time taken with kruskal=', (end_kruskal-start_kruskal))

#%%
for i in range(5):
    sparse_graph = generate_graph(5000, float(6/5000))
    print('applying on sparse graph---------------------------------------------------- /n -----------------------------------------')
    apply_different_algorithms(sparse_graph,5)
    dense_graph = generate_graph(5000, 0.20)
    print('applying on dense graph---------------------------------------------------- /n -----------------------------------------')
    apply_different_algorithms(dense_graph,5)

#%%
