import random as rand
import heap 

class Node:
    def __init__(self, name_of_node):
        self.name_of_node = name_of_node
        self.adjacency_list = []


class Edge:
    def __init__(self, source_number, target_number, weight):
        self.source_number = source_number
        self.target_number = target_number
        self.weight = weight

    def __lt__(self, other):
        return self.weight < other.weight

    def __gt__(self, other):
        return self.weight > other.weight

    @staticmethod
    def generate_positive_weight():
        return rand.randint(1, 0XFFFF)

class Graph:
    def __init__(self):
        self.v = []
        self.num_of_edges = 0

    def connect_nodes(self, i, j, edge_weight):
        self.num_of_edges += 1
        self.v[i].adjacency_list.append(Edge(i, j, edge_weight))
        self.v[j].adjacency_list.append(Edge(j, i, edge_weight))

    @staticmethod
    def generate_graph(N, probablility_of_connection):
        g = Graph()
        if(probablility_of_connection * N <1):
            raise ValueError('Probablity should be enough for atleast one node to connect')
        
        #Create a empty graph
        for i in range(0, N):
            node = Node(i)            
            g.v.append(node)
       
        for i in range(0, N):
            #Connect every node compulsorily to its next node
            g.connect_nodes(i, (i+1)%N, Edge.generate_positive_weight())
            for neighbour in range(i+2, N):
                #Consider evey pair of vertices only once
                chance = rand.random() 
                if((chance * N) <= (probablility_of_connection * N - 1)):
                    g.connect_nodes(i, neighbour, Edge.generate_positive_weight())
        return g

    def print_graph(self):
        for vertex in self.v:
            print(f"Vertex {vertex.name_of_node} has neighbours.....")
            for edge in vertex.adjacency_list:
                print(edge.target_number)

    def cout_avg_num_edges(self):
        sum = 0.0
        for vertex in self.v:
            sum += len(vertex.adjacency_list)
        return sum/len(self.v)

    def get(self, i, j):
        for edge in self.v[i].adjacency_list:
            if(edge.target_number == j):
                return edge.weight
        return -1

    def contains_node(self, node_number):
        return node_number < len(self.v)

    def __getitem__(self, key):
        return self.v[key]

    def sort_and_iterate_edges(self):
        edge_heap = heap.EdgeHeap(self.num_of_edges)
        for i in range(0, len(self.v)):
            for edge in self[i].adjacency_list:
                if(edge.source_number < edge.target_number):
                    edge_heap.push(edge.source_number, edge.target_number, edge.weight)
        for _ in range(0, self.num_of_edges):
            yield edge_heap.pop()
