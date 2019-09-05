import sys
#%% Test Function
def test_graph(N, graph, source, target):
    if N<20:
        max_bandwidth_identifiers = [-sys.maxsize -1, []] 
        def go(v, bw, graph, target, visited_nodes, max_bandwidth_identifiers):
            visited_nodes.append(v)
            if(v == target):
                if (max_bandwidth_identifiers[0] < bw):
                    max_bandwidth_identifiers[0] = bw
                    max_bandwidth_identifiers[1].clear()
                    max_bandwidth_identifiers[1].append(visited_nodes)
                elif(max_bandwidth_identifiers[0] == bw):
                    max_bandwidth_identifiers[0] = bw
                    max_bandwidth_identifiers[1].append(visited_nodes)
                return
            for edge in graph[v].adjacency_list:
                if edge.target_number not in visited_nodes:
                    go(edge.target_number, min(edge.weight, bw), graph, target, visited_nodes.copy(), max_bandwidth_identifiers)
        visited_nodes = []
        go(source, sys.maxsize, graph, target, visited_nodes, max_bandwidth_identifiers)
        print(max_bandwidth_identifiers)
