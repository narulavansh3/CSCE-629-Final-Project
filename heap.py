class FringeHeap:
    def __init__(self, capacity):
        self.size = 0
        self.H = [-1] * capacity
        self.K = [-1] * capacity
        self.D = [-1] * capacity

    def push(self, node, value):
        self.H[self.size] = node
        self.D[self.size] = value
        self.K[node] = self.size
        self.size += 1

        k = self.size - 1
        while k>0 and self.D[k] > self.D[self.get_parent(k)]:
            self.swap(k, self.get_parent(k))
            #self.max_heapify(k)
            k = self.get_parent(k)
            
    def get_left_child(self, i):
        return (2 * i + 1) if (2* i + 1)< self.size else -1
    
    def get_right_child(self, i):
        return (2 * i + 2) if (2* i + 2)< self.size else -1

    def get_parent(self, i):
        if  (i>=self.size) or (((i-1)/2) >= self.size):
            raise ValueError('Index out of bounds of the heap')
        return int((i-1)/2)
    
    def reset_node(self, node_number, value):
        index = self.K[node_number]
        if(index == -1):
            raise IndexError('Node not in Heap')
        self.D[index] = value
        parent = self.get_parent(index)
        while(self.D[parent]< self.D[index]):
            self.swap(index, parent)
            index = parent
            if(parent == 0):
                break
            parent = self.get_parent(index)

        self.max_heapify(index)
        
    def pop(self):
        element_to_remove = self.H[0], self.D[0]
        self.swap(0, self.size-1)
        self.K[self.H[0]] = -1
        #print("After swap")
        #self.print_heap()
        self.size -= 1
        #print("Attempting to max heapify")
        self.max_heapify(0)
        return element_to_remove

    def swap(self, i, j):
        self.K[self.H[i]] = j
        self.K[self.H[j]] = i        
        self.H[i], self.H[j] = self.H[j], self.H[i]
        self.D[i], self.D[j] = self.D[j], self.D[i]

    def max_heapify(self, i):
        l = self.get_left_child(i)
        r = self.get_right_child(i)
        largest = i

        #check if left heap is larger
        if l != -1:
            if l< self.size and self.D[l] > self.D[i]:
                largest = l
            else:
                largest = i

         #check if right heap is larger
        if r!= -1:
            if r<= self.size and self.D[r] > self.D[largest]:
                largest = r
            else:
                largest = largest
        
        if largest != i:
            #print("Max heapifying on element:", self.D[largest], "i:", i)
            self.swap(i, largest)
            self.max_heapify(largest)

        #self.print_heap()
    
    def get_max_element(self):
        if self.size == 0:
            return None, None
        else:
            return self.H[0], self.D[0]

    def print_heap(self, node = 0, tab = ''):
        if(node == -1):
            return
        print(tab,str(self.D[node]))
        self.print_heap(self.get_left_child(node),  tab + '\t')
        self.print_heap(self.get_right_child(node), tab + '\t')


class EdgeHeap:
    def __init__(self, capacity):
        self.size = 0
        self.capacity  = capacity
        self.S = [-1] * capacity
        self.T = [-1] * capacity
        self.D = [-1] * capacity


    def push(self, source, target, value):
        if(self.size == self.capacity):
            raise OverflowError('Overflow error: Trying to add element in heap with full capacity')
        self.S[self.size] = source
        self.T[self.size] = target
        self.D[self.size] = value
        self.size += 1

        k = self.size - 1
        while k>0 and self.D[k] > self.D[self.get_parent(k)]:
            self.swap(k, self.get_parent(k))
            #self.max_heapify(k)
            k = self.get_parent(k)
            
    def get_left_child(self, i):
        return (2 * i + 1) if (2* i + 1)< self.size else -1
    
    def get_right_child(self, i):
        return (2 * i + 2) if (2* i + 2)< self.size else -1

    def get_parent(self, i):
        if  (i>=self.size) or (((i-1)/2) >= self.size):
            raise ValueError('Index out of bounds of the heap')
        return int((i-1)/2)
    
    def pop(self):
        element_to_remove = self.S[0], self.T[0], self.D[0]
        self.swap(0, self.size-1)
        self.size -= 1
        self.max_heapify(0)
        return element_to_remove

    def swap(self, i, j):
        self.S[i], self.S[j] = self.S[j], self.S[i]
        self.T[i], self.T[j] = self.T[j], self.T[i]
        self.D[i], self.D[j] = self.D[j], self.D[i]

    def max_heapify(self, i):
        l = self.get_left_child(i)
        r = self.get_right_child(i)
        largest = i

        #check if left heap is larger
        if l != -1:
            if l< self.size and self.D[l] > self.D[i]:
                largest = l
            else:
                largest = i

         #check if right heap is larger
        if r!= -1:
            if r<= self.size and self.D[r] > self.D[largest]:
                largest = r
            else:
                largest = largest
        
        if largest != i:
            #print("Max heapifying on element:", self.D[largest], "i:", i)
            self.swap(i, largest)
            self.max_heapify(largest)

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
        self.print_heap(self.get_left_child(node),  tab + '\t')
        self.print_heap(self.get_right_child(node), tab + '\t')
