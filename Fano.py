from Encoding import Encoding, Probability
from typing import List, Union
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class FanoGraph:
    class Node:
        def __init__(self) -> None:
            self.probs: List[Probability] = []
            self.parent: Union[FanoGraph.Node, None] = None
            self.children: List[FanoGraph.Node] = []

            self.code: List[str] = []

        def value(self):
            return np.round(sum(p.value() for p in self.probs), 4)

        def is_single(self):
            return len(self.probs) == 1

        def is_last_leaf(self):
            return len(self.children) == 0

        def get_code(self):
            return "".join(self.code)

        def get_L(self):
            return len(self.code)

        def get_comm_name(self):
            return '.'.join([i.name for i in self.probs])

        def __str__(self):
            new_line = '\n'
            return f"{self.value()}\n{self.get_comm_name()}\n{self.get_code()}"

    def __init__(self, probabilities) -> None:
        self.probabilities = probabilities
        self.head: Union[FanoGraph.Node, None] = None
        self.base_layer: List[FanoGraph.Node] = []
        self.leaves = []

        self.L = None
        self.H = None

        self.build()
        self.compute_code()

        self.print_codes_and_Ls()
        self.print_L()
        self.print_H()
        self.print_r()
        self.draw_graph()
        print()

    def to_nodes(self, probabilities):
        layer = []
        n = FanoGraph.Node()
        for p in probabilities:
            n.probs.append(p)
        layer.append(n)
        return layer

    def build(self):
        self.base_layer.extend(self.to_nodes(self.probabilities))
        layer = self.base_layer[:]

        while len(layer):
            new_layer = []
            for n in layer:
                g_node, l_node = self.split_node(n)
                if g_node.is_single():
                    self.leaves.append(g_node)
                else:
                    new_layer.append(g_node)
                if l_node.is_single():
                    self.leaves.append(l_node)
                else:
                    new_layer.append(l_node)
            layer = new_layer
        self.head = self.base_layer[0]
        print()

    def get_arr_val(self, arr):
        return np.round(sum(p.value() for p in arr), 4)

    def split_node(self, node: Node):
        gt = []
        lt = []

        node.probs.sort(key=lambda p: p.value(), reverse=True)

        last_gt = []
        last_lt = []

        for i in range(1, len(node.probs)):
            f1 = node.probs[:i]
            f2 = node.probs[i:]

            if self.get_arr_val(f1) < self.get_arr_val(f2):
                last_gt = f1[:]
                last_lt = f2[:]
                continue
            else:
                last_val_delta = abs(self.get_arr_val(last_gt) - self.get_arr_val(last_lt))
                curr_val_delta = abs(self.get_arr_val(f1) - self.get_arr_val(f2))
                if curr_val_delta < last_val_delta:
                    gt = f1
                    lt = f2
                elif last_val_delta == 0:
                    gt = f1
                    lt = f2
                else:
                    gt = last_gt
                    lt = last_lt
                break
        g_node = FanoGraph.Node()
        g_node.probs = gt
        g_node.parent = node
        l_node = FanoGraph.Node()
        l_node.probs = lt
        l_node.parent = node
        node.children = [g_node, l_node]
        return (g_node, l_node)

    def connect(self, node1, node2, g):
        g.add_edge(node1, node2)

    def dfs_connect(self, visitor, g):
        def dfs_inner(node, v, parent, val):
            v(node, parent, val)
            if node.is_last_leaf():
                return
            else:
                left = node.children[0]
                right = node.children[1]
                parent = node
                dfs_inner(left, visitor, parent, g)
                dfs_inner(right, visitor, parent, g)

        left = self.head.children[0]
        right = self.head.children[1]
        parent = self.head
        dfs_inner(left, visitor, parent, g)
        dfs_inner(right, visitor, parent, g)

    def add_code(self, node: Node, past: List[str], item: str):
        node.code.extend(past)
        node.code.append(item)

    def dfs(self, visitor):
        def dfs_inner(node, v, parent, val):
            v(node, parent.code, val)
            if node.is_last_leaf():
                return
            else:
                left = node.children[0]
                right = node.children[1]
                parent = node
                dfs_inner(left, visitor, parent, '0')
                dfs_inner(right, visitor, parent, '1')

        left = self.head.children[0]
        right = self.head.children[1]
        parent = self.head
        dfs_inner(left, visitor, parent, '0')
        dfs_inner(right, visitor, parent, '1')

    def compute_code(self):
        self.dfs(self.add_code)

    def draw_graph(self):
        plt.title("Fano")
        G = nx.Graph()
        self.dfs_connect(self.connect, G)
        nx.draw(G, with_labels=True)
        plt.show()

    def print_codes_and_Ls(self):
        for i, node in enumerate(self.leaves):
            print(f"{node.get_comm_name():3} = {node.value():5} = {node.get_code():10} \t\t\tL{i} = {node.get_L()}")

    def print_L(self):
        components = []
        results = []
        for i in self.leaves:
            components.append(f"{i.value()} * {i.get_L()}")
            results.append(np.round(i.value() * i.get_L(), 4))
        res = np.round(sum(results), 4)
        self.L = res
        print(f"L = {' + '.join(components)} = {' + '.join([str(i) for i in results])} = {res}")

    def print_H(self):
        components = []
        results = []
        for i in self.leaves:
            components.append(f"{i.value()} * log2({i.value()})")
            results.append(np.round(i.value() * np.log2(i.value()), 4))
        res = np.round(-sum(results), 4)
        self.H = res
        print(f"H = -( {' + '.join(components)} ) = -( {' '.join([str(i) for i in results])} ) = {res}")

    def print_r(self):
        print(f"r = L - H = {self.L} - {self.H} = {np.round(self.L - self.H, 4)}")


if __name__ == '__main__':
    e = Encoding("task3_huffman_fano\\input1.txt")
    hg = FanoGraph(e.probabilities)
    print()
