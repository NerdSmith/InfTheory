# from datetime import datetime
from typing import List, Union
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from time import time
from Encoding import Probability, Encoding
# from networkx.drawing.nx_agraph import graphviz_layout

from TaskInpType import TInpType


class HuffGraph:
    class Node:
        def __init__(self):
            self.prob: Union[Probability, None] = None
            self.parent: Union[HuffGraph.Node, None] = None
            self.children: List[HuffGraph.Node] = []
            self.creation_time = time()

            self.code: List[str] = []

        def is_last_leaf(self):
            return len(self.children) == 0

        def value(self):
            return self.prob.val

        def get_code(self):
            return "".join(self.code)

        def get_L(self):
            return len(self.code)

        def get_creation_time(self):
            return self.creation_time

        def __str__(self):
            new_line = '\n'
            return f"{self.prob.value()}\n{self.prob.name + new_line if self.prob.name != '_' else ''}{self.get_code()}"

    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.head: Union[HuffGraph.Node, None] = None
        self.base_layer: List[HuffGraph.Node] = []
        self.L = None
        self.H = None

        self.build()
        self.compute_code()

        self.print_codes_and_Ls()
        self.print_L()
        self.print_H()
        self.print_r()
        self.draw_graph()

    def build(self):
        self.base_layer.extend(self.to_nodes(self.probabilities))
        layer = self.base_layer[:]
        while len(layer) > 1:
            layer.sort(key=lambda node: (node.value(), -node.get_creation_time()), reverse=True)
            to_merge = layer[-2:]
            del layer[-2:]
            new_node = HuffGraph.Node()
            new_node.children = to_merge
            new_prob = Probability("_", to_merge[0].prob.value() + to_merge[1].prob.value())
            new_node.prob = new_prob
            for n in to_merge:
                n.parent = new_node
            layer.append(new_node)
        self.head = layer[0]
        print()

    def add_code(self, node: Node, past: List[str], item: str):
        node.code.extend(past)
        node.code.append(item)

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

    def to_nodes(self, probabilities):
        layer = []
        for p in probabilities:
            n = HuffGraph.Node()
            n.prob = p
            layer.append(n)
        return layer

    def draw_graph(self):
        plt.title("Huffman")
        G = nx.Graph()
        self.dfs_connect(self.connect, G)
        nx.draw(G, with_labels=True)
        plt.show()

    def print_codes_and_Ls(self):
        for i, node in enumerate(self.base_layer):
            print(f"{node.prob.name:3} = {node.prob.value():5} = {node.get_code():10} \t\t\tL{i} = {node.get_L()}")

    def print_L(self):
        components = []
        results = []
        for i in self.base_layer:
            components.append(f"{i.prob.value()} * {i.get_L()}")
            results.append(np.round(i.prob.value() * i.get_L(), 4))
        res = np.round(sum(results), 4)
        self.L = res
        print(f"L = {' + '.join(components)} = {' + '.join([str(i) for i in results])} = {res}")

    def print_H(self):
        components = []
        results = []
        for i in self.base_layer:
            components.append(f"{i.prob.value()} * log2({i.prob.value()})")
            results.append(np.round(i.prob.value() * np.log2(i.prob.value()), 4))
        res = np.round(-sum(results), 4)
        self.H = res
        print(f"H = -( {' + '.join(components)} ) = -( {' '.join([str(i) for i in results])} ) = {res}")

    def print_r(self):
        print(f"r = L - H = {self.L} - {self.H} = {np.round(self.L - self.H, 4)}")


if __name__ == '__main__':
    e = Encoding("task3_huffman_fano\\input1.txt")
    hg = HuffGraph(e.probabilities)
    print()
