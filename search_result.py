
from dataclasses import dataclass
from typing import Optional
from networkx import Graph

@dataclass
class SearchResult:
    path: list
    explored_graph: Graph
    path_len: int
    visited_nodes: int
    evaluated_nodes: int
    ppr_results: Optional[dict] = None
    evaluated_edges: int = 0


    def print_metrics(self, include_path = False):
        print('Path Len: {a} \n# of visited Nodes: {b} \n# of evaluated nodes: {c}'.format(a=self.path_len, b=self.visited_nodes, c=self.evaluated_nodes))
        if(include_path):
            print(f'Path: {self.path}')