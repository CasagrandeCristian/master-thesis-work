import networkx as nx
from dataclasses import dataclass
from collections import namedtuple

SignificanceResult = namedtuple('SignificanceResult',
                                       ['statistic', 'pvalue'])
@dataclass
class PPRResults():
    explored_nodes_percentage : float
    explored_edges_percentage : float
    kendall_tau : SignificanceResult
    ppr_explored_graph : dict
    ppr_ovr : dict

@dataclass
class MetricResults():
    visited_nodes: dict
    evaluated_nodes: dict
    metric_on_sub_graph : dict
    metric_on_full_graph : dict

def PPR(G: nx.Graph, source: nx.classes.reportviews.NodeView, alpha: float, nstart: dict = None, max_iter: int = 100):
    
    if(nstart != None):
        r = dict({node: nstart[node] if node in nstart else 0 for node in G.nodes()})
    else:
        r = dict({node: 1 if node == source else 0 for node in G.nodes()})

    iter_counter = 0
    tolerance = 1e-6
    delta = float('inf')
    while iter_counter < max_iter: #algorithm converge
        r_cap = dict({node : (1 - alpha if node == source else 0) for node in G.nodes()})
        for i in G.nodes():
            d = G.degree[i]
            if d > 0:
                for j in G.neighbors(i):
                    r_cap[j] = r_cap[j] + alpha * r[i]/d
            else:
                r_cap[source] = r_cap[source] + alpha * r[i]
        
        delta = sum(abs(r_cap[node] - r[node]) for node in G.nodes())
        if delta < G.number_of_nodes() * tolerance:
            return r_cap
        r = r_cap
        iter_counter+=1    
    return r






