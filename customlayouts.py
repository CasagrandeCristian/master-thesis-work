import networkx
import random
import time
import numpy as np
import numpy.typing as np_typing
import math
from typing import List, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import multiprocessing
from enum import Enum

def test_back_track(graph : networkx.Graph):
    layout = dict()
    for index, node in enumerate(graph.nodes):
        layout[node] = [0, index] if node != 100 else [0,0] 
    return layout

def distributed_layout(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True)

def distributed_layout_800_iter(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True, 800)

def distributed_layout_400_iter(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True, 400)

def distributed_layout_300_iter(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True, 300)

def distributed_layout_100_iter(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True, 100)

def distributed_layout_200_iter(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True, 200)

def distributed_layout_25_iter(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True, 25)

def distributed_layout_12_iter(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True, 12)

def distributed_layout_15_iter(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True, 12)

def distributed_layout_6_iter(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True, 6)

def distributed_layout_19_iter(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True, 19)

def distributed_layout_37_iter(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True, 37)

def distributed_layout_30_iter(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, True, 30)

def distributed_layout_no_norm(graph : networkx.Graph, dim: int):
    return distributed_layout_internal(graph, dim, False)


def distributed_layout_internal(graph : networkx.Graph, dim: int, normalize: bool = True, max_iter = 50):
    MAX_ITER = max_iter
    START_RANGE = 1
    END_RANGE = dim+1
    EPSILON = 0.0000001
    LIKE_PAPER = True

    layout = dict()
    
    #initialization
    u = np.zeros((dim+1, graph.number_of_nodes()))
    random.seed(time.time())

    node_to_index = {node: idx for idx, node in enumerate(graph.nodes)}
    nodes_list = list(graph.nodes)

    #Find a vector D-Orthogonal to u1 = 1
    for j in range(START_RANGE, END_RANGE):
        random_permutation = random.sample(nodes_list, len(nodes_list))
        iterator = iter(random_permutation)
        for node1 in iterator:
            node2 = next(iterator, None)
            if(node2 is not None):
                random_n = random.randint(0, 100)
                u[j, node_to_index[node1]] = random_n/graph.degree(node1)
                u[j, node_to_index[node2]] = -random_n/graph.degree(node2)
            else:
                u[j, node_to_index[node1]] = 0


    nbdr_idxs_lists = [[node_to_index[nbr] for nbr in graph.neighbors(node)] for node in graph.nodes]
    
    #Calculate coordinates
    for i in range(MAX_ITER):
        u_prev = np.array(u, copy=True)
        for node_idx, nbdr_idxs_list in enumerate(nbdr_idxs_lists):
            neighbor_matrix = (u_prev if LIKE_PAPER else u)[START_RANGE:END_RANGE, nbdr_idxs_list]
            neighbor_mean = np.mean(neighbor_matrix, axis=1)
            u[START_RANGE:END_RANGE, node_idx] = 0.5 * (u[START_RANGE:END_RANGE, node_idx] + neighbor_mean)

        #DRAW GRAPH
        # layout = dict()
        # for index, node in enumerate(graph.nodes):
        #     layout[node] = np.array([u[j, index] for j in range(1, dim+1)])
        # networkx.draw_networkx_nodes(graph, pos=layout, label=True)
        # networkx.draw_networkx_labels(graph, pos=layout)
        # networkx.draw_networkx_edges(graph, pos=layout)
        # plt.title("{i}".format(i = i))
        # plt.savefig('gif_pics\\gif_{i}.png'.format(i = i))
        # plt.clf()


    #Normalization
    if(normalize):
        m = u ** 2
        for i in range(MAX_ITER):
            m_prev = np.array(m, copy=True)
            for node_idx, nbdr_idxs_list in enumerate(nbdr_idxs_lists):
                neighbor_matrix = m_prev[START_RANGE:END_RANGE, nbdr_idxs_list]
                neighbor_mean = np.mean(neighbor_matrix, axis=1)
                m[START_RANGE:END_RANGE, node_idx] = 0.5 * (m_prev[START_RANGE:END_RANGE, node_idx] + neighbor_mean)
        m_sqrt = np.sqrt(m)
        u = np.divide(u, m_sqrt, out=m, where=(m > EPSILON))    
            
    #Build Layout to return
    layout = dict()
    for index, node in enumerate(graph.nodes):
        layout[node] = u[START_RANGE:END_RANGE, index]
    
    return layout 

@dataclass
class SocsNodeData:
    index : int
    pos : np_typing.NDArray
    vel : float
    dir : np_typing.NDArray
    social_neighbours: List[networkx.classes.reportviews.NodeView]
    converged: bool

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    dot_prod = np.dot(a,b)
    div = norm_a * norm_b
    return np.divide(dot_prod, div)


def get_social_neighbours(graph: networkx.Graph, socs_data: Dict[networkx.classes.reportviews.NodeView, SocsNodeData], 
                          node: networkx.classes.reportviews.NodeView, neighbours_count: int, with_cyclon: bool, random_init: bool):
    
    candidate_nbrs = set()

    base_nbrs = (random.sample(graph.nodes, neighbours_count) if random_init else graph.neighbors(node)) if socs_data[node].social_neighbours is None else socs_data[node].social_neighbours 
    for nbr in base_nbrs:
        candidate_nbrs.add(nbr)
        
        candidate_nbrs.update(graph.neighbors(nbr) if socs_data[node].social_neighbours is None else socs_data[nbr].social_neighbours)

    if(with_cyclon):
        cyclon = random.sample(list(graph.nodes), neighbours_count)
        for x in cyclon:
            candidate_nbrs.add(x)

    candidate_nbrs.discard(node)
    
    candidate_nbrs_list = list(candidate_nbrs)
    candidate_nbrs_count = len(candidate_nbrs)
    if candidate_nbrs_count <= 0:
        return []
    
    node_pos = socs_data[node].pos
    nbr_positions = np.array([socs_data[nbr].pos for nbr in candidate_nbrs_list])
    distances = np.linalg.norm(nbr_positions - node_pos, axis=1)

    if len(distances) > neighbours_count:
        sorting = np.argpartition(distances, neighbours_count)[:neighbours_count]
        sorted_nbrs_pos = np.array([socs_data[candidate_nbrs_list[i]].pos for i in sorting])
        social_nbrs  = np.array([candidate_nbrs_list[i] for i in sorting])
        return (sorted_nbrs_pos[:neighbours_count], social_nbrs[:neighbours_count])  
    else:
        return ([socs_data[candidate_nbr].pos for candidate_nbr in candidate_nbrs_list], candidate_nbrs_list) 



def get_social_neighbours_nons(graph: networkx.Graph, socs_data: Dict[networkx.classes.reportviews.NodeView, SocsNodeData], 
                          node: networkx.classes.reportviews.NodeView, neighbours_count: int, with_cyclon: bool, random_init: bool):
    

    if(socs_data[node].social_neighbours is None):
        candidate_nbrs = set()
        for nbr in graph.neighbors(node):
            candidate_nbrs.add(nbr)
            candidate_nbrs.update(graph.neighbors(nbr))
        candidate_nbrs.discard(node)
        candidate_nbrs_list = list(candidate_nbrs)
        socs_data[node].social_neighbours = candidate_nbrs_list
    

    node_pos = socs_data[node].pos
    nbr_positions = np.array([socs_data[nbr].pos for nbr in socs_data[node].social_neighbours])
    distances = np.linalg.norm(nbr_positions - node_pos, axis=1)
    if(len(socs_data[node].social_neighbours) > neighbours_count):
        sorted_nbrs_pos = np.array([socs_data[socs_data[node].social_neighbours[i]].pos for i in np.argpartition(distances, neighbours_count)[:neighbours_count]])
    else:
        sorted_nbrs_pos = np.array([socs_data[nb].pos for nb in socs_data[node].social_neighbours])

    return (sorted_nbrs_pos, socs_data[node].social_neighbours)


def socs_layout_iter_based(graph : networkx.Graph, dim: int):
    max_iter = 100
    return socs_layout(graph, dim, max_iter)

def socs_layout_iter_based_50(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, max_iter=50)

def socs_layout_iter_based_1000(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, max_iter=1000)

def socs_layout_iter_based_1000_correct(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, max_iter=1000)

def socs_layout_iter_based_2000_correct(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, max_iter=2000)

def socs_layout_iter_based_1000_random_init(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, max_iter=1000, random_init=True)

def socs_layout_iter_based_1000_random_init_correct(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, max_iter=1000, random_init=True)

def socs_layout_cyclon(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, with_cyclon=True, max_iter=1000, epsilon_based_on_dim=True)

def socs_layout_cyclon_diff_force(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, with_cyclon=True, max_iter=1000, epsilon_based_on_dim=True, FA=2, FR=-1)

def socs_layout_cyclon_random_init(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, with_cyclon=True, random_init=True, max_iter=1000, epsilon_based_on_dim=True)

def socs_layout_cyclon_R50(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, with_cyclon=True, R=50)

def socs_layout_cyclon_R100(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, with_cyclon=True, R=100)

def socs_layout_R50(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, R=50)

def socs_layout_R100(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, R=100)

def socs_layout_all(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, R = -1)

def socs_layout_lin_log(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, with_cyclon=True, FA=0, FR=-1)

def socs_layout_social_random_init(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = True)

def socs_layout_social_random_init_R50(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = True, R=50)

def socs_layout_social_random_init_R100(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = True, R=100)

def socs_layout_fake_fruchtermann_reingold(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = True, velCalcStrategy = VelCalcStrategy.FruchtermanReingold, max_iter=50) #in realtà qui ci va a=2 r=-1

def socs_layout_fruchtermann_reingold(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = True, velCalcStrategy = VelCalcStrategy.FruchtermanReingold, max_iter=50, FA=2, FR=-1)

def socs_layout_fruchtermann_reingold_final_50(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = False, velCalcStrategy = VelCalcStrategy.FruchtermanReingold, max_iter=50, FA=2, FR=-1)

def socs_layout_fruchtermann_reingold_final_100(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = False, velCalcStrategy = VelCalcStrategy.FruchtermanReingold, max_iter=100, FA=2, FR=-1)

def socs_layout_fruchtermann_reingold_final_100_cyclon(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = False, velCalcStrategy = VelCalcStrategy.FruchtermanReingold, with_cyclon=True, max_iter=100, FA=2, FR=-1)

def socs_layout_spring_final_100_cyclon(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = False, velCalcStrategy = VelCalcStrategy.Spring, with_cyclon=True, max_iter=100, FA=2, FR=-1)

def socs_layout_spring_final_100(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = False, velCalcStrategy = VelCalcStrategy.Spring, max_iter=100)

def socs_layout_fruchtermann_reingold_final_200(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = False, velCalcStrategy = VelCalcStrategy.FruchtermanReingold, max_iter=200, FA=2, FR=-1)

def socs_layout_fake_spring(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = True, velCalcStrategy = VelCalcStrategy.Spring, max_iter=50, FA=2, FR=-1)

def socs_layout_fruchtermann_reingold_correct(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = True, velCalcStrategy = VelCalcStrategy.FruchtermanReingold, max_iter=50, FA=2, FR=-1)

def socs_layout_spring(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, random_init = True, velCalcStrategy = VelCalcStrategy.Spring, max_iter=50)

def socs_layout_fruchtermann_reingold_no_random_init(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, velCalcStrategy = VelCalcStrategy.FruchtermanReingold, max_iter=50, FA=2, FR=-1)

def socs_layout_fake_fruchtermann_reingold_no_random_init(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, velCalcStrategy = VelCalcStrategy.FruchtermanReingold, max_iter=50)

def socs_layout_fake_spring_no_random_init(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, velCalcStrategy = VelCalcStrategy.Spring, max_iter=50, FA=2, FR=-1)

def socs_layout_spring_no_random_init(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, velCalcStrategy = VelCalcStrategy.Spring, max_iter=50)

# epsilon based on dimension
def socs_layout_varying_epsilon_iter_1000(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, max_iter=1000, epsilon_based_on_dim=True)

def socs_layout_varying_epsilon_variant_2_iter_1000(graph : networkx.Graph, dim: int):
    return socs_layout(graph, dim, max_iter=1000, epsilon_based_on_dim=True)


class VelCalcStrategy(Enum):
    OriginalSocs = 1,
    FruchtermanReingold = 2,
    Spring = 3


def socs_layout(graph : networkx.Graph, dim: int, max_iter = -1, with_cyclon : bool = False, R : int = 10, FA : int = 1, FR : int= -2, velCalcStrategy: VelCalcStrategy = VelCalcStrategy.OriginalSocs, random_init = False, epsilon_based_on_dim = True):
    EPSILON = 0.005

    if epsilon_based_on_dim and dim > 10:
        EPSILON *= dim/10

    MAX_ITER = max_iter
    T = 10

    T_DECREASE = T/(max_iter+1) if max_iter != -1 else 1
    random.seed(2)

    #Initialization
    socs_data = dict()
    for index, node in enumerate(graph.nodes):
        pos = np.array([random.random() * 100  for x in range(dim)])
        vel = 1
        dir = np.array([(random.random()) * 2 - 1 for x in range(dim)])
        dir = dir/np.linalg.norm(dir)
        socs_data[node] = SocsNodeData(index, pos=pos, vel=vel, dir=dir, social_neighbours=None, converged=False)

    everybody_converged = False
    convergence_counter = 0
    iter_counter = 0
    stop_criteria = (lambda : iter_counter < MAX_ITER and not everybody_converged) if MAX_ITER != -1 else (lambda : not everybody_converged)
    
    nbr_dict = {node: list(graph.neighbors(node)) for node in graph.nodes}
    
    while stop_criteria():
        start_time = time.time()
        everybody_converged = True
        new_socs_data = dict()
        nbr_dict = {node: np.array([socs_data[nbr].pos for nbr in graph.neighbors(node)]) for node in graph.nodes}

        for index, node in enumerate(graph.nodes):
            if(socs_data[node].converged):
                new_socs_data[node] = socs_data[node]
                continue

            nbr_positions = nbr_dict[node]
            dir = compute_direction(nbr_positions, socs_data[node].pos, FA)

            (social_neighbours_positions, social_neighbours) = get_social_neighbours(graph, socs_data, node, R, with_cyclon, random_init) if R != -1 else np.array([socs_data[nbr].pos for nbr in graph.nodes if nbr != node])
            dir_r = compute_direction(social_neighbours_positions, socs_data[node].pos, FR)

            dir = dir - dir_r
            if(velCalcStrategy == VelCalcStrategy.FruchtermanReingold):
                vel = min(np.linalg.norm(dir), T)
                dir = np.divide(dir,np.linalg.norm(dir))
            elif(velCalcStrategy == VelCalcStrategy.OriginalSocs):
                dir = np.divide(dir,np.linalg.norm(dir))     
                vel = (0.5 * cosine_similarity(dir, socs_data[node].dir) + 1) * socs_data[node].vel
            elif(velCalcStrategy == VelCalcStrategy.Spring):
                dir = np.divide(dir,np.linalg.norm(dir))     
                vel = T
            new_pos = socs_data[node].pos + dir * vel

            c = False
            if(np.linalg.norm(np.subtract(socs_data[node].pos,new_pos)) > EPSILON):
                everybody_converged = False
            else:
                convergence_counter+=1
                print("{node} converged!, progress : {prog}%".format(node = node, prog = (convergence_counter/graph.number_of_nodes())*100))
                c = True

            new_socs_data[node] = SocsNodeData(socs_data[node].index, new_pos, vel, dir, social_neighbours, c) 
        
        if(velCalcStrategy != VelCalcStrategy.OriginalSocs and T > T_DECREASE):
            T-=T_DECREASE
        iter_counter+=1
        socs_data = new_socs_data
        print(iter_counter, time.time()-start_time, (convergence_counter/graph.number_of_nodes())*100)
        
        #DRAW GRAPH
        # layout = dict()
        # for index, node in enumerate(graph.nodes):
        #     layout[node] = socs_data[node].pos
        # networkx.draw_networkx_nodes(graph, pos=layout, label=True)
        # networkx.draw_networkx_labels(graph, pos=layout)
        # networkx.draw_networkx_edges(graph, pos=layout)
        # plt.title("{i}".format(i = i))
        # plt.savefig('gif_pics\\gif_{i}.png'.format(i = i))
        # plt.clf()

    
    layout = dict()
    for k,data in socs_data.items():
        layout[k] = data.pos

    return layout

def compute_direction(nbr_positions, node_pos, F):
    nbr_positions = np.asarray(nbr_positions)
    if len(nbr_positions) == 0:
        return np.zeros_like(node_pos)
    pi_pj = nbr_positions - node_pos
    pi_pj_norms = np.linalg.norm(pi_pj, axis=1, keepdims=True)
    valid_mask = pi_pj_norms > 0
    eij = np.divide(pi_pj, pi_pj_norms, out=np.zeros_like(pi_pj), where=valid_mask)
    return np.sum((pi_pj_norms ** F) * eij, axis=0)
    
def line(graph : networkx.Graph, dim: int):
    pass

def hope(graph : networkx.Graph, dim: int):
    pass

def laplacian_eigenmaps(graph : networkx.Graph, dim: int):
    pass

def node2vec(graph : networkx.Graph, dim: int):
    pass

def graph_factorization(graph : networkx.Graph, dim: int):
    pass



#Region Rooted Spanning Tree Embedding
@dataclass
class RootedSpanningTreeInvitation():
    tree_id: int
    sender_node: networkx.classes.reportviews.NodeView
    sender_level: int

@dataclass
class RootedSpanningTreeNodeData():
    invitations: List[RootedSpanningTreeInvitation]
    neighbors_parent_count : Dict[networkx.classes.reportviews.NodeView, int]


def root_election(graph : networkx.Graph, number: int):
    roots = random.sample(list(graph.nodes), number)
    return roots

def add_to_tree(tree: networkx.DiGraph, tree_id: int, rst_data: Dict[networkx.classes.reportviews.NodeView, RootedSpanningTreeNodeData], node: networkx.classes.reportviews.NodeView, parent: networkx.classes.reportviews.NodeView, parent_level: int, original_graph: networkx.Graph, new_invites_data: Dict[networkx.classes.reportviews.NodeView, List[RootedSpanningTreeInvitation]] = None):
    # After a node u is included in the i-th tree, u sends invitations (i, u) to all its neighbors inviting them to
    # be its children in tree i.
    if(parent == None):
        tree.add_node(node)
    else:
        tree.add_edge(parent, node)
        rst_data[node].neighbors_parent_count[parent] = rst_data[node].neighbors_parent_count.get(parent, 0) + 1

    for nbr in original_graph.neighbors(node):
        invite = RootedSpanningTreeInvitation(tree_id, node, parent_level+1)
        if(new_invites_data is None):
            rst_data[nbr].invitations.append(RootedSpanningTreeInvitation(tree_id, node, parent_level+1))
        else:
            if nbr not in new_invites_data:
                new_invites_data [nbr] = []
            new_invites_data[nbr].append(invite)
            

def random_selection(candidates: List[RootedSpanningTreeInvitation]) -> RootedSpanningTreeInvitation:
    return random.sample(candidates, 1)[0]

def random_lowest_level_selection(candidates: List[RootedSpanningTreeInvitation]) -> RootedSpanningTreeInvitation:
    min_level = min([invitation.sender_level for invitation in candidates])
    mins = [x for x in candidates if x.sender_level == min_level]
    return random.sample(mins, 1)[0]

def embed_spanning_tree(root: networkx.classes.reportviews.NodeView, spanning_tree: networkx.DiGraph, max_n: int):
    layout = dict({root : np.array([])})
    queue = [root]
    while len(queue) != 0:
        parent = queue.pop()
        child_ids = set()
        parent_coord = layout[parent]
        for child in spanning_tree.successors(parent):
            child_id = random.randint(0, max_n)
            while child_id in child_ids:
                child_id = random.randint(0, max_n)
            child_ids.add(child_id)
            layout[child] = np.append(parent_coord, child_id)
            queue.append(child)
    return layout


def rooted_spanning_tree(graph : networkx.Graph, dim: int, selection = random_selection, Q = 50, wait_next_iter_to_send_invites = False, LAMBDA = 5):

    MAX_COORD_NUMBER = 10000
    rst_data_per_node = {node: RootedSpanningTreeNodeData(list(), {nbr: 0 for nbr in graph.neighbors(node)}) for node in graph.nodes()}
    spanning_trees = [networkx.DiGraph() for _ in range(LAMBDA)]

    #Simulate Root Election
    roots = root_election(graph, LAMBDA)
    for tree_id, root in enumerate(roots):
        add_to_tree(spanning_trees[tree_id], tree_id, rst_data_per_node, root, None, -1, graph)
    
    network_has_valid_invitations = True

    while network_has_valid_invitations:

        network_has_valid_invitations = False
        new_invites = {}

        #Tree construction round
        for node in graph.nodes:
            rst_data = rst_data_per_node[node]
            min_parent_count = min(rst_data.neighbors_parent_count.values(), default=0)
            valid_invitations = []
            mins_invitations = []
            for invitation in rst_data.invitations:
                if(node in spanning_trees[invitation.tree_id]):
                    continue

                if rst_data.neighbors_parent_count.get(invitation.sender_node) == min_parent_count:
                    mins_invitations.append(invitation)
                
                valid_invitations.append(invitation)
            
            min_found = len(mins_invitations) > 0
            should_accept_an_invite = (random.randint(0,100) <= Q) or min_found
            node_has_valid_invitations = len(valid_invitations) > 0
            network_has_valid_invitations = network_has_valid_invitations or node_has_valid_invitations

            if(node_has_valid_invitations and should_accept_an_invite):
                accepted_invite = selection(mins_invitations if min_found else valid_invitations)
                add_to_tree(spanning_trees[accepted_invite.tree_id], accepted_invite.tree_id, rst_data_per_node, node, accepted_invite.sender_node, accepted_invite.sender_level, graph, new_invites_data = new_invites if wait_next_iter_to_send_invites else None)
                valid_invitations.remove(accepted_invite)
            
            rst_data.invitations = valid_invitations
        
        for (node, invites) in new_invites.items():
            for invite in invites:
                rst_data_per_node[node].invitations.append(invite)
    
    layouts = {}
    for tree_id, root in enumerate(roots):
        layouts[tree_id] = embed_spanning_tree(root, spanning_trees[tree_id], MAX_COORD_NUMBER)
    return layouts
            
    
                
def rooted_spanning_tree_lowest_level_selection(graph : networkx.Graph, dim: int):
    return rooted_spanning_tree(graph, dim, selection=random_lowest_level_selection)

def patched_rooted_spanning_tree_lowest_level_selection(graph : networkx.Graph, dim: int):
    return rooted_spanning_tree(graph, dim, wait_next_iter_to_send_invites= True, selection=random_lowest_level_selection)

def patched_rooted_spanning_tree(graph : networkx.Graph, dim: int):
    return rooted_spanning_tree(graph, dim, wait_next_iter_to_send_invites= True)

def rooted_spanning_tree_q_25(graph : networkx.Graph, dim: int):
    return rooted_spanning_tree(graph, dim, Q=25)

def rooted_spanning_tree_q_75(graph : networkx.Graph, dim: int):
    return rooted_spanning_tree(graph, dim, Q=75)

def rooted_spanning_tree_1_tree(graph : networkx.Graph, dim: int):
    return rooted_spanning_tree(graph, dim, LAMBDA=1)

def rooted_spanning_tree_15_tree(graph : networkx.Graph, dim: int):
    return rooted_spanning_tree(graph, dim, LAMBDA=15)

def rooted_spanning_tree_best(graph : networkx.Graph, dim: int):
    return rooted_spanning_tree(graph, dim, LAMBDA=15, Q=75, selection=random_lowest_level_selection)

def rooted_spanning_tree_10_tree(graph : networkx.Graph, dim: int):
    return rooted_spanning_tree(graph, dim, LAMBDA=10)

def rooted_spanning_tree_2_tree(graph : networkx.Graph, dim: int):
    return rooted_spanning_tree(graph, dim, LAMBDA=2)

def rooted_spanning_tree_3_tree(graph : networkx.Graph, dim: int):
    return rooted_spanning_tree(graph, dim,  LAMBDA=3)
