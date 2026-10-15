from collections import deque
import math
import random
import networkx as nx
from ppr import PPR
import inspect
import numpy as np
from numpy import linalg as LA
from search_result import SearchResult
from enum import Enum
from functools import reduce

UNLIMITED = 10000
BUDGET = 50
#Fisso la condizione di stop in base al numero di nodi esplorati, 

class AugmentingPathMode(Enum):
    TOWARDS_SOURCE = 1
    PING_PONG = 2
    CHANGE_STOP_CRITERIA = 3


class BeamStackItem():
    neighbors:list
    neighbors_parent: dict
    last_item_selected_index:int

    @property
    def already_visited_items_count(self):
        return self.last_item_selected_index + 1


    def __init__(self, neighbors, neighbors_parent, last_item_selected_index):
        self.last_item_selected_index = last_item_selected_index
        self.neighbors_parent = neighbors_parent
        self.neighbors = neighbors
        
def cartesian_dist_in_layout(a, b, layout):
    #return math.dist(layout[a], layout[b])
    return LA.norm(layout[b] - layout[a])

def common_prefix_length(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i

def tree_distance(a, b, layout):
    return len(layout[a]) + len(layout[b]) - 2*common_prefix_length(layout[a], layout[b])

def tree_distance_multi_layout(a, b, layout):
    return min([tree_distance(a,b, single_layout) for single_layout in layout])

def ppr_and_dist_priority(a, b, layout, ppr, dist_fun = cartesian_dist_in_layout, blending_param = 0.5):
    return  pow(1-ppr[b], blending_param) * pow(dist_fun(a, b, layout), 1-blending_param)

def get_nearest_neighbor(G, layout, evaluated_nodes, explored_graph, visited_nodes, first_deg_nbr, source, dest, neighbours_to_sample):

    all_neighbors = list(G.neighbors(source))
    neighbours_pool = all_neighbors if neighbours_to_sample == -1 else random.sample(all_neighbors, min(neighbours_to_sample, len(all_neighbors))) + [source]
    neighbours = []
    for nbr in neighbours_pool:
        explored_graph.add_edge(source, nbr, capacity=1)
        evaluated_nodes.add(nbr)

        if nbr in visited_nodes:
            continue
        neighbours.append(nbr)

    if(len(neighbours) == 0):
        return None

    neighbours.sort(key = lambda nbr_index: cartesian_dist_in_layout(dest, nbr_index, layout))
    return neighbours[0]


def get_nearest_neighbor_dist(G, layout, evaluated_nodes, explored_graph, visited_nodes, first_deg_nbr, source, dest, neighbours_to_sample, dist_fun):

    all_neighbors = list(G.neighbors(source))
    neighbours_pool = all_neighbors if neighbours_to_sample == -1 else (random.sample(all_neighbors, min(neighbours_to_sample, len(all_neighbors))) + [nbr])
    neighbours = []
    for nbr in neighbours_pool:
        explored_graph.add_edge(source, nbr, capacity=1)
        evaluated_nodes.add(nbr)

        if nbr in visited_nodes or nbr in first_deg_nbr:
            continue
        neighbours.append(nbr)

    if(len(neighbours) == 0):
        return None

    neighbours.sort(key = lambda nbr_index: dist_fun(dest, nbr_index, layout))
    return dist_fun(neighbours[0], dest)


def a_star_like_layout_oriented(G: nx.Graph, layout, source_index, destination_index):
    print("A* - Like - Looking for a way to go from {a} to {b} ...".format(a = source_index, b = destination_index))
    visited_nodes = set([source_index])
    nodes_in_open_set = dict({source_index: 0})
    open_set = [(source_index, 0)]

    current_node = open_set.pop()
    while current_node[0] != destination_index:
        
        if nodes_in_open_set[current_node[0]] < current_node[1]:
            current_node = (current_node[0], nodes_in_open_set[current_node[0]])

        for nbr in G.neighbors(current_node[0]):
            if nbr in visited_nodes:
                continue

            if nbr in nodes_in_open_set:
                if nodes_in_open_set[nbr] > current_node[1] + 1:
                    nodes_in_open_set[nbr] = current_node[1] + 1
                continue

            nodes_in_open_set[nbr] = current_node[1] + 1
            open_set.append((nbr, current_node[1] + 1))
        
        if(len(open_set) == 0):
            print("Impossible to find a path from {a} to {b}".format(a = source_index, b = destination_index))
            return
        
        open_set.sort(key = lambda nbr: cartesian_dist_in_layout(destination_index, nbr[0], layout), reverse= True)
        current_node = open_set.pop()
        visited_nodes.add(current_node[0])

    print("Len -> {b} (source included),\n# of Visited nodes -> {c}".format( b=current_node[1], c=len(visited_nodes)))
    return

def beam_backtrack(beam_stack, path_len, beam_width):
    path_len-=1
    remaining_items = 0

    while path_len >= 0:
        remaining_items = len(beam_stack[path_len].neighbors) - beam_stack[path_len].already_visited_items_count
        if(remaining_items > 0):
            break
        path_len-=1

    if path_len < 0:
        return ([], 0)

    new_beams_count = min(remaining_items, beam_width)
    new_first_beam_index = beam_stack[path_len].last_item_selected_index + 1
    new_last_beam_index = new_first_beam_index + new_beams_count

    beams = beam_stack[path_len].neighbors[new_first_beam_index:new_last_beam_index]
    beam_stack[path_len].last_item_selected_index += new_beams_count

    path_len+=1
    #Remove everything that come after path len because it is now outdated
    beam_stack = beam_stack[:path_len]
    assert(len(beam_stack) == path_len)
    return (beam_stack, beams, path_len)

def beam_layout_oriented_with_backtrack(G: nx.Graph, layout, source, destination, beam_width, find_more_paths: AugmentingPathMode = None, dist_fun = cartesian_dist_in_layout, blending_param = -1, pre_calc_ppr = None, befs_w = -1, with_non = False, non_to_sample = -1):
    print("Looking for a way to go from {a} to {b} ...".format(a = source, b = destination))
    evaluated_nodes = set()

    should_calculate_ppr = blending_param != -1 and pre_calc_ppr == None

    #dict containing as key visited nodes and as value his predecessor
    visited_nodes = dict({source : -1})
    explored_graph = nx.Graph()

    beams = [source]
    path_len = 0
    beam_stack = []
    current_ppr = pre_calc_ppr

    befs_frontier = []
    befs_frontier_parent = {}

    continue_criteria = (lambda : destination not in visited_nodes) if find_more_paths != AugmentingPathMode.CHANGE_STOP_CRITERIA and find_more_paths != AugmentingPathMode.PING_PONG else (lambda: len(visited_nodes.keys()) < BUDGET)

    destination_found = False

    while continue_criteria(): 
        neighbors = []
        neighbors_parent = dict()
        neighbors_count = 0

        for beam in beams:
            for nbr in G.neighbors(beam):
                explored_graph.add_edge(nbr, beam, capacity=1)
                evaluated_nodes.add(nbr)

                if nbr in visited_nodes or nbr in neighbors_parent:
                    continue

                neighbors_count+=1
                neighbors.append(nbr)
                neighbors_parent[nbr] = beam

                if nbr == destination:
                    destination_found = True
                
                if not nbr in befs_frontier_parent:
                    befs_frontier_parent[nbr] = beam
                    befs_frontier.append(nbr)
        
        if(destination_found and find_more_paths != AugmentingPathMode.CHANGE_STOP_CRITERIA):
            path_len += 1
            beams = [destination]
            visited_nodes[destination] = neighbors_parent[destination]
            break


        if(should_calculate_ppr):
            current_ppr = ppr.PPR(explored_graph, source, 0.85, current_ppr, 1)


        parent_dict = (befs_frontier_parent if befs_w != -1 else neighbors_parent)
        frontier = (befs_frontier if befs_w != -1 else neighbors)

        distances = {}
        if(with_non):
            for nbr in frontier:
                best_non = get_nearest_neighbor(G, layout, evaluated_nodes, explored_graph, visited_nodes, frontier, nbr, destination, non_to_sample)
                if best_non != None:
                    distances[nbr] = dist_fun(destination, best_non, layout) 
                else:
                    distances[nbr] = 100000000000
            

        if(blending_param != -1):
            pre_calc_dist = {nbr : ppr_and_dist_priority(destination, nbr, layout, current_ppr, dist_fun, blending_param=blending_param) for nbr in frontier}
            frontier.sort(key = lambda nbr: pre_calc_dist[nbr])
        else:
            key = (lambda nbr: dist_fun(destination, nbr, layout)) if not with_non else (lambda nbr: distances[nbr])
            frontier.sort(key = key)

        if(neighbors_count == 0 and (befs_w == -1 or len(frontier) == 0)):
            (beam_stack, beams, path_len) = beam_backtrack(beam_stack, path_len, beam_width)

            if(path_len <= 0):
                print("Impossible to find a path from {a} to {b}".format(a = source, b = destination))
                return None

            for beam in beams:
                visited_nodes[beam] = beam_stack[path_len-1].neighbors_parent[beam]
            continue

        new_beams_count = min(beam_width, len(frontier))    
        beams = frontier[:new_beams_count]
        
        for beam in beams:
            visited_nodes[beam] = parent_dict[beam]
            if(befs_w == UNLIMITED):
                frontier.remove(beam)
                parent_dict.pop(beam)

        if(befs_w != -1 and befs_w != UNLIMITED):
            aux2 = []
            aux = {}
            for i in range(new_beams_count, new_beams_count + befs_w):
                if(len(frontier) > i):
                    aux[frontier[i]] = parent_dict[frontier[i]]
                    aux2.append(frontier[i])
            befs_frontier_parent = aux
            befs_frontier = aux2 
        
        if(len(beam_stack) <= path_len):
            beam_stack.append(BeamStackItem(frontier, parent_dict, new_beams_count - 1 ))

        path_len += 1

    #reconstruct path
    if(destination in visited_nodes):
        path = [destination]
        while path[0] != source:
            path.insert(0, visited_nodes[path[0]])


        if(find_more_paths != None and find_more_paths != AugmentingPathMode.CHANGE_STOP_CRITERIA):
            add_other_paths(G, layout, explored_graph=explored_graph, evaluated_nodes=evaluated_nodes, source=source, destination=destination, visited_nodes=visited_nodes, path_len=path_len, dist_fun=dist_fun, mode=find_more_paths)

        res = SearchResult(path, explored_graph, len(path)-1, len(visited_nodes), len(evaluated_nodes), ppr_results = ppr.PPR(explored_graph, source, 0.85, current_ppr, 1) if should_calculate_ppr else None, evaluated_edges= explored_graph.number_of_edges())
        return res
    else: 
        return None
    
def beam_layout_oriented_with_backtrack_clean(G: nx.Graph, layout, source, destination, beam_width, dist_fun = cartesian_dist_in_layout):
    print("Looking for a way to go from {a} to {b} ...".format(a = source, b = destination))
    evaluated_nodes = set()

    #dict containing as key visited nodes and as value his predecessor
    visited_nodes = dict({source : -1})
    explored_graph = nx.Graph()

    beams = [source]
    path_len = 0
    beam_stack = []

    while destination not in visited_nodes: #fissare un numero di nodi 
        
        neighbors = []
        neighbors_parent = dict()
        neighbors_count = 0

        for beam in beams:
            for nbr in G.neighbors(beam):
                explored_graph.add_edge(nbr, beam, capacity=1)
                evaluated_nodes.add(nbr)

                if nbr in visited_nodes or nbr in neighbors_parent:
                    continue

                neighbors_count+=1
                neighbors.append(nbr)
                neighbors_parent[nbr] = beam
        
        neighbors.sort(key = lambda nbr: dist_fun(destination, nbr, layout))

        if(neighbors_count == 0):
            (beam_stack, beams, path_len) = beam_backtrack(beam_stack, path_len, beam_width)

            if(path_len <= 0):
                print("Impossible to find a path from {a} to {b}".format(a = source, b = destination))
                return None

            for beam in beams:
                visited_nodes[beam] = beam_stack[path_len-1].neighbors_parent[beam]
            continue

        new_beams_count = min(beam_width, len(neighbors))    
        beams = neighbors[:new_beams_count]
        
        for beam in beams:
            visited_nodes[beam] = neighbors_parent[beam]
        
        if(len(beam_stack) <= path_len):
            beam_stack.append(BeamStackItem(neighbors, neighbors_parent, new_beams_count - 1 ))

        path_len += 1

    #reconstruct path
    if(destination in visited_nodes):
        path = [destination]
        while path[0] != source:
            path.insert(0, visited_nodes[path[0]])
        assert(explored_graph.number_of_nodes() == len(evaluated_nodes))
        res = SearchResult(path, explored_graph, len(path)-1, len(visited_nodes), len(evaluated_nodes), None, explored_graph.number_of_edges())
        return res
    else: 
        return None
    

def shortest_path_on_sub_graph(G: nx.Graph, layout, source, destination, search):
    aux = search(G, layout, source, destination)
    aux.path_len = nx.shortest_path_length(aux.explored_graph, source, destination)
    return aux 

def BeFS_shortest_path_on_sub_graph(G: nx.Graph, layout, source, destination):
    aux = beam_layout_oriented_with_backtrack(G, layout, source, destination, 1, befs_w=15, with_non=True, non_to_sample=5)
    aux.path_len = nx.shortest_path_length(aux.explored_graph, source, destination)
    return aux 

def dfs_layout_oriented(G: nx.Graph, layout, source, destination):
    return beam_layout_oriented_with_backtrack_clean(G, layout, source, destination, 1)

def unlimited_befs(G: nx.Graph, layout, source, destination):
    return beam_layout_oriented_with_backtrack(G, layout, source, destination, 1, befs_w=UNLIMITED)

def limited_befs(G: nx.Graph, layout, source, destination):
    return beam_layout_oriented_with_backtrack(G, layout, source, destination, 1, befs_w=15)

def limited_befs_with_non(G: nx.Graph, layout, source, destination):
    return beam_layout_oriented_with_backtrack(G, layout, source, destination, 1, befs_w=15, with_non=True)

def limited_befs_with_nonrs(G: nx.Graph, layout, source, destination):
    return beam_layout_oriented_with_backtrack(G, layout, source, destination, 1, befs_w=15, with_non=True, non_to_sample=5)

def dfs_layout_oriented_with_non(G: nx.Graph, layout, source, destination, non_to_sample = -1):
    return beam_layout_oriented_with_backtrack_and_non(G, layout, source, destination, 1, non_to_sample=non_to_sample)

def dfs_layout_oriented_with_non_decs(G: nx.Graph, layout, source, destination):
    return beam_layout_oriented_with_backtrack_and_non_without_optimization(G, layout, source, destination, 1)


def augmented_dfs_layout_oriented(G: nx.Graph, layout, source, destination):
    return beam_layout_oriented_with_backtrack(G, layout, source, destination, 1, find_more_paths=AugmentingPathMode.TOWARDS_SOURCE)

def augmented_pingpong_dfs_layout_oriented(G: nx.Graph, layout, source, destination):
    return beam_layout_oriented_with_backtrack(G, layout, source, destination, 1, find_more_paths=AugmentingPathMode.PING_PONG)

def augmented_dfs_layout_oriented_with_non(G: nx.Graph, layout, source, destination):
    return beam_layout_oriented_with_backtrack_and_non(G, layout, source, destination, 1, True)


def beam_layout_oriented_with_backtrack_and_non(G: nx.Graph, layout, source, destination, beam_width, find_more_paths: bool = False, dist_fun = cartesian_dist_in_layout, non_to_sample: int = -1, befs_w = -1):
    print("Looking for a way to go from {a} to {b} ...".format(a = source, b = destination))
    evaluated_nodes = set()
    visited_nodes = dict({source : -1})
    explored_graph = nx.Graph()

    beams = [source]
    path_len = 0
    beam_stack = []

    befs_frontier = []
    befs_frontier_parent = {}

    while destination not in beams:

        first_degree_neighbours = []
        neighbors_parent = dict()
        neighbors_count = 0
        destination_found = False

        for beam in beams:
            for nbr in G.neighbors(beam):
                explored_graph.add_edge(nbr, beam, capacity=1)
                evaluated_nodes.add(nbr)

                if nbr in visited_nodes or nbr in neighbors_parent:
                    continue

                if nbr == destination:
                    destination_found = True

                neighbors_count += 1
                first_degree_neighbours.append(nbr)
                neighbors_parent[nbr] = beam
            
        if destination_found:
            path_len += 1
            beams = [destination]
            visited_nodes[destination] = neighbors_parent[destination]
            break

        nons = []
        nons_count = 0
        for nbr in first_degree_neighbours:

            #There's an optimization here because it does not consider neighbours that are also first_degree and neither nons already picked by others
            #in a decentralized system this would not be possible of course
    
            best_non = get_nearest_neighbor(G, layout, evaluated_nodes, explored_graph, visited_nodes, neighbors_parent, nbr, destination, non_to_sample)
            if(best_non != None):
                nons_count += 1
                if(best_non not in neighbors_parent):
                    neighbors_parent[best_non] = nbr
                nons.append(best_non)
            
        nons.sort(key = lambda nbr_index: dist_fun(destination, nbr_index, layout))

        if(nons_count == 0):
            (beam_stack, beams, path_len) = beam_backtrack(beam_stack, path_len, beam_width)
            if(path_len == 0):
                print("Impossible to find a path from {a} to {b}".format(a = source, b = destination))
                return None

            for beam in beams:
                visited_nodes[beam] = beam_stack[path_len-1].neighbors_parent[beam]
            continue

        counter = 0
        beams = []
        sorted_nbrs = []

        for non in nons:
            node_to_add = (neighbors_parent[non] if non not in first_degree_neighbours else non)
            if(counter < beam_width):
                counter+=1
                beams.append(node_to_add)
            sorted_nbrs.append(node_to_add)
        
        for beam in beams:
            visited_nodes[beam] = neighbors_parent[beam]
        
        if(len(beam_stack) <= path_len):
            beam_stack.append(BeamStackItem(sorted_nbrs, neighbors_parent, min(beam_width, nons_count)  - 1 ))

        path_len += 1

    #reconstruct path
    path = [destination]
    while path[0] != source:
        path.insert(0, visited_nodes[path[0]])

    if(find_more_paths):
        add_other_paths(G, layout, explored_graph=explored_graph, evaluated_nodes=evaluated_nodes, source=source, destination=destination, visited_nodes=visited_nodes, path_len=path_len)
    
    res = SearchResult(path, explored_graph, path_len, len(visited_nodes), len(evaluated_nodes))
    return res


def beam_layout_oriented_with_backtrack_and_non_without_optimization(G: nx.Graph, layout, source, destination, beam_width):
    print("Looking for a way to go from {a} to {b} ...".format(a = source, b = destination))
    evaluated_nodes = set()
    visited_nodes = dict({source : -1})
    explored_graph = nx.Graph()

    beams = [source]
    path_len = 0
    beam_stack = []

    while destination not in beams:

        first_degree_neighbours = []
        neighbors_parent = dict()
        neighbors_count = 0
        destination_found = False

        for beam in beams:
            for nbr in G.neighbors(beam):
                explored_graph.add_edge(nbr, beam, capacity=1)
                evaluated_nodes.add(nbr)

                if nbr in visited_nodes or nbr in neighbors_parent:
                    continue

                if nbr == destination:
                    destination_found = True

                neighbors_count += 1
                first_degree_neighbours.append(nbr)
                neighbors_parent[nbr] = beam
            
        if destination_found:
            path_len += 1
            beams = [destination]
            visited_nodes[destination] = neighbors_parent[destination]
            break

        nons = []
        nons_count = 0
        for nbr in first_degree_neighbours:
            
            best_non = get_nearest_neighbor(G, layout, evaluated_nodes, explored_graph, visited_nodes, dict(), nbr, destination)
            if(best_non != None):
                nons_count += 1
                nons.append((best_non, nbr))
            
        nons.sort(key = lambda non: cartesian_dist_in_layout(destination, non[0], layout))

        if(nons_count == 0):
            (beam_stack, beams, path_len) = beam_backtrack(beam_stack, path_len, beam_width)
            if(path_len == 0):
                print("Impossible to find a path from {a} to {b}".format(a = source, b = destination))
                return None

            for beam in beams:
                visited_nodes[beam] = beam_stack[path_len-1].neighbors_parent[beam]
            continue

        counter = 0
        beams = []
        sorted_nbrs = []

        for non in nons:
            if(counter < beam_width):
                counter+=1
                beams.append(non[1])
            sorted_nbrs.append(non[1])
        
        for beam in beams:
            visited_nodes[beam] = neighbors_parent[beam]
        
        if(len(beam_stack) <= path_len):
            beam_stack.append(BeamStackItem(sorted_nbrs, neighbors_parent, min(beam_width, nons_count)  - 1 ))

        path_len += 1

    #reconstruct path
    path = [destination]
    while path[0] != source:
        path.insert(0, visited_nodes[path[0]])

    return SearchResult(path, explored_graph, path_len, len(visited_nodes), len(evaluated_nodes))


def beam_layout_oriented_with_backtrack_and_non_support_befs(G: nx.Graph, layout, source, destination, beam_width, find_more_paths: bool = False, dist_fun = cartesian_dist_in_layout, non_to_sample: int = -1, befs_w = -1):
    print("Looking for a way to go from {a} to {b} ...".format(a = source, b = destination))
    evaluated_nodes = set()
    visited_nodes = dict({source : -1})
    explored_graph = nx.Graph()

    beams = [source]
    path_len = 0
    beam_stack = []

    befs_frontier = []
    befs_frontier_parent = {}

    while destination not in beams:

        first_degree_neighbours = []
        neighbors_parent = dict()
        neighbors_count = 0
        destination_found = False

        for beam in beams:
            for nbr in G.neighbors(beam):
                explored_graph.add_edge(nbr, beam, capacity=1)
                evaluated_nodes.add(nbr)

                if nbr in visited_nodes or nbr in neighbors_parent:
                    continue

                if nbr == destination:
                    destination_found = True

                neighbors_count += 1
                first_degree_neighbours.append(nbr)
                neighbors_parent[nbr] = beam
            
        if destination_found:
            path_len += 1
            beams = [destination]
            visited_nodes[destination] = neighbors_parent[destination]
            break

        nons = []
        nons_count = 0
        for nbr in first_degree_neighbours:
    
            best_non = get_nearest_neighbor(G, layout, evaluated_nodes, explored_graph, visited_nodes, neighbors_parent, nbr, destination, non_to_sample)
            if(best_non != None):
                nons_count += 1
                if(nbr not in neighbors_parent):
                    neighbors_parent[best_non] = nbr
                nons.append(best_non)
            
        nons.sort(key = lambda nbr_index: dist_fun(destination, nbr_index, layout))

        if(nons_count == 0):
            (beam_stack, beams, path_len) = beam_backtrack(beam_stack, path_len, beam_width)
            if(path_len == 0):
                print("Impossible to find a path from {a} to {b}".format(a = source, b = destination))
                return None

            for beam in beams:
                visited_nodes[beam] = beam_stack[path_len-1].neighbors_parent[beam]
            continue

        counter = 0
        beams = []
        sorted_nbrs = []

        for non in nons:
            node_to_add = (neighbors_parent[non] if non not in first_degree_neighbours else non)
            if(counter < beam_width):
                counter+=1
                beams.append(node_to_add)
            sorted_nbrs.append(node_to_add)
        
        for beam in beams:
            visited_nodes[beam] = neighbors_parent[beam]
        
        if(len(beam_stack) <= path_len):
            beam_stack.append(BeamStackItem(sorted_nbrs, neighbors_parent, min(beam_width, nons_count)  - 1 ))

        path_len += 1

    #reconstruct path
    path = [destination]
    while path[0] != source:
        path.insert(0, visited_nodes[path[0]])

    if(find_more_paths):
        add_other_paths(G, layout, explored_graph=explored_graph, evaluated_nodes=evaluated_nodes, source=source, destination=destination, visited_nodes=visited_nodes, path_len=path_len)
    
    res = SearchResult(path, explored_graph, path_len, len(visited_nodes), len(evaluated_nodes))
    return res


def dfs_layout_oriented_old(G: nx.Graph, layout, source_index, destination_index):
    print("DFS - Looking for a way to go from {a} to {b} ...".format(a = source_index, b = destination_index))
    visited_nodes = set([source_index])
    back_track = deque()

    while source_index != destination_index:
        neighbors = []
        for nbr in G.neighbors(source_index):
            if nbr in visited_nodes:
                continue
            neighbors.append(nbr)

        if(len(neighbors) == 0):
            if(len(back_track) == 0):
                print("Impossible to find a path from {a} to {b}".format(a = source_index, b = destination_index))
                return
            else:
                source_index = back_track.pop()
                continue

        back_track.append(source_index)
        neighbors.sort(key = lambda nbr_indx: cartesian_dist_in_layout(destination_index, nbr_indx, layout))
        source_index = neighbors[0]
        visited_nodes.add(source_index)

    back_track.append(source_index)
    print("Path -> {a},\nLen -> {b} (source included),\n# of Visited nodes -> {c}".format(a=back_track, b=len(back_track)-1, c=len(visited_nodes)))
    return SearchResult(explored_graph=None, path= back_track, path_len=len(back_track)-1, visited_nodes=len(visited_nodes), evaluated_nodes=0)


def dfs_classic(G: nx.Graph, layout, source_index, destination_index):
    stack = deque([(source_index, None)])
    parents = dict()

    while len(stack) != 0:
        (node, parent) = stack.pop()
        if node in parents:
            continue
        parents[node] = parent

        if(node == destination_index):
            res = [node]
            while parents[node] != None:
                node = parents[node]
                res.append(node)
            return SearchResult(explored_graph=None, path=res[::-1], path_len=len(res)-1, visited_nodes=len(parents), evaluated_nodes=0)
        
        neighbors = [n for n in G.neighbors(node) if n not in parents]
        neighbors.sort(key = lambda nbr_indx: cartesian_dist_in_layout(nbr_indx, destination_index, layout), reverse=True)
        for n in neighbors:
            stack.append((n, node))
    
    return None

def get_nearest_neighbor_distance(G, layout, visited_nodes, source, dest):

    if(dest == source):
        return -1
    
    neighbours = [n for n in G.neighbors(source) if n not in visited_nodes]
    if(len(neighbours) == 0):
        return 10000000000000000
    
    neighbours.sort(key = lambda nbr_index: cartesian_dist_in_layout(nbr_index, dest, layout))
    return cartesian_dist_in_layout(neighbours[0], dest, layout)

def dfs_with_non_classic(G: nx.Graph, layout, source_index, destination_index):
    stack = deque([(source_index, None)])
    parents = dict()
    while len(stack) != 0:
        (node, parent) = stack.pop()
        if node in parents:
            continue
        parents[node] = parent

        if(node == destination_index):
            res = [node]
            while parents[node] != None:
                node = parents[node]
                res.append(node)
            return SearchResult(explored_graph=None, path=res[::-1], path_len=len(res)-1, visited_nodes=len(parents), evaluated_nodes=0)
        
        neighbors = [n for n in G.neighbors(node) if n not in parents]

        non_distances = dict()
        for nbr in neighbors:
            non_distance = get_nearest_neighbor_distance(G, layout, parents, nbr, destination_index)
            non_distances[nbr] = non_distance

        neighbors.sort(key = lambda nbr_indx: non_distances[nbr_indx], reverse=True)
        for n in neighbors:
            stack.append((n, node))
    
    return None




def add_other_paths(G: nx.Graph, layout, explored_graph: nx.Graph, evaluated_nodes: set, source, destination, visited_nodes, path_len : int, dist_fun = cartesian_dist_in_layout, mode : AugmentingPathMode = AugmentingPathMode.TOWARDS_SOURCE):

    print(mode)
    STEPS = max(100, path_len)
    i = 0
    current_source = destination
    current_destination = source
    current_node = current_source

    triggered_turn_back = set()  # Take track of edges that have triggered a turn back to avoid infinite ping-ponging
    loop_guard = 0
    max_loops = 2000             # Avoid freezing in case of unexpected behavior

    while (i < STEPS if mode == AugmentingPathMode.TOWARDS_SOURCE else len(visited_nodes) < BUDGET):
        loop_guard += 1
        if loop_guard > max_loops:
            break

        neighbors = []
        neighbors_parent = dict()
        neighbors_count = 0

        turn_back = False

        for nbr in G.neighbors(current_node):
            explored_graph.add_edge(nbr, current_node, capacity=1)
            evaluated_nodes.add(nbr)

            if nbr is current_destination:
                # Create a unique key for the edge to track turn_back
                edge_key = tuple(sorted((current_node, nbr)))
                if edge_key not in triggered_turn_back:
                    triggered_turn_back.add(edge_key)
                    turn_back = True
                continue
            
            if nbr in visited_nodes or nbr in neighbors_parent:
                continue

            neighbors_count+=1
            neighbors.append(nbr)
            neighbors_parent[nbr] = current_node

        if(mode == AugmentingPathMode.PING_PONG and turn_back):
            aux = current_source
            current_source = current_destination
            current_destination = aux
            current_node = current_source
            continue

        if(neighbors_count < 1):
            if(current_node == current_source): #no chance to backtrack
                break
            #BackTrack
            current_node = visited_nodes[current_node]
            i+=1
            break

        neighbors.sort(key = lambda nbr: dist_fun(source, nbr, layout))
        current_node = neighbors[0]
        visited_nodes[current_node] = neighbors_parent[current_node]
        i+=1

def rooted_spanning_tree_search(G: nx.Graph, layout, source, destination, considered_tree = 3, use_trees_together : bool = False, select_nearer_trees = False, befs_w =-1, nonrs = -1, augmenting_mode = None, beam = 1):
    if(use_trees_together):
        return beam_layout_oriented_with_backtrack(G, random.sample(list(layout.values()), considered_tree), source, destination, beam, dist_fun=tree_distance_multi_layout, befs_w=befs_w, non_to_sample=nonrs, find_more_paths= augmenting_mode)
    
    if select_nearer_trees:
        sorted_layout_by_distance_from_dest = sorted(layout.values(), key = lambda x: tree_distance(destination, source, x))
        layouts_to_use = sorted_layout_by_distance_from_dest[:considered_tree] if considered_tree < len(layout.values()) else sorted_layout_by_distance_from_dest
    else:
        layouts_to_use = random.sample(list(layout.values()), considered_tree)
    results = [beam_layout_oriented_with_backtrack(G, extracted_layout, source, destination, beam, dist_fun = tree_distance, befs_w=befs_w, non_to_sample=nonrs, find_more_paths= augmenting_mode) for extracted_layout in layouts_to_use]

    if (considered_tree > 1):
        min_path = min([(res.path, res.path_len) for res in results], key=lambda x: x[1])
        visited_nodes = len(set([x for res in results for x in res.path]))
        explored_graph = nx.compose_all([res.explored_graph for res in results])
        evaluated_nodes = explored_graph.number_of_nodes()
        return SearchResult(path=min_path[0], explored_graph=explored_graph, path_len=min_path[1], visited_nodes=visited_nodes, evaluated_nodes=evaluated_nodes)
    else:
        return results[0]
    





def ppr_and_dist_search(G: nx.Graph, layout, source, destination, pre_calc_ppr = None, blending_param = 0.5):
    return beam_layout_oriented_with_backtrack(G, layout, source, destination, 1, blending_param=blending_param, pre_calc_ppr=pre_calc_ppr)

def ppr_and_dist_search_befs(G: nx.Graph, layout, source, destination, pre_calc_ppr = None, blending_param = 0.5, befs_w = 15):
    return beam_layout_oriented_with_backtrack(G, layout, source, destination, 1, blending_param=blending_param, befs_w=befs_w, pre_calc_ppr=pre_calc_ppr, find_more_paths=AugmentingPathMode.CHANGE_STOP_CRITERIA)
