import random
import networkx as nx
from collections import defaultdict, Counter
from discrete_event_sim import Simulation
import network_events as evt
from search_algorithms import (rooted_spanning_tree_search, limited_befs_with_nonrs, shortest_path_on_sub_graph, dfs_layout_oriented, augmented_pingpong_dfs_layout_oriented, AugmentingPathMode)


class DistributedNetwork(Simulation):
    def __init__(self, n_nodes, join_rate, leave_rate, trade_rate, 
                 trust_threshold=0.4, default_rep=0.5, malicious_rate=0.2, 
                 poolTrade=5, punishment="none", snap_interval = 5.0, 
                 start_connection=3, algorithm="dijkstra", ping_pong_choice="ppr", embedded_type= "none", 
                 emb_dimensions=100, consensus_threshold=0.66,
                 embedding_interval=5.0, attack_type="none"):
        """
        Initializes the distributed network simulation with specified parameters.
        """
        
        super().__init__()

        # Simulation parameters and general configuration
        self.n_nodes = n_nodes                       
        self.next_id = 0
        self.start_connection = start_connection  # Number of initial connections for new nodes
        self.join_rate = join_rate
        self.leave_rate = leave_rate
        self.trade_rate = trade_rate
        self.default_rep = default_rep            # Default reputation value when a new node joins the network 
        self.trust_threshold = trust_threshold    # Threshold for determining trustworthiness in the network
        self.snap_interval = snap_interval        # Snap for the metrics
        self.embedding_interval = embedding_interval 
        self.algorithm = algorithm.lower()
        self.ping_pong_choice = ping_pong_choice.lower()
        self.poolTrade = poolTrade                # Amount of nodes to sample for the transaction (from the pool we analyze the best spot to do the transaction) 

        # Attack configuration and countermeasures
        self.malicious_rate = malicious_rate      # Percentage of malicious nodes in the network
        self.attack_type = attack_type.lower()    # Type of attack to simulate
        self.punishment = punishment              # Whether to enable punishment mechanisms for malicious nodes
        self.current_sender = None                # For targeted attack countermeasures
        self.consensus_threshold = consensus_threshold  # Consensus threshold for accepting information from neighbors in countermeasures against targeted attacks

        # Network data structures (O(1))
        self.active_nodes = set()                                    # Hash set               -> Active_nodes(42) = True or Active_nodes(42) = False   O(1)
        self.active_nodes_list = []                                  # Dynamic array          -> Take directly a random element in O(1)
        self.node_to_index = {}                                      # Hash map               -> Value -> position in the active_nodes_list for O(1)
        self.reputation = defaultdict(dict)                          # Nested Hash Map        -> Reputation[src][dst] = weight in O(1)
        self.in_edges = defaultdict(set)                             # Adjacency Map of Sets  -> in_edges[dst] = {src1, src2, ...} in O(1)
        self.trade_counts = defaultdict(lambda: defaultdict(int))    # Double-Nested Hash Map -> trade_counts[src][dst] = count in O(1)
        self.node_behaviors = {}                                     # Persistent Hash Map    -> node_id: "honest" or "malicious" in O(1)
        self.total_node_received_trades = defaultdict(int)           # Hash Map               -> total_node_received_trades[node_id] = count in O(1)

        # Embedding and layouts
        self.embedded_type = embedded_type.lower()                  # Type of embedding used in the network ("none", "stge", "socs")
        self.emb_dimensions = emb_dimensions                        # Number of dimensions for the embedding space (used in network events)
        self.layout = {}                                            # Hash Map               -> node_id: embedding_vector in O(1)
        self.stge_layouts = {}                                      # Will store the full multi-tree dictionary for STGE

        # Metrics and statistics
        self.total_attempts = 0                                     # Total number of transaction attempts in the network
        self.failed_pathfinding = 0                                 # Total number of failed pathfinding attempts in the network
        self.trusted_trades = 0                                     # Total number of trusted trades in the network
        self.untrusted_trades = 0                                   # Total number of untrusted trades in the network
        self.honest_good_trades = 0                                 # Total number of good trades performed by honest nodes
        self.honest_bad_trades = 0                                  # Total number of bad trades performed by honest nodes
        self.malicious_good_trades = 0                              # Total number of good trades performed by malicious nodes
        self.malicious_bad_trades = 0                               # Total number of bad trades performed by malicious nodes
        self.malicious_scams_count = defaultdict(int)               # Counts scams performed by each node after the 5 honest trades
        self.completed_scams_records = []                           # Stores how many scams each isolated node has done
        self.honest_nodes_isolation = 0                             # Counts how many honest nodes have been isolated in the network
        self.malicious_nodes_isolation = 0                          # Counts how many malicious nodes have been isolated in the network
        self.isolated_nodes = set()                                 # Set to track isolated nodes for analysis
        self.mean_trades_honest = [0,0]                             # Pos 1 Mean for honest nodes, pos 2 Count for honest nodes
        self.mean_trades_malicious = [0,0]                          # Pos 1 Mean for malicious nodes, pos 2 Count for malicious nodes
        self.honest_src_trades = 0                                  # Total transactions initiated by honest source nodes
        self.malicious_src_trades = 0                               # Total transactions initiated by malicious source nodes
        self.honest_dst_trades = 0                                  # Total transactions targeting honest destination nodes
        self.malicious_dst_trades = 0                               # Total transactions targeting malicious destination nodes

        # History for analysis and visualization
        self.history = {
            't': [], # Time t
            'trusted': [], # Successful trades at time t 
            'untrusted': [], # Unsuccessful trades at time t 
            'success_rate': [], # Percentage of successful trades over total attempts at time t
            'avg_scams_per_node': [] # Avg Scam At time t 
        }

        # Persistent graph 
        self.G = nx.DiGraph()
 
        # Initial nodes join the system
        for _ in range(n_nodes):
            self._add_node()

        self.schedule(random.expovariate(join_rate), evt.NodeJoin())
        self.schedule(0, evt.StatSnapshot())

        if self.embedded_type != "none":
            self.schedule(0, evt.EmbeddingUpdate())

    def _add_node_to_active(self, node_id):
        """
        Adds a node to active structures and ensures the node exists in the graph G.
        """

        if node_id not in self.active_nodes:
            self.active_nodes.add(node_id)
            self.node_to_index[node_id] = len(self.active_nodes_list) # If list contain [0,1,2,3,4], the position of the new node will be 5
            self.active_nodes_list.append(node_id)                    # Add the node for take it randomely in O(1)
            self.G.add_node(node_id)                                  # Add the node in the graph because is active now

    def _remove_node_from_active(self, node_id):
        """
        Removes a node from active structures using swap-with-last logic and removes edges from G.
        """

        if node_id in self.active_nodes:                              

            # Save the index where is the node to remove and the last node in the list for the swap
            idx = self.node_to_index[node_id]                        # Position of the node to remove
            last_node = self.active_nodes_list[-1]                   # Last node in the list for the swap
            
            # Swap without save the node to remove because we will pop it
            self.active_nodes_list[idx] = last_node                  # Put the last node in the position of the node to remove
            self.node_to_index[last_node] = idx                      # Update the index of the last node to its new position
            
            # Pop the last element
            self.active_nodes_list.pop()                             # Duplicated so we pop 
            del self.node_to_index[node_id]                          # Remove the node from the index map
            self.active_nodes.discard(node_id)                       # Remove the node from the active set
            
            # Remove the node from the graph G if it exists
            if self.G.has_node(node_id):
                self.G.remove_node(node_id)

    def _add_node(self):
        """
        Adds a new node to the network, creates initial connections, and schedules its first events.
        Returns the new node's ID.
        """

        node_id = self.next_id
        self.next_id += 1
        
        # Assign behavior 

        self.node_behaviors[node_id] = "malicious" if random.random() < self.malicious_rate else "honest"
        
        self._add_node_to_active(node_id)                                              # Call the method to add the node in active structures
        
        # Initial connections using sampling from active nodes
        if len(self.active_nodes_list) > 1:
            n_to_sample = min(len(self.active_nodes_list) - 1, self.start_connection)  # Sample up to n neighbors, but not more than the number of active nodes minus one (to avoid sampling itself)
            others = random.sample(self.active_nodes_list, n_to_sample + 1)            # Take 2 or 3 random nodes, i add + 1 because it could take the new node itself
            initial_neighbors = [n for n in others if n != node_id][:n_to_sample]      # [10,20,30][:2] -> [10,20] or [10,new_node,30][:2] -> [10,30]
            
            for neighbor in initial_neighbors:
                # Bidirectional initial connection with default value
                self.reputation[node_id][neighbor] = self.default_rep
                self.reputation[neighbor][node_id] = self.default_rep
                self.in_edges[neighbor].add(node_id)
                self.in_edges[node_id].add(neighbor)
                
                # Check threshold before adding to the graph
                self.update_reputation(node_id, neighbor, -1)
                self.update_reputation(neighbor, node_id, -1)
                self.log_info(f"Initial connection: Node {node_id} with Node {neighbor}")

        self.schedule(random.expovariate(self.trade_rate), evt.Transaction(node_id))
        self.schedule(random.expovariate(self.leave_rate), evt.NodeLeave(node_id))
        return node_id

    def _check_isolation(self, node_id):
        """ Checks if a malicious/honest node has lost all incoming trust edges in G. """

        if node_id in self.isolated_nodes:
            return

        if (self.node_behaviors.get(node_id) == "malicious" and 
            self.G.has_node(node_id) and 
            self.G.in_degree(node_id) == 0):
            
            # Record how many scams it did before isolation
            scams = self.malicious_scams_count.pop(node_id, 0)
            self.completed_scams_records.append(scams)
            self.malicious_nodes_isolation += 1
            self.isolated_nodes.add(node_id)
            self.log_info(f"Isolation: Node {node_id} isolated after {scams} scams.")

        elif (self.node_behaviors.get(node_id) == "honest" and 
                  self.G.has_node(node_id) and 
                  self.G.in_degree(node_id) == 0):
            self.honest_nodes_isolation += 1
            self.isolated_nodes.add(node_id)
            self.log_info(f"Isolation: Honest Node {node_id} isolated.")

    def find_path_metric(self, start_node, target_node): 
        """
        Finds the metric for the start_node and target_node.
         - For Dijkstra: returns (path, total_weight)
         - For Max Flow: returns (None, flow_value)
         - For PPR: returns (None, ppr_scores_dict)
         - For Tree/Geometric Embeddings (STGE/SoCS): returns (path, SearchResult_object)
           The SearchResult object contains path_len, explored_graph, ppr_results, etc.
        """

        # Check if the start node exists in the graph, if not return None for path and metric
        if not self.G.has_node(start_node):
            return None, None
            
        # For Dijkstra, Max Flow and embedding algorithms, we also need to check if the target node exists in the graph before attempting to find a path. 
        # For PPR, we can compute scores for all nodes even if the target node is not active, but we will only consider candidates that are active.
        if self.algorithm != "ppr" and not self.G.has_node(target_node):
            return None, None
            

        
        # No Embedding Active (Standard NetworkX Algorithms)
        if self.embedded_type == "none":
            try:
                if self.algorithm == "dijkstra":
                    # Dijkstra: lower cost = better
                    cost, path = nx.single_source_dijkstra(self.G, source=start_node, target=target_node, weight='weight')
                    return path, cost
                elif self.algorithm == "max_flow":
                    # Max Flow: higher flow = better
                    flow_value, _ = nx.maximum_flow(self.G, start_node, target_node, capacity='capacity', flow_func=nx.algorithms.flow.edmonds_karp)
                    return None, flow_value
                elif self.algorithm == "ppr":
                    # Personalized PageRank: higher score = better
                    ppr_scores = nx.pagerank(self.G, alpha=0.85, personalization={start_node: 1}, weight='weight')  # Compute PPR scores personalized to the start_node and return a dictionary of node: score
                    return None, ppr_scores
            except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, ValueError, IndexError):
                return None, None


        # Tree Embedding (STGE)
        elif self.embedded_type == "stge":
            try:
                if self.algorithm == "befs":
                    # Standard Tree BeFS with queue size 15
                    res = rooted_spanning_tree_search(
                        G=self.G, layout=self.stge_layouts, source=start_node, destination=target_node,
                        considered_tree=1, select_nearer_trees=True, befs_w=15
                    )
                    return (res.path, res) if (res and hasattr(res, 'path')) else (None, None) # List path and the SearchResult object with path_len, explored_graph, ppr_results, etc.
                    
                elif self.algorithm == "greedy_dfs":
                    # Standard Tree Greedy DFS
                    res = rooted_spanning_tree_search(
                        G=self.G, layout=self.stge_layouts, source=start_node, destination=target_node,
                        considered_tree=1, select_nearer_trees=True
                    )
                    return (res.path, res) if (res and hasattr(res, 'path')) else (None, None) # List path and the SearchResult object with path_len, explored_graph, ppr_results, etc.
                    
                elif self.algorithm == "ping_pong_dfs":
                    # Create a subgraph using ping-pong DFS search on STGE
                    res = rooted_spanning_tree_search(
                        G=self.G, layout=self.stge_layouts, source=start_node, destination=target_node,
                        considered_tree=1, select_nearer_trees=True, augmenting_mode=AugmentingPathMode.PING_PONG
                    )
                    if res and hasattr(res, 'explored_graph'): # If the subgraph exists we can compute on it the max flow or the ppr score
                        if self.ping_pong_choice == "max_flow":
                            flow_value, _ = nx.maximum_flow(res.explored_graph, start_node, target_node, capacity='capacity', flow_func=nx.algorithms.flow.edmonds_karp)
                            res.flow_value = flow_value
                            return res.path, res # List path and the SearchResult object with path_len, explored_graph, ppr_results, etc.                   
                        elif self.ping_pong_choice == "ppr":
                            ppr_scores = nx.pagerank(res.explored_graph, alpha=0.85, personalization={start_node: 1})
                            res.ppr_score = ppr_scores.get(target_node, 0.0)
                            return res.path, res # List path and the SearchResult object with path_len, explored_graph, ppr_results, etc.                   
                    return (res.path, res) if (res and hasattr(res, 'path')) else (None, None)
            except Exception:
                return None, None

        # Social Coordinates Embedding (SoCS)

        elif self.embedded_type == "socs":
            try:
                if self.algorithm == "befs_nonrs":
                    # BeFS with NoNRS Optimization on SoCS
                    res = limited_befs_with_nonrs(
                        G=self.G, layout=self.layout, source=start_node, destination=target_node
                    )
                    return (res.path, res) if (res and hasattr(res, 'path')) else (None, None) # List path and the SearchResult object with path_len, explored_graph, ppr_results, etc.
                    
                elif self.algorithm == "greedy_dfs_sps":
                    # Greedy DFS with SPS Subgraph Smoothing on SoCS
                    res = shortest_path_on_sub_graph(
                        G=self.G, layout=self.layout, source=start_node, destination=target_node,
                        search=dfs_layout_oriented
                    )
                    return (res.path, res) if (res and hasattr(res, 'path')) else (None, None) # List path and the SearchResult object with path_len, explored_graph, ppr_results, etc.
                    
                elif self.algorithm == "ping_pong_dfs":  
                    # Create a subgraph using ping-pong DFS search on SoCS
                    res = augmented_pingpong_dfs_layout_oriented(
                        G=self.G, layout=self.layout, source=start_node, destination=target_node
                    )
                    if res and hasattr(res, 'explored_graph'): # Same as before if exists we can compute the max flow or the ppr score on the subgraph
                        if self.ping_pong_choice == "max_flow":
                            flow_value, _ = nx.maximum_flow(res.explored_graph, start_node, target_node, capacity='capacity', flow_func=nx.algorithms.flow.edmonds_karp)
                            res.flow_value = flow_value
                            return res.path, res # List path and the SearchResult object with path_len, explored_graph, ppr_results, etc.
                    
                        elif self.ping_pong_choice == "ppr":
                            ppr_scores = nx.pagerank(res.explored_graph, alpha=0.85, personalization={start_node: 1})
                            res.ppr_score = ppr_scores.get(target_node, 0.0)
                            return res.path, res # List path and the SearchResult object with path_len, explored_graph, ppr_results, etc.
                    return (res.path, res) if (res and hasattr(res, 'path')) else (None, None)
            except Exception:
                return None, None

        return None, None

    def update_reputation(self, src, dst, outcome):
        """
        Updates the reputation from src to dst based on the outcome of a transaction.
        Uses an arithmetic mean and updates the graph G if trust is maintained.
        """

        old_val = self.reputation[src].get(dst, None) # If exist use the dst value for src, otherwise None
        count = self.trade_counts[src][dst]
        
        # Case 1: First direct transaction, create arc and update inverse index
        if old_val is None:
            new_val = float(outcome)
            self.trade_counts[src][dst] = 1
            self.in_edges[dst].add(src) 
        # Case 2: Enter in the network for the first time and create the initial connection with default value (no transaction yet)
        elif outcome == -1:
            new_val = self.default_rep 
        # Case 3: Restoration of old reputation without change (e.g., node rejoining)
        elif outcome == -2:
            new_val = old_val
        # Case 4: Subsequent transactions, update the mean reputation value if there was at least one previous transaction
        else: 
            if count == 0: # If there was a initial connection but not a single transaction yet 
                new_val = float(outcome)  
                self.trade_counts[src][dst] = 1
            else:          # If there was at least one previous transaction, update the mean reputation value
                new_count = count + 1  
                new_val = ((old_val * count) + outcome) / new_count
                self.trade_counts[src][dst] = new_count 
            
        self.reputation[src][dst] = new_val
        
        # We update only if it's the dst is trusted and both node are active, otherwise we remove the edge from G and check if the dst is isolated
        if new_val >= self.trust_threshold and src in self.active_nodes and dst in self.active_nodes:
            if self.algorithm in ["dijkstra"]: # Dijkstra use a minimization of the cost, if you have other minimization algorithms you have to add them here 
               self.G.add_edge(src, dst, weight=(1.0-new_val), capacity=new_val)
            else: 
                self.G.add_edge(src, dst, weight=new_val, capacity=new_val)
        else:
            if self.G.has_edge(src, dst):
                self.G.remove_edge(src, dst)
                self._check_isolation(dst) # Check isolation whenever an edge is removed

        if outcome >= 0: # Only log actual transaction outcomes, not initial connections (outcome=-1) or restorations (outcome=-2)        
           self.log_info(f"Reputation update: From Node {src} to Node {dst}, valutation: {outcome:.3f}, new mean: {new_val:.3f}.")

    def punish_node(self, malicious_node, outcome):
        """
        Broadcast the news of a bad interaction to all witnesses, if the trustworthiness protocoll for the single node is satisfied  
        they punish the node and if the threshold is crossed removes edges from G.
        """

        witnesses = list(self.in_edges[malicious_node])           # Get all nodes that have an edge to the malicious node 
        
        # Apply punishment in base of the valutation, if the valutation is very bad we apply a stronger punishment, otherwise a lighter one.
        for witness in witnesses:
            if witness in self.active_nodes:
                if not self.countermeasures_targeted_attacks(witness, self.current_sender): # Trustworthiness protocoll 
                    continue # Skip punishment for this witness if the sender is not trusted by the witness (countermeasure against targeted attacks)
                
                old_val = self.reputation[witness][malicious_node]
                
                # Apply punishment based on the outcome and the algorithm type.
                penalty_factor = 0.5 if outcome < 0.1 else 0.3
                new_val = old_val * (1.0 - penalty_factor)
                
                self.trade_counts[witness][malicious_node] += 1    # Edge case: if the witness never had a direct transaction with the malicious node (e.g new_val = ((0.2 * 0) + outcome) / 1). (ex initial connection with default value but no transaction yet)             
                self.reputation[witness][malicious_node] = new_val # Update the reputation value for the witness towards the malicious node (Update rep in the reputation dictionary)
                
                # If malicious behavior, remove the edge from G else update the edge weight and capacity in G
                if new_val < self.trust_threshold:                                   
                    if self.G.has_edge(witness, malicious_node):
                        self.G.remove_edge(witness, malicious_node)
                        self._check_isolation(malicious_node)      
                else: 
                    if self.G.has_edge(witness, malicious_node):
                        if self.algorithm in ["dijkstra"]: # If you have other minimization algorithms you have to add them here
                            self.G[witness][malicious_node]['weight'] = (1.0 - new_val)
                        else:
                            self.G[witness][malicious_node]['weight'] = new_val
                        self.G[witness][malicious_node]['capacity'] = new_val

                self.log_info(f"Punishment: Node {witness} updated {malicious_node}. New mean: {new_val:.3f}")
    
    def punish_path(self, path): 
        """
        Applies a distance-based punishment to recommenders in the path if the trustworthiness protocoll for the single node is satisfied. 
        Punishment occurs ONLY if the victim has a pre-existing connection with the recommender.
        The penalty decreases as the distance from the victim increases and if the threshold is crossed removes edges from G.
        """

        # No punish needed for direct trades (Victim -> Malicious)
        if not path or len(path) < 3:
            return

        victim = path[0]
        # Intermediate nodes are between the victim (0) and the malicious node (-1)
        for i in range(1, len(path) - 1):
            recommender = path[i]
            
            # Check if the victim has a pre-existing connection with this recommender
            if recommender in self.reputation[victim]:
                # Distance-based decay: The further the node, the lighter the punishment
                # i=1 (direct neighbor) -> 100% of base penalty
                # i=2 (second hop) -> 50% of base penalty
                # i=n (nth hop) -> 1/n of base penalty
                distance_decay = 1.0 / i
                base_penalty = 0.1 # If we increase this value we will punish more the recommender, if we decrease it we will punish less the recommender
                effective_penalty = base_penalty * distance_decay

                old_val = self.reputation[victim][recommender]
                
                # Actually only dijkstra return a path that can be used to punish the recommender 
                if self.algorithm in ["dijkstra"]:  # add here other algorithms that return a path if needed
                    new_val = old_val * (1.0 - effective_penalty)
                else: 
                    continue
                # We can not punish the recommender in maxflow and ppr because we do not have a path to do a reference

                # Update internal reputation records
                self.reputation[victim][recommender] = new_val
                self.trade_counts[victim][recommender] += 1
                
                # Update or remove the edge in the persistent graph G
                if new_val < self.trust_threshold:
                    if self.G.has_edge(victim, recommender):
                        self.G.remove_edge(victim, recommender)
                        self._check_isolation(recommender)
                else:
                    if self.G.has_edge(victim, recommender):
                        if self.algorithm in ["dijkstra"]: # If you have other minimization algorithms you have to add them here
                            self.G[victim][recommender]['weight'] = (1.0 - new_val)
                        else:   
                            self.G[victim][recommender]['weight'] = new_val
                        self.G[victim][recommender]['capacity'] = new_val

                self.log_info(f"Path Punishment: Victim {victim} penalized recommender {recommender} (distance {i}). New reputation: {new_val:.3f}")

    def punish_recommender(self, recommender_node):
        """
        Punishes the node that suggested a malicious path if the trustworthiness protocoll for the single node is satisfied, broadcasting the news to all witnesses. 
        It can be applied only for metrics that return a path, in our case dijkstra.
        Updates reputation and manages G edges based on the threshold.
        """

        # Get all witnesses that have an edge to the recommender node and apply a punishment
        witnesses = list(self.in_edges[recommender_node])         
        for witness in witnesses:
            if witness in self.active_nodes:

                if not self.countermeasures_targeted_attacks(witness, self.current_sender):
                    continue # Skip punishment for this witness if the sender is not trusted by the witness (countermeasure against targeted attacks)
                
                old_val = self.reputation[witness][recommender_node]
                
                if self.algorithm in ["dijkstra"]:
                    new_val = old_val * (1.0 - 0.05)  # Really light punishment right now, we can increase it if we want to punish more the recommender, if we want to increment modify the 0.05 value.
                else: 
                    continue
                #We can not punish the recommender in ppr or maxflowbecause we do not have a path to do a reference but only scores, 
                #we can only punish the malicious node directly.
            
                self.trade_counts[witness][recommender_node] += 1
                self.reputation[witness][recommender_node] = new_val
                
               # If the new reputation value indicates malicious behavior, remove the edge from G
                if new_val < self.trust_threshold:
                    if self.G.has_edge(witness, recommender_node):
                        self.G.remove_edge(witness, recommender_node)
                        self._check_isolation(recommender_node)
                else:
                    if self.G.has_edge(witness, recommender_node):
                        if self.algorithm in ["dijkstra"]: # If you have other minimization algorithms you have to add them here
                            self.G[witness][recommender_node]['weight'] = (1.0 - new_val)
                        else:
                            self.G[witness][recommender_node]['weight'] = new_val
                        self.G[witness][recommender_node]['capacity'] = new_val
                
                self.log_info(f"Punishment: Recommender {recommender_node} penalized. New mean: {new_val:.3f}")


    def countermeasures_targeted_attacks(self, receiver, sender):
        """
        Evaluates the trustworthiness of the 'sender' who broadcasted an alert from the perspective of the 'receiver'.
        Returns True if the sender is trusted, False otherwise.
        """

        # If the receiver is the same as the sender, we consider it trusted by default (self-trust).
        if receiver == sender:
            return True

        # 1. Direct connection check, we do not have to ask to the neighbors if the sender is trusted or not. 
        if sender in self.reputation[receiver]:
            return self.reputation[receiver][sender] >= self.trust_threshold
            

        # 2. Peer polling if no direct connection exists, gather opinions from active neighbors who have a direct connection to the sender.
        opinions = []
        for neighbor in list(self.reputation[receiver].keys()):
            if neighbor in self.active_nodes:
                neighbor_rep = self.reputation[receiver][neighbor]
                
                # Check if the neighbor itself is trusted
                neighbor_trusted = neighbor_rep >= self.trust_threshold
                
                # If the neighbor is trusted and knows the sender, gather its opinion
                if neighbor_trusted and sender in self.reputation[neighbor]:
                    sender_rep = self.reputation[neighbor][sender]
                    
                    # Check if the neighbor's opinion about the sender is trusted
                    sender_trusted = sender_rep >= self.trust_threshold

                    # Append the neighbor's opinion about the sender to the opinions list
                    opinions.append(sender_trusted)

        # 3. Consensus rule among active neighbors based on configured threshold the default is 2/3 of the opinions must be positive to consider the sender trusted.
        if opinions:
            approval_rate = sum(1 for vote in opinions if vote) / len(opinions)
            return approval_rate >= self.consensus_threshold

        # No information available means untrusted source
        return False

    def slander_attack(self):
       """
       Simulates a slander attack where malicious nodes attempt to decrease the reputation of honest nodes.
       """

       return 0.0 #every time is the lowest possible value. 


    def collusion_attack(self, src):
        """
        Increase the reputation of a malicious node by having its accomplices (malicious friends) boost its reputation.
        The boosting is applied to a random subset of malicious friends of the source node
        """
         
        # Filter all currently active malicious nodes, excluding the source itself (We can not boost the reputation of the source node itself (same incoming arc))
        all_malicious = [i for i in self.active_nodes_list if self.node_behaviors[i] == "malicious" and i != src]
        
        # Select from 0 to 3 random malicious friends to apply the boosting outcome (more friends are and more trustworthy 
        #the malicious node will be, but it's a inrealistic attack because technically the network should be able to detect a massive group of malicious nodes boosting)
        sample_size = min(random.randint(0, 3), len(all_malicious))
        malicious_friends = random.sample(all_malicious, sample_size)

        # Apply the boosting outcome to the selected subset of accomplices
        for malicious_friend in malicious_friends:
            self.update_reputation(src, malicious_friend, 1.0)  # Boosting outcome 
        