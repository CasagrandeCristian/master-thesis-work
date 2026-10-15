import random
from discrete_event_sim import Event
from customlayouts import socs_layout, rooted_spanning_tree

class EmbeddingUpdate(Event):
    """
    Periodic event to update node embeddings using the selected distributed layout algorithm.
    Schedules itself at regular user-defined intervals.
    """

    def process(self, sim: "DistributedNetwork"):
        # Abort if embedding mode is disabled
        if not sim.embedded_type or sim.embedded_type == "none":
            return

        # Compute embeddings only if the graph contains nodes
        if sim.G.number_of_nodes() > 0:
            
            if sim.embedded_type == "socs":
            # Configured with 'socs_layout_cyclon_diff_force' parameters (from thesis)
                sim.layout = socs_layout(
                    graph=sim.G, 
                    dim=sim.emb_dimensions,
                    with_cyclon=True,
                    max_iter=1000,
                    epsilon_based_on_dim=True,
                    FA=2,
                    FR=-1
                )
                
                
            elif sim.embedded_type == "stge":
                stge_layouts = rooted_spanning_tree(
                    graph=sim.G, 
                    dim=sim.emb_dimensions,
                    LAMBDA=5
                )
                # Store all generated spanning trees for potential multi-tree routing
                sim.stge_layouts = stge_layouts 
                # Use the first tree as the primary layout (Could be randomized or rotated for load balancing in future iterations, but for now use always the first)
                sim.layout = stge_layouts[0]

            sim.log_info(f"Embedding Update: Recalculated positions for {len(sim.layout)} nodes at t={sim.t:.2f} using {sim.embedded_type}.")
        else:
            sim.log_info("Embedding Update: Skipped, graph is currently empty.")

        # Schedule the next periodic embedding update
        sim.schedule(sim.embedding_interval, EmbeddingUpdate())

class StatSnapshot(Event):
    """
    Periodic event to capture statistics for plotting.
    Schedules itself at regular intervals to record:
        1. Cumulative honest and malicious transactions.
        2. Pathfinding success rate.
        3. Average scams per malicious node before isolation.
    """

    def process(self, sim: "DistributedNetwork"):

        # Capture current stats for plotting:
        sim.history['t'].append(sim.t)
        sim.history['trusted'].append(sim.trusted_trades)
        sim.history['untrusted'].append(sim.untrusted_trades)
        
        # Pathfinding Success Rate history
        rate = (sim.total_attempts - sim.failed_pathfinding) / sim.total_attempts if sim.total_attempts > 0 else 1.0
        sim.history['success_rate'].append(rate)
        
        # Average scams per malicious node before isolation
        avg_scams = sum(sim.completed_scams_records) / len(sim.completed_scams_records) if sim.completed_scams_records else 0
        sim.history['avg_scams_per_node'].append(avg_scams)

        sim.schedule(sim.snap_interval, StatSnapshot())

class NodeJoin(Event):
    """
    Handles a new node entering the network via _add_node.
    Schedules the next Join event.
    """

    def process(self, sim: "DistributedNetwork"):

        node_id = sim._add_node()
        sim.log_info(f"Join: Node {node_id} ({sim.node_behaviors[node_id]})")
        sim.schedule(random.expovariate(sim.join_rate), NodeJoin())

class NodeLeave(Event):
    """
    Represents a node leaving the network. Removes it from active structures and G.
    Schedules a rejoin event after a random delay.
    """

    def __init__(self, node_id): 
        self.node_id = node_id

    def process(self, sim: "DistributedNetwork"):

        if self.node_id not in sim.active_nodes: 
            return

        sim._remove_node_from_active(self.node_id)
        sim.log_info(f"Leave: Node {self.node_id}")
        sim.schedule(random.expovariate(sim.join_rate), NodeRejoin(self.node_id))

class NodeRejoin(Event):
    """
    Handles a node rejoining. Restores edges in G by looking up historical reputation.
    Schedules the next Transaction and Leave events for the rejoining node.
    """

    def __init__(self, node_id): 
        self.node_id = node_id

    def process(self, sim: "DistributedNetwork"):

        if self.node_id in sim.active_nodes: 
            return

        n_id = self.node_id
        sim._add_node_to_active(n_id)
        
        # Restore active edges from historical memory for the graph G.

        # 1. Outgoing edges (neighbors the rejoining node knows)
        for target, val in sim.reputation[n_id].items():
            if target in sim.active_nodes:
                sim.update_reputation(n_id, target, -2) # Use a special outcome to indicate restoration of old reputation without change
        
        # 2. Incoming edges (active nodes that know the rejoining node) using in_edges index
        for src in sim.in_edges[n_id]:
            if src in sim.active_nodes:
                sim.update_reputation(src, n_id, -2)    

        sim.log_info(f"Rejoin: Node {n_id}")
        sim.schedule(random.expovariate(sim.trade_rate), Transaction(n_id))
        sim.schedule(random.expovariate(sim.leave_rate), NodeLeave(n_id))

class Transaction(Event):
    """
    Represents a trade attempt between a source and a destination, updates reputation and punishments if a trusted path is found.
    Schedules the next Transaction for the source node.
    """

    def __init__(self, source): 
        self.source = source

    def process(self, sim: "DistributedNetwork"):
        src = self.source
        if src not in sim.active_nodes: 
            return

        if len(sim.active_nodes_list) < 2:  # Need at least 2 active nodes to attempt a trade, if not, reschedule the transaction for later
            sim.schedule(random.expovariate(sim.trade_rate), Transaction(src))
            return

        sim.total_attempts += 1 # Increment total trade attempts for pathfinding success rate calculation

        # Select potential candidates for trade, excluding the source node
        potential_targets = [n for n in sim.active_nodes_list if n != src]     
        candidates = random.sample(potential_targets, min(len(potential_targets), sim.poolTrade)) 

        best_dst = None
        best_path = None
        best_metric = 0.0

        # Define algorithms that prefer lower metric values (like Dijkstra) for pathfinding.
        low_metric_algorithms = ["dijkstra", "befs", "befs_nonrs", "greedy_dfs", "greedy_dfs_sps"]
        
        # Case 1. No Embedding (Global Graph Algorithms)
    
        if sim.embedded_type == "none" or not sim.embedded_type:
            if sim.algorithm == "dijkstra":
                best_metric = float('inf') # Initialize to infinity for minimization
                for candidate in candidates:
                    path, metric = sim.find_path_metric(src, candidate)
                    if path and metric < best_metric:                                                           
                        best_metric, best_dst, best_path = metric, candidate, path
                        
            elif sim.algorithm == "max_flow":
                best_metric = -1.0 # Initialize to -1 for maximization
                for candidate in candidates:
                    _, flow_value = sim.find_path_metric(src, candidate)  
                    if flow_value > 0 and flow_value > best_metric:
                        best_metric, best_dst = flow_value, candidate
                        
            elif sim.algorithm == "ppr":
                best_metric = -1.0
                _, ppr_scores = sim.find_path_metric(src, None)
                if ppr_scores is not None:
                    for candidate in candidates:
                        if candidate in ppr_scores and ppr_scores[candidate] > best_metric:
                            best_metric, best_dst = ppr_scores[candidate], candidate

        
        # Case 2. Distributed Embedding Algorithms (SoCS / STGE)
        
        else:
            # The only algorithm embedding that really care about weights is ping-pong because it uses PPR and Max Flow Metrics, so we have to follow the maximazation rule. 
            # For the other algorithms we still need of weights but just for the thresholding, so we can use both, in my case i prefer use the minimization. 
            is_minimization = sim.algorithm in low_metric_algorithms
            
            best_metric = float('inf') if is_minimization else -1.0

            for candidate in candidates:
                path, res = sim.find_path_metric(src, candidate)
                
                # Every embedding algorithm should return a path and a SearchResult object with the relevant metric (path_len, ppr_score, flow_value, etc.)
                if path is not None: 
                    if is_minimization:
                        metric_value = float(res.path_len) if hasattr(res, 'path_len') else float(len(path))
                        if metric_value < best_metric:
                            best_metric = metric_value
                            best_dst = candidate
                            best_path = path
                            
                    # PPR or Max Flow via Ping-Pong DFS in base the ping_pong_metric
                    else:
                        if hasattr(res, 'ppr_score'):  
                            metric_value = float(res.ppr_score)
                        elif hasattr(res, 'flow_value'):
                            metric_value = float(res.flow_value)
                        
                        if metric_value > best_metric:
                            best_metric = metric_value
                            best_dst = candidate
                            best_path = path 

        
        # Transaction process 
        
        if best_dst is not None:
            outcomes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # Values for the outcome of the trade.  
            is_malicious_src = (sim.node_behaviors[src] == "malicious")
            is_malicious_dst = (sim.node_behaviors[best_dst] == "malicious")
            trades_done = sim.total_node_received_trades[best_dst]        # For switch into the malicious behavior        

            # Update trade counters for source and destination nodes based on their behavior
            if is_malicious_src:
                sim.malicious_src_trades += 1
            else:
                sim.honest_src_trades += 1

            if is_malicious_dst:
                sim.malicious_dst_trades += 1
            else:
                sim.honest_dst_trades += 1

            # Transaction behavior 
            if is_malicious_dst and trades_done >= 5:                # If the destination is malicious and has done already 5 trades switch its behavior.
                weights = [0.15, 0.15, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.02, 0.02, 0.01]
            else:                                                    # Honest or malicious in this case doesn't matter, the behavior is the same they play good trades.
                weights = [0.001, 0.002, 0.003, 0.004, 0.005, 0.005, 0.01, 0.04, 0.08, 0.15, 0.70] 
                  
            outcome = random.choices(outcomes, weights=weights)[0]

            # Handle targeted slander attacks if applicable
            if is_malicious_src and not is_malicious_dst and (sim.attack_type == "slander" or sim.attack_type == "slander_collusion"):
                outcome = sim.slander_attack()
            # Handle targeted collusion attacks if applicable
            if is_malicious_src and (sim.attack_type == "collusion" or sim.attack_type == "slander_collusion"):
                sim.collusion_attack(src)
                if is_malicious_dst:
                    outcome = 1.0 # We want to make happy our malicious friend 


            # Update simulation statistics and apply punishments
            if outcome <= sim.trust_threshold:                  
                sim.untrusted_trades += 1
                if is_malicious_dst:
                    sim.malicious_scams_count[best_dst] += 1
                if is_malicious_src:
                    sim.malicious_bad_trades += 1
                else: 
                    sim.honest_bad_trades += 1
                    
                if sim.punishment:
                    sim.current_sender = src # Set current sender for broadcast 
                    if sim.punishment == "single":
                        sim.punish_node(best_dst, outcome)
                    elif sim.punishment == "path" and sim.algorithm == "dijkstra":  
                        sim.punish_path(best_path)
                    elif sim.punishment == "recommender" and sim.algorithm == "dijkstra":  
                        recommender = best_path[-2]
                        sim.punish_recommender(recommender)
                    sim.current_sender = None # Reset after punishment
            else: # We want to see how many good trades a malicious node can receive to compare it with the honest nodes.
                if is_malicious_src:
                    sim.malicious_good_trades += 1 
                else:                    
                    sim.honest_good_trades += 1
                sim.trusted_trades += 1

            sim.total_node_received_trades[best_dst] += 1
            # Update mean trades for malicious and honest nodes
            if is_malicious_src:
                sim.mean_trades_malicious[1] += 1
                sim.mean_trades_malicious[0] += (outcome - sim.mean_trades_malicious[0]) / sim.mean_trades_malicious[1]
            else:
                sim.mean_trades_honest[1] += 1
                sim.mean_trades_honest[0] += (outcome - sim.mean_trades_honest[0]) / sim.mean_trades_honest[1]

            sim.log_info(f"Trade: between Node {src} and Node {best_dst} Path: {best_path} Metric: {best_metric:.3f}.")
            sim.update_reputation(src, best_dst, outcome)
        else:
            sim.failed_pathfinding += 1
            sim.log_info(f"No trusted path found for any of the {len(candidates)} candidates from Node {src}")
            
        sim.schedule(random.expovariate(sim.trade_rate), Transaction(src))