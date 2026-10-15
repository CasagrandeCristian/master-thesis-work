import logging
import argparse
import random
from network_model import DistributedNetwork
from network_visualization import plot_combined_summary, plot_individual_metrics, save_summary_csv, plot_malicious_rate_analysis, plot_good_transactions_breakdown, plot_good_trades_share_distribution, plot_bad_trades_share_distribution, plot_combined_negative_trends, plot_results_attack, plot_results_attack_only_2
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Distributed Network Trust Simulation")
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--rep", type=float, default=0.5)
    parser.add_argument("--malicious", type=float, default=0.2)
    parser.add_argument("--punishment", type=str, default="none", choices=["single", "path", "recommender", "none"])
    parser.add_argument("--poolTrade", type=int, default=5)
    parser.add_argument("--algo", type=str, default="dijkstra", choices=["dijkstra", "max_flow", "ppr", "befs", "befs_nonrs", "greedy_dfs", "greedy_dfs_sps", "ping_pong_dfs"])
    parser.add_argument("--ping_pong_choice", type=str, default="ppr", choices=["ppr", "max_flow"])
    parser.add_argument("--time", type=int, default=500)
    parser.add_argument("--snap_interval", type=float, default=5.0)
    parser.add_argument("--start_connection", type= int, default=5)
    parser.add_argument("--join_rate", type=float, default=0.1)
    parser.add_argument("--leave_rate", type=float, default=0.05)
    parser.add_argument("--trade_rate", type=float, default=1.0)
    parser.add_argument("--embedded_type", type=str, default="none", choices=["socs", "stge", "none"])
    parser.add_argument("--emb_dimensions", type=int, default=100)
    parser.add_argument("--consensus_threshold", type=float, default=0.66)
    parser.add_argument("--embedding_interval", type=float, default=5.0)
    parser.add_argument("--attack_type", type=str, default="none", choices=["slander","collusion","slander_collusion" ,"none"])
    args = parser.parse_args()

    logging.basicConfig(format='{message}', level=logging.INFO, style='{')
    random.seed(43)

    sim = DistributedNetwork(
        n_nodes=args.nodes, 
        join_rate=args.join_rate, 
        leave_rate=args.leave_rate, 
        trade_rate=args.trade_rate,
        trust_threshold=args.threshold, 
        default_rep=args.rep,
        malicious_rate=args.malicious, 
        poolTrade=args.poolTrade,
        punishment=args.punishment,
        snap_interval=args.snap_interval,
        start_connection=args.start_connection,
        algorithm=args.algo,
        ping_pong_choice=args.ping_pong_choice,
        embedded_type=args.embedded_type,
        emb_dimensions=args.emb_dimensions,
        consensus_threshold=args.consensus_threshold,
        embedding_interval=args.embedding_interval,
        attack_type=args.attack_type

    )

    # Run simulation
    #sim.run(max_t=args.time)
    # Generate final analysis plots
    #plot_combined_summary(sim)
    #plot_individual_metrics(sim)
    #save_summary_csv(sim) 
    #plot_results_attack(folder_path="experiments/attacksWithDef/path/collusion", output_dir="experiments/attacksWithDef/path/collusion/plots")

if __name__ == "__main__":
    main()
