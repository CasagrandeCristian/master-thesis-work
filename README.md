# Distributed Trust Network Simulation

A research-oriented simulation framework for studying trust-aware routing, reputation dynamics, and adversarial behavior in distributed network environments.

This project models a dynamic network of nodes that join, leave, and rejoin over time while exchanging transactions and maintaining local trust relationships. It evaluates how different pathfinding strategies perform when the network is subjected to malicious behavior such as slander, collusion, and coordinated reputation attacks.

## Project vision

The simulation is designed to explore the interaction between:

- trust and reputation management,
- graph-based routing and search strategies,
- embedding-based network layouts,
- malicious behavior and targeted attacks,
- network-level countermeasures and punishment policies.

The goal is to quantify how routing decisions change under adversarial conditions and which methods remain robust in the presence of deception or reputation manipulation.

## Key capabilities

- Dynamic network churn with node joins, leaves, and rejoining events.
- Transaction-level trust updates based on reputation scores.
- Multiple routing algorithms for selecting trade paths or recommendations.
- Attack simulation for slander, collusion, and combined attack patterns.
- Punishment models for malicious nodes and malicious recommendation chains.
- Embedding-based graph layouts using social-coordinate and tree-based network structures.
- Statistics, snapshots, and plot generation for experimental analysis.
- CSV export for comparison across attack scenarios and algorithm configurations.

## Supported routing and search strategies

The project includes a range of pathfinding and recommendation approaches, including:

- Dijkstra shortest path
- Max-flow routing
- Personalized PageRank (PPR)
- BeFS-style search variants
- Greedy DFS strategies
- Ping-pong DFS variants
- Layout-aware traversal over embedded graph representations

These algorithms are evaluated in both standard graph settings and embedding-assisted settings using the project’s SoCS and STGE layouts.

## Attack and defense model

The simulator supports several malicious scenarios:

- `none`: no target attack
- `slander`: misleading reputation propagation
- `collusion`: coordinated manipulation among malicious nodes
- `slander_collusion`: combined reputation attacks

Countermeasures include:

- `none`: no punishment
- `single`: direct punishment of malicious actors
- `path`: penalties applied to nodes that participate in a malicious path
- `recommender`: punishment of nodes propagating malicious recommendations

## Repository structure

- `main.py` — primary command-line entry point for a single simulation run
- `network_model.py` — core network simulation, trust logic, and routing evaluation
- `network_events.py` — event-driven node joins, leaves, transactions, and metrics snapshots
- `search_algorithms.py` — layout-aware and graph search implementations
- `search_result.py` — structured search outcome container
- `ppr.py` — personalized PageRank helper routines
- `customlayouts.py` — SoCS and STGE layout generation utilities
- `network_visualization.py` — plotting, summaries, and output charts
- `plot.py` — utility plotting helper for exported results
- `discrete_event_sim.py` — generic event-scheduling simulation engine
- `run_standard.py` — batch run for standard algorithms
- `run_stge_standard.py` — STGE-focused batch experiments
- `run_socs_standard.py` — SoCS-focused batch experiments
- `run_ping_pong.py` — ping-pong pathfinding experiments
- `experiments/` — generated experiment outputs, CSV files, and plots

## Getting started

### Requirements

This project requires Python and the packages listed in [`requirements.txt`](./requirements.txt).

To create the virtual environment and install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `venv\Scripts\activate` instead.

### Running a single simulation

Use the main entry point to run one configured experiment:

```bash
python main.py --algo dijkstra --attack_type none --malicious 0.2 --punishment none
```

You can customize the simulation with arguments such as:

- `--nodes`
- `--threshold`
- `--rep`
- `--malicious`
- `--punishment`
- `--poolTrade`
- `--algo`
- `--ping_pong_choice`
- `--time`
- `--snap_interval`
- `--start_connection`
- `--join_rate`
- `--leave_rate`
- `--trade_rate`
- `--embedded_type`
- `--attack_type`

### Running experiment batches

Several runner scripts are provided for broader comparative studies:

```bash
python run_standard.py
python run_stge_standard.py
python run_socs_standard.py
python run_ping_pong.py
```

These scripts generate repeated runs over multiple algorithm, attack, malicious-rate, and punishment combinations.

## Experiment outputs

Results and visualizations are written into the `experiments/` directory. This includes:

- CSV result files grouped by scenario and algorithm,
- attack-specific folders such as normal runs, slander, collusion, and punishment variants,
- generated plots for comparing performance and attack resistance.

The repository is organized to support comparative analysis between healthy network behavior and adversarial network conditions.

## Typical research questions the project addresses

- Which routing strategies are most resilient to malicious reputation manipulation?
- How effective are local punishment schemes versus path-aware or recommender-aware countermeasures?
- How does network churn influence the stability of trust relations over time?

## Suggested workflow

1. Define the network parameters and attack configuration.
2. Run a baseline simulation without attacks.
3. Compare behavior under slander and collusion scenarios.
4. Evaluate differences across routing algorithms and punishment policies.
5. Use the generated plots and CSV summaries to interpret robustness and trade-offs.

## Notes

This repository is structured as a simulation and analysis toolkit rather than a general-purpose production library. It is best suited for experimentation, research, and comparative evaluation of trust-aware distributed routing under adversarial conditions.

## Summary

This project provides a flexible and extensible framework for investigating trust, reputation, and routing performance in dynamic distributed networks under adversarial conditions. It combines discrete-event simulation, graph search, embedding-based routing, and experimental plotting into a single reproducible research environment.
