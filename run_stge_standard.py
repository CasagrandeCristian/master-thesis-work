import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_simulation(algo, attack, malicious, punishment, pool_trade):
    command = [
        "python", "main.py",
        "--algo", str(algo),
        "--embedded_type", "stge",
        "--attack_type", str(attack),
        "--malicious", str(malicious),
        "--punishment", str(punishment),
        "--poolTrade", str(pool_trade),
        "--nodes", "100",
        "--time", "500",
        "--leave_rate", "1e-9",
        "--start_connection", "5"
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Simulation error: {e}")

def main():
    algorithms = ["befs","greedy_dfs"]
    attack_types = ["none", "slander", "collusion", "slander_collusion"]
    malicious_rates = [0.2, 0.3, 0.4, 0.51]
    punishments = ["single", "none"]
    pool_trades = [1, 5]

    tasks = []
    for algo in algorithms:
        for attack in attack_types:
            for rate in malicious_rates:
                for punish in punishments:
                    for pool in pool_trades:
                        tasks.append((algo, attack, rate, punish, pool))


    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_simulation, *t) for t in tasks]
        for f in futures:
            f.result()

if __name__ == "__main__":
    main()