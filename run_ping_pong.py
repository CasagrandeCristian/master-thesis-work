import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_simulation(emb_type, metric, attack, malicious, punishment, pool_trade):
    command = [
        "python", "main.py",
        "--algo", "ping_pong_dfs",
        "--embedded_type", str(emb_type),
        "--ping_pong_choice", str(metric),
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
    embedded_types = ["stge", "socs"]
    ping_pong_metrics = ["ppr", "max_flow"]
    attack_types = ["none", "slander", "collusion", "slander_collusion"]
    malicious_rates = [0.2, 0.3, 0.4, 0.51]
    punishments = ["single", "none"]
    pool_trades = [1, 5]

    tasks = []
    for emb in embedded_types:
        for metric in ping_pong_metrics:
            for attack in attack_types:
                for rate in malicious_rates:
                    for punish in punishments:
                        for pool in pool_trades:
                            tasks.append((emb, metric, attack, rate, punish, pool))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_simulation, *t) for t in tasks]
        for f in futures:
            f.result()

if __name__ == "__main__":
    main()