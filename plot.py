import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.special import gamma

# ==========================================
# Stile Grafico Accademico Standard (IEEE / Nature)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'axes.edgecolor': '#444444',
    'axes.linewidth': 0.9,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'xtick.color': '#333333',
    'ytick.color': '#333333'
})

# Palette standard accademica Okabe-Ito (Blu, Vermiglio, Verde)
standard_colors = ['#0072B2', '#D55E00', '#009E73'] 

# ==========================================
# 1. Distribuzione di Poisson (PMF)
# ==========================================
plt.figure(figsize=(7, 4.2))

x_poisson = np.linspace(0, 20, 500)
lambdas_poisson = [1, 4, 10]

for lam, color in zip(lambdas_poisson, standard_colors):
    y = (lam**x_poisson * np.exp(-lam)) / gamma(x_poisson + 1)
    plt.plot(x_poisson, y, color=color, label=f'$\\lambda = {lam}$', linewidth=2)

plt.title('Poisson Distribution (PMF)', fontsize=11, fontweight='bold', pad=10)
plt.xlabel('Number of Events ($k$)', fontsize=10)
plt.ylabel('Probability $P(X = k)$', fontsize=10)
plt.xlim(0, 20)
plt.ylim(bottom=0)

plt.legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('poisson_distribution.png', dpi=300)
plt.close()


# ==========================================
# 2. Distribuzione Esponenziale (PDF)
# ==========================================
plt.figure(figsize=(7, 4.2))

x_exp = np.linspace(0, 5, 500)
lambdas_exp = [0.5, 1.0, 1.5]  # Valori standard con passo 0.5

for lam, color in zip(lambdas_exp, standard_colors):
    pdf = expon.pdf(x_exp, scale=1/lam)
    plt.plot(x_exp, pdf, color=color, label=f'$\\lambda = {lam}$', linewidth=2)

plt.title('Exponential Distribution (PDF)', fontsize=11, fontweight='bold', pad=10)
plt.xlabel('Inter-arrival Time ($x$)', fontsize=10)
plt.ylabel('Probability Density $f(x)$', fontsize=10)
plt.xlim(0, 5)
plt.ylim(bottom=0)

plt.legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('exponential_distribution.png', dpi=300)
plt.close()

print("Grafici aggiornati generati con successo:")
print(" - poisson_distribution.png (lambda = 1, 4, 10)")
print(" - exponential_distribution.png (lambda = 0.5, 1.0, 1.5)")