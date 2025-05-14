import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, cauchy, poisson, uniform
import math

np.random.seed(30)

# --- ПАРАМЕТРЫ РАСПРЕДЕЛЕНИЙ ---
distributions_info = [
    (np.random.standard_normal, norm(0, 1), "normal"),
    (np.random.standard_cauchy, cauchy(0, 1), "cauchy"),
    (lambda size: np.random.poisson(10, size), poisson(10), "poisson"),
    (lambda size: np.random.uniform(-math.sqrt(3), math.sqrt(3), size), uniform(-math.sqrt(3), 2*math.sqrt(3)), "uniform")
]

# --- ЧАСТЬ 1: ГРАФИКИ ---
# Информация о распределениях
distributions_info = [
    (np.random.standard_normal, norm(0, 1), "normal"),
    (np.random.standard_cauchy, cauchy(0, 1), "cauchy"),
    (lambda size: np.random.poisson(10, size), poisson(10), "poisson"),
    (lambda size: np.random.uniform(-math.sqrt(3), math.sqrt(3), size), uniform(-math.sqrt(3), 2*math.sqrt(3)), "uniform")
]

sample_sizes_part1 = [10, 50, 1000]

for gen_func, dist, name in distributions_info:
    for size in sample_sizes_part1:
        sample = gen_func(size)
        plt.figure(figsize=(8, 6))

        # === РАСПРЕДЕЛЕНИЕ КОШИ ===
        if name == "cauchy":
            # Более толстые столбцы за счёт ручной настройки binwidth
            sns.histplot(sample, stat='density', binwidth=2, label='Histogram', color='skyblue')
            x = np.linspace(min(sample), max(sample), 1000)
            plt.plot(x, dist.pdf(x), 'r-', label='Theoretical PDF')

        # === РАСПРЕДЕЛЕНИЕ ПУАССОНА ===
        elif name == "poisson":
            # Строим вручную bar-гистограмму и PMF
            values, counts = np.unique(sample, return_counts=True)
            probs = counts / size
            plt.bar(values, probs, width=0.8, color='skyblue', label='Empirical PMF')
            x = np.arange(min(values), max(values)+1)
            plt.plot(x, dist.pmf(x), 'ro-', label='Theoretical PMF')

        # === ДЛЯ НОРМАЛЬНОГО И РАВНОМЕРНОГО ===
        else:
            sns.histplot(sample, stat='density', bins='auto', label='Histogram', color='skyblue')
            x = np.linspace(min(sample), max(sample), 1000)
            plt.plot(x, dist.pdf(x), 'r-', label='Theoretical PDF')

        plt.title(f"{name.capitalize()} distribution, size={size}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{name}_size_{size}.png")
        plt.close()

# --- ЧАСТЬ 2: СТАТИСТИКИ ---
def compute_statistics(dist_func):
    sample_sizes = [10, 100, 1000]
    stats = {}

    for size in sample_sizes:
        means, medians, zqs = [], [], []

        for _ in range(1000):
            sample = dist_func(size)
            sample_sorted = np.sort(sample)
            q1 = np.percentile(sample_sorted, 25)
            q3 = np.percentile(sample_sorted, 75)
            zq = (q1 + q3) / 2

            means.append(np.mean(sample))
            medians.append(np.median(sample))
            zqs.append(zq)

        for label, values in zip(['mean', 'median', 'zq'], [means, medians, zqs]):
            values = np.array(values)
            ez = np.mean(values)
            ez2 = np.mean(values**2)
            dz = ez2 - ez**2
            stats[(size, label)] = (round(ez, 4), round(dz, 4))

    return stats

# --- ВЫВОД РЕЗУЛЬТАТОВ В КОНСОЛЬ ---
for gen_func, _, name in distributions_info:
    stats = compute_statistics(gen_func)
    print(f"\n{name.upper()} DISTRIBUTION STATISTICS:")
    print("Sample size | Statistic | E(z)   | D(z)")
    print("-"*40)
    for size in [10, 100, 1000]:
        for label in ['mean', 'median', 'zq']:
            ez, dz = stats[(size, label)]
            print(f"{size:<12} {label:<9} {ez:<7} {dz:<7}")
