import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd

np.random.seed(42)

distributions = {
    "Normal": lambda size: np.random.normal(0, 1, size),
    "Cauchy": lambda size: np.random.standard_cauchy(size),
    "Poisson": lambda size: np.random.poisson(10, size),
    "Uniform": lambda size: np.random.uniform(-math.sqrt(3), math.sqrt(3), size)
}

sample_sizes = [20, 100, 1000]
outlier_stats = []

for name, dist_func in distributions.items():
    all_data = []
    labels = []

    for size in sample_sizes:
        sample = dist_func(size)

        # Подсчёт выбросов
        q1 = np.percentile(sample, 25)
        q3 = np.percentile(sample, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((sample < lower) | (sample > upper)).sum()
        outlier_stats.append({
            "Распределение": name,
            "Размер выборки": size,
            "Число выбросов": outliers,
            "Относительная доля": round(outliers / size, 4)
        })

        if name == "Cauchy":
            # Каждый бокс-плот на своей картинке
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=sample, color='skyblue', fliersize=4)
            plt.title(f"{name} distribution, n = {size}")
            plt.xlabel("Значения")
            plt.tight_layout()
            plt.savefig(f"{name.lower()}_{size}_boxplot.png")
            plt.close()
        else:
            all_data.append(sample)
            labels.append(f"n = {size}")

    # Все выборки на одной картинке (кроме Cauchy)
    if name != "Cauchy":
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=all_data, orient="h")
        plt.yticks(ticks=[0, 1, 2], labels=labels)
        plt.xlabel("Значения")
        plt.title(f"{name} distribution (бокс-плоты Тьюки)")
        plt.tight_layout()
        plt.savefig(f"{name.lower()}_boxplots_all_sizes.png")
        plt.close()


# Таблица с выбросами
df = pd.DataFrame(outlier_stats)
print(df.to_string(index=False))
