import random
import numpy as np
import pandas as pd
import joblib
from deap import base, creator, tools, algorithms
from scipy.spatial import distance

# 1. Load Model and Data
model = joblib.load("et_regressor_model.joblib")
df = pd.read_csv("./data/cleaned.csv")
X = df.iloc[:, :-1]

# 2. Define Genetic Algorithm Components
# Problem Definition: Multi-objective optimization (Maximize Quality, Minimize Distance)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # (Quality, Distance)
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Define search space for each feature (gene)
param_ranges = {
    col: (X[col].mean() - 2 * X[col].std(), X[col].mean() + 2 * X[col].std())
    for col in X.columns
}

from functools import partial

# Register functions to create individuals and populations
attribute_list = [
    partial(random.uniform, param_ranges[col][0], param_ranges[col][1])
    for col in X.columns
]
toolbox.register(
    "individual", tools.initCycle, creator.Individual, tuple(attribute_list), n=1
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation Function
mean_vec = np.mean(X, axis=0)
cov_matrix = np.cov(X.T)
inv_cov_matrix = np.linalg.inv(cov_matrix)


def evaluate(individual):
    # Flatten the individual list and reshape for prediction
    recipe = pd.DataFrame(
        np.array(individual).flatten().reshape(1, -1), columns=X.columns
    )

    # Objective 1: Predicted Quality
    predicted_quality = np.mean(model.predict(recipe))

    # Objective 2: Mahalanobis Distance
    mahal_dist = distance.mahalanobis(
        recipe.values.flatten(), mean_vec.values, inv_cov_matrix
    )

    return predicted_quality, mahal_dist


# Register GA operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selNSGA2)  # Use NSGA-II for multi-objective selection


# 3. Run Optimization
def run_optimization():
    NGEN = 100
    MU = 100  # Population size
    CXPB = 0.7  # Crossover probability
    MUTPB = 0.2  # Mutation probability

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()  # Use ParetoFront for multi-objective hall of fame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, logbook = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=MU,
        lambda_=MU,
        cxpb=CXPB,
        mutpb=MUTPB,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )
    return pop, logbook, hof


if __name__ == "__main__":
    pop, logbook, hof = run_optimization()

    # Save results
    best_recipes = pd.DataFrame([list(ind) for ind in hof], columns=X.columns)
    best_recipes_fitness = pd.DataFrame(
        [list(ind.fitness.values) for ind in hof], columns=["quality", "distance"]
    )

    best_recipes.to_csv("best_recipes.csv", index=False)
    best_recipes_fitness.to_csv("best_recipes_fitness.csv", index=False)

    print("\nAdvanced simulation finished.")
    print(f"Found {len(hof)} optimal recipes in the Pareto front.")
    print("Results saved to 'best_recipes.csv' and 'best_recipes_fitness.csv'.")
