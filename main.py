from problem import Problem
from sa import SA
from ga import GA
from ts import TS

# # configurations
# problem_kwargs = {
#     'num_workers': 500,
#     'period': 28,
#     'num_shifts': 4,
#     'num_beds': 750,
#     'beds_occupied_ratio': 0.8   
# }

# problem = Problem(**problem_kwargs)
# solution = problem.generate_random_solution()
# print(problem.evaluate(solution))

ga_kwargs = {
    'num_shifts': 4,
    'num_workers': 50,
    'period': 7,
    'population_size': 20,
    'max_generation': 100,
    'num_parent': 10,
    'mutate_rate': 0.03,
    'target_score': 5000.0,
    'num_beds': 5,
}
ga = GA(**ga_kwargs)
ga.run_GA()

sa_kwargs = {
    'num_workers': 50,
    'period': 7,
    'num_shifts': 4,
    'max_iteration': 150,
    'target_score': 0.0,
    'T_max': 100,
    'T_min': 1e-7,
    'flip_rate': 0.1,
}
sa = SA(**sa_kwargs)
sa.run_sa()


ts_kwargs = {
    'num_workers': 50,
    'period': 7,
    'num_shifts': 4,
    'num_beds': 5,
    'max_iteration': 100,
    'num_neighbors': 20,
    'tabu_list_size': 100,
}
ts = TS(**ts_kwargs)
ts.run_TS()