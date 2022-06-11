import random
from textwrap import wrap
from tqdm import tqdm
from problem import Problem
from dataclasses import dataclass
from queue import PriorityQueue


@dataclass
class GA:
    num_shifts: int
    num_workers: int
    period: int # period must be a multiple of 7
    population_size: int
    max_generation: int
    num_parent: int
    mutate_rate: float
    target_score: float
    num_beds: int


    def init(self):
        # configurations
        problem_kwargs = {
            'num_workers': self.num_workers,
            'period': self.period,
            'num_shifts': self.num_shifts,
            'num_beds': self.num_beds,
            'beds_occupied_ratio': 0.8   
        }
        self.problem = Problem(**problem_kwargs)
        self.population_len = int(self.num_workers * self.period * self.num_shifts)
        self.best_history = []
        return None


    def generatePopulation(self) -> list:
        # schedule_list = [
        #     '1000', '0100', '0010',
        #     '1001', '0101', '0011',
        # ]
        # population = ''.join([random.choice(schedule_list) for _ in range(self.num_workers * self.period)])
        # population = ''.join([random.choice(['0','1']) for _ in range(self.num_workers * self.period * self.num_shifts)])
        population = ''.join(['0' for _ in range(self.num_workers * self.period * self.num_shifts)])
        return population


    # defined how "good" the solution is
    def evaluate_score(self) -> list:
        self.fitness = []
        for i in range(self.population_size):
            fit = {
                'population': self.population[i],
                'score': self.problem.evaluate(self.population[i]),
            }
            self.fitness.append(fit)
        return None


    def crossover(self, mate1: str, mate2: str) -> str:
        left = random.randint(0, len(mate1)/2)
        right = random.randint(len(mate1)/2 + 1, len(mate1)-1)
        return mate1[:left]+mate2[left: right]+mate1[right:]


    # recombine parents and generate child
    def recombine(self):
        parent = self.best_history[-self.num_parent:]
        new_population = []
        for _ in range(self.population_size):
            random.shuffle(parent)
            new_population.append(self.crossover(parent[0]['population'], parent[1]['population']))
        self.population = new_population
        return None


    def mutate(self):
        # mutate_num = int(self.mutate_rate * self.population_len)
        mutate_num = 3
        for i in range(len(self.population)):
            order = [c for c in self.population[i]]
            for flip in range(mutate_num):
                lucky = random.randint(0, len(order)-1)
                order[lucky] = '0' if order[lucky] == '1' else '1'
            self.population[i] = ''.join(order)
        return None


    def run_GA(self):
        
        self.init()

        # generating initial population (population)
        self.population = [self.generatePopulation() for i in range(self.population_size)]

        self.pbar = tqdm(range(self.max_generation))

        avg_score_list = []
        best_score_list = []
        for i in self.pbar:
            # evaluate fitness
            self.evaluate_score()

            self.best_history.extend(self.fitness)
            self.best_history = sorted(self.best_history, key=lambda d: d['score'])[-self.population_size:]
            avg_score = sum([data["score"] for data in self.best_history])/len(self.best_history)
            avg_score_list.append(avg_score)
            best_score_list.append(self.best_history[-1]['score'])
            print(f"avg score: {avg_score:.3f}\tbest score: {self.best_history[-1]['score']:.3f}")
            
            # recombine new population
            self.recombine()

            # mutate
            self.mutate()
        
        print(self.best_history[-1])
        print()
        print(avg_score_list)
        print()
        print(best_score_list)
        return None


"""
{
    'population': "00110010010110001000000001010100001001011001100001001000001101010010010001000011100010000101100001011001100000111011010101001001100100100101100100000101100110000101010010011001001010000010010110010100001010010011001000101001100000100000010100110101001110010101010010000010000001010010000001000101010000100101100101011000000010000101001010000010100000110010010011101000100101000000010100001000011001000101010000100101100000001001000000000100010000110101001110000101100101010010000001000000010001000010100010001011010001000011100110000011010001011000001000110011010000001000010101011001001100100011001100000010010100100011010000110101100000100011000000100000010001000101010010010100001100000011001100000011010100110011001110000100010100111001100001011000010010000010100001011001010001000100010101010000010100000011001110010100100101010101010000110011010110010000010100110100000000110101011110000100100101010011000010001001000001000100001100111000010110010011010100110011000000000011010100001001100000110011001100100100100010011001010100000100100110000011001101001001010010010100001101001001100110001001001000100100001100110100001001010010100100101001010000001111100100100011001100001000001100100000100110010000010101010011010100101000010010010011001000111000001101010010100001010100100010001000100100100010100000001000100010001001010100100011100001010010001110010101000000000000100010000100010101000011",
    'score': -52593.950, 
}, {
    'population': "00110010010110001000000001000100001001011001100001001000001101010010010001000011100000000101101001011001100000111001010101001001000100100101100100000101100110000101010010011001000010000010010110010100000010010011001000101001100000000000010100110101001110010101000010000000000001010010100001000101010000100101100101011000000010000101001010000010100000110010010011101000100101000000010100001000011001000000010000100101000001001001010000000100010000110100001110000101100101010010000000000000010001000010100010001001010001000011100110010011010001010100001000110011010000001000010101011001001100100000001100100010010100100011010100110101100000100011000001000000010100000000010010010000001100000011001100000011010100110011001111100101010100111001100001001000010010000010100001011001000001010000010101010000010101000011001110010100100101010101010100110011010101001000010100000101000000110101010110010100100101010011000010000101000001000101001100111001010010010011010100110011000000000011010100001001100000110011001100100000100010011001010100100000100110000101001101001001010100110100001101001001100100001000001000100100001100110000001001010010100100101001010000101111100100100011001100001001001100000000100110000000010101010011010100101000010010010011001000111000001101010011100001010100001110000000100000110011100000001001100010001000010100100011100001000011001110000101100000100000100000000100010101000011",
    'score': -52701.175,
}, {
    'population':"01001001010000110100010100110011111101000000001100000100100100101000001001000100100001011000001100111000001110001000001100110101010000000010001101000011001100110100001110010101100101000000000001010000000010001000000000110101100010000010100000110101010010000011000001001001001000001001010000100011001100111001100000000101100100100000001100100010100100111001100010000101100000000011001010010100001100110100010110010011010110010101010010011001100001000100001000111000001000110010001110011000001101010100010101001000010110010010100101010011010110000000001000000000010101001000010000001000010010000010100000110010010000110101000001011001010000100100001110010010001000001001010010000100100010110011010101010000100101000101010101000011001110010011001001010011110100100010100001000100010010001001100110010011001100100011100010010100100000000000100100100010100001010010001101010010010110010000100100110011100001010010001101001001100101010010001010011001010010011000100010000011000010000011001001000000100100100101010110010100100001000011100000110001001000100101010100101000001100100011010010000100010100101001010001001001100100100000010010010001100110000000001100000010100101010010010001011001100110001001001101010011100110000011010100110011001010000100001000100101001110000100001000100011001010001001001101000100010001010100001110010000110010000101001100100100001010001001100001010011000000110101010010000011",
    'score': -52579.982,
}, {
    'population': "01001001010000110101010100110011111001000000001000000101100100101000001001000100100001010000100100111001001110001000001100100100010100000000001000000011001100110100001110010101100101001000000001000000001010001001001100110101010110000110100100110100010110000010000001001001001000001001010000000011001100111001100001000101100100100000001100100010100100111001100010000101100000000011001010010100001100110101010111010011010110010101010010001001100001000100001000111001001100110010001110001000001101010000010101001000010110010010100001010011010110000000001000000000010101001000010000001000010010000010100000110010010000110101000001011000010100100101001110010010001010011000010010000100100100000011010101010100100101000100010101000011001110010011001001010011100100100010001101000100010010001001100110010011001100100011100010010000000000100000100100100011000000110010001101010010010110011000100100110011100001010010001110001001100101010010000010001001010010011000100110000011000000000011001001000000100100110101010110010100100001000010100000110000001000000101010100101001001100100011110000100100010100000000010001001001100100100000010010000011100110000100001100100010100101010011000001011001100110001000001101000011100010010011010100010011011110000100001100100100001110010100000000100011001010001000001101010101010001010101001110010000100000110101001100000100000010001001100001010011000000110101010010000011",
    'score': -52626.366,
}, {
    'population': "01000100111001000000001010000101100001001001010000110101010001011001010101011001100100000011010101011001010001011001010110000011100000110010010100101000001001000101100001000011010100001001010010010000100001010011100101010010100110000010100000101001010000001001010010000101000010010000010001010010010100001001100110011000001100111001100010010100010101011001100010000000001101010100000001010000100101000111000000101001100110000101010100000101010000000000100100000101010000100010010001000000010010000011100110010100100001010010010000101001001010000101100100000011010000101001010000110010001010011001001010000011001010001001010101011000010100111001010000000101111000000100001010001000010000101001000000111000001010001000001100100010001101001001010101000010001001010011001010010011001001010011001110000000100110000000001001000011001100100011001100100011010010010100010110010011000010010010010100000101010010010011000000110100001010000101100101001001001010010100001101000100100110010011010010010011000010000101001110010010001001100100100110000101000000001001001100100100100000111000000010000101100101000100100110000011000010010100010001011001010001000000010001000101100100001000001100111000100001010011000010000100100001000010100001011001100010010011010001010000010101111000010001010000001111001000010100110000100000001000001001010010010001010101010101010010100100001001010101000010100101010000010010011001",
    'score': -52628.374,
}, {
    'population': "01000101111001000000001010000101100001001001010000110101010001011001010101011001100100000011010101011001010001011001010110000011100000110010010100101000001001000101100001000011010100001001010010010000100001010011100101010010100110000010100000101001010000001001010010000101000010010000010001010010010100001001100110001000001100111001100010010100010001011001100010000000001101010100000001010000100101000011000000101001100110000101010100000101010001000000100100000101010000100000010001000000010010010011100110010101100001010011010000101001001010000101100101000011001000101001000000110010001010011001100010000011001010001001010001001000010100111001010000000101111000000100001110001000010000101001000000111000001010001000001100100010001101001001010101000010001001010011001010010011001001000011001110000000100110000000001001000010001100100011001100100011010010010101010110010011000010010010011100000101000011110011000000110100001010000101100101001001000110010100001101000100100110010011010010010011000010000101001110010010001001000101100110000101000000001001001100100100100000111000000010000101100101010100100110010011000010000100010001001001010001000000010001000101100100001000001100111000100101010011000010000100100001000010100001011001100010010011010001010000010100111010010001010000001010001001010100110000100001011000000001010010000001010101010101010010100000001001010101000010100101010100010010011001",
    'score': -52638.479,
}, {
    'population': "00000100001001010011001010000011001001010011100000110010100010000011000001010011010101000010010100100100100001010000010101011000001100111001100110000011001100100000010010011001001101000000100100100100010101011001001000100010010001010011100000100010000000110100100010011011010101000100100100100011000001011001010001010000001110010000001101011001100110010010100000110000100101000011001000100100010001000011001010000000000000110100010001010101100000000101010001011000100110010010100000110011100110000100100100110100100000101001100010010100010110000011010101001000100111010011010001000011010000101000100000000011100100100100001010000011010100110000100110010101100010001000001001000011010110000010010001011001100100111000100100101000001110011001010010001100010010001001010110000100001001011001001010000010010100111000100101000100001010000101100010000101100001000101100100111000010000110101100110001000001100101011001101010000100000000010000010001001001110010100010000000000010000111000010000110100001010000000001000100101001110010011010000000101001000100101000001010100001101000011010010010010001001010101001110001001010000001001011000001000010010110101000001000010100100110010001100110011100110001001100000110010001100111001010010010010100000100100100000100101001101011000001100110010001001010100001110001001000000110010000000100101010110011000000010000010001101010100010101010101010000110100010000110000",
    'score': -52565.815,
}, {
    'population': "00000100001001010011001000000011001001010011100000110010100010000011000001010011010101000011010100100100100001010000010101011000001100111001100110000011001100100000010010011000001101000000100100100100010101011001001000100010010001000011100000100010000000110100100010011011010101000100100100100011000001011001010001010000001110010000001101011001100110010010100000110000100101000011001000100000010001000011001010010000000000110100010001010101100000000101010001011000100110010010100000110011100110000100100100110100100000101001100010010100010110000011010101001000100101010011010001000011010000101000100000000011100100100100001010000011010100110000100110010101100010001000001001000011010110000010010001011001100100111000100100101000001110011001010010001100010010001001010010000101001001011000001010000010010100111000100101010100001010000101100010000101100101000101100100111001000000110101100110001000001100100011001101011000100000100010100010001001001110010101010010000000010000101000010000110100001010000000001000101111001110010011010000000101001000000101000001010100001100000011010010000010001001010101001110001001000000001001010000001000010000110101000001000010100100110011001000110011100110001001100000100010001101111000010010010010000000100000100000100101001101011001001100110000000001110101001110001001100000111000100000100101010110010001100010000010001100000101010001010101010000110000010100111001",
    'score': -52599.899,
}, {
    'population': "01010100001000110000100100000000010001000000001000110011100000000010010000110100001101010010100101010000001001010101100100100010010110011000001100110101100100100101001100100100100001001000100000001001001000110101010100110101000000000101100100101000010000100101100010011001010100100010100100110010100100110000001110010100010110010100001100111000011000110101001010001001000000100100100001001001000010001000010101000010010100000100001110010101010001010100001100110100000010011000010010000100001110000101100100110011010010000010001010011001010101010100001100100011100000101001001100101001100001010100100101001000001100100010000101010101100000110100001100000000010010010000010000101000100001000010100101001001010010000011001000000010000010000000010101011000001001010011100010010101010101010010000000000011000000111001100101001001001001010100001000001001100000000010001110010000100110000010010110010000100111001000010010001000100000110100010100100101100000100010100101000101010010010100001000000101001001010011001100100101100110000010001000110100000010010011100101010011001101000010010100110100100110010010001110011000001000101000001001000100010100000010001001011000001101010010000010001111100000101001010101000100100001000100010010010010100101010011010110001001100000111000001101010100011000110101010101011001000000100100001001001000010100110101010110010011001110000101100010011001001000101000010110001001",
    'score': -51596.074,
}, {
    'population': "01010100000000110000100100000100010001000000001000110011100101000010010000110100001101010010100101010000001001010101100000100010010110011000001100110101100100100101011100100100100001001000100000001000001000110101010100110101100000100101100100101000010000100101100010011001010100100011100100110010100100110000001110010000010110010100001100111000010000110101001010001001000000001100100001001001000010001000010101000010010110000100001110010101010001010100001100110100000010011000010010000110001100000101100100110011010010010010000010011001010101010100001100100011100000111001001100101000100001010100100101001000001100100010000101010101100000110100001001000000010010010000010000101000100101000010100101001001010110000011001010000010000010000100010101001000001001010011000010010101010101010010000000000011000000111001100101001001001001010100001001001001100000000010001110010000100110000010010110010010100001001001010110011000100000110100010100100101100000100010100101000101010010010100001000000101001101010011001100100101100110000010001000110101001010010011100101011011001100000010010100110100100110010010001110010000001000101000001001000100010100000010001001011001001101010010000010001101100000101000010101000100100101000100010010010010100101010011010100101001100000111000001101000100010000111001010101011001000000100100000001001000010100110101010110010011001110000101000010011001001000100000010110001001",
    'score': -52586.155,
}, {
    'population': "10001001001001010100010100100011001100111001100100111001100100110101001000110011001110011001001000110011010001000101010000110010010100101000100010011001100110000011001101000000001101011000001110000100100000000100001010001000010101010101001000000011010001000101001001000010100110001000001010010101000001011001000010011010100100111000010001001000010001000101100010001000100110001000100000110010100101010000010001011000010100010010001100100100010100100101010100100011100110011000100000111001001001000000100100001001100010011001001101000101100110010010001100110101000000110101000000101001100101011000001100000000001110010100100101000011100000101101001100111001100010000010010001000100001100001001100010001101100001010100100110001001100001010100010100100010010000100010010000110000000000110101001101000101100001010010000010011001001100100100010010010010001000110010100000110101100000110100100001011000100110010000000010010010010101000011001110000010010001010101100100001000010110001001000000110000100110001000010010010011010101000101001110010101010100000100010010010011001000000101100010010011100000100000100001010010001001000010001100110010100110000100001010000101100100110011001000100101001000100100100010001000100100000100010000110100010101000100010001010011010001011000001100000000100001011001010101001000100111111000010010010100100000110100100101001000110100100011001010010011001010001000001010010100",
    'score': -52541.671,
}, {
    'population': "10001001001001010100010100110011001100111001100100111001100100110101001000110011001110011000001000110011010001000100010000110010010100101000100010001001100110000011001101000000001101011000001010000100100000000100001010001000010101010101001000000011010001000101001001000010100110001000001010010101000001011001000010010010100100111000010001001000010001000101100010001000100110001000100000110010100101010000010001011000010101000010001100100100010100000101010100100011100110011000100000111001001001000000100100001001100010011001001001000101100110010010001100111101000000110101000000101001100101011000001100000000001110010100100101000011100000101001001100111001100110000010010001000100001100001001100010001001100001010100100110001001100001010101010100100010010000100010010000110000000000110101001101000101100101010010000010001001001100100100000010010011001000110000100000110101100000100100100001011000100110010000000010010010010101000011001110000010010001010101100100001000010110001001000000110000100100001000010010010011010101000101001110010101010110000100010010010011001000000101010010010011100000101000100001010010000001000010000100110010100110000000001010010101101100110011001000000101001000110100100000001000100100000100010100110100010001000100011001010011010001011001001100000000100001011001010101001000100101111000010010010100000000110100100101000000100100101111000010010011001010001000001010010100",
    'score': -52614.320,
}, {
    'population': "10001000001001010101010100100011001100101001100100111001100000000010010000110100001101010010100101010000001001010101100100100010010110011000001100110101100100100100001100100000100001001000100000001001001000110101010100110101000000000101100100101001010000100101100010011001010100100010100100110010100100110000001110010000010110010100001100111001010000110101001010001001000000000100100101001000000010001001010101000011010100000100001110010101010001010100001100110100000010011000010010000100001110010101100000110011010010000010001010011001010101010100001100100011100000101001001100101001100001010100100100001000001100100010001101010101100000110100001000000000010010010011010000101000000001000010001101001001010000100011001000000000000010000100010101011000001000110010010000110000000000110101001100000101100101010011010010001001001100100100000010010011001000110000100000110101001000100100100001011000100101010000000010010010010101000011001110000010010001000101100100001000010110001001000000111000100110001000010010010011010101000101001010010101010000000100010110010011001000000100100010010011100000110010100001010010000001000010001100110010100110000100001010000101100100110010001000100101001000100101100010001000100100000100000000110100010101000100010101000011010001011000001100000000100001011001010100000100100111111000010010010100100100110101100101001000100100100011001010010011001110001001001010010100",
    'score':-47644.537,
}, {
    'population': "00001000000001010101010100100011001100101001100100111001100000000010010000110100001101010010100101010000000001010101100100100010010110011000001100110101100100100101001100100100100001001000100100001001001000110101010100110101001000100101100100101000010000100101100010011001000100100010100101010010100100110000001010010100010110010100001100111000010000110101001010001001000000100100100001001001000010001000010101000010010100000100001110010101010001010100001100110100000010011000010010000100001100000101100000100011010010000010000010011001010101010100001100100011100000101000001100001001100001010100100101001000001100100010001101000101000000100100001101000100010010000000010000001000100001000010100001011001010010000011001000000010000010000101010000110011010000100010010100110000100000110100001101000100010100110000100101000100001010000101100010000101100101010101100100111001000000110100100110001000001100100011001101011000100000100010001010001000001110010101010010000000010000101000010100110100001010001000010010010011010101000101001010010101010000000100010110010011001000000101100010010011100000110000100001010010001001000010001100110010100110000100001010000101100100110011001000100101001000100101100010001000100100000100000000110100010101000100010101000011010001011000001100000000100101011001010101000100100111111000010010010100100100110100100101001000100100100011001010010011001010001000001010010100",
    'score':-48618.445,
}
"""