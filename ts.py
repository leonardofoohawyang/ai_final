import random
from problem import Problem
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import copy
import itertools


@dataclass
class TS:
    num_workers: int
    period: int
    num_shifts: int
    num_beds: int
    max_iteration: int
    num_neighbors: int
    tabu_list_size: int


    def init(self):
        problem_kwargs = {
            'num_workers': self.num_workers,
            'period': self.period,
            'num_shifts': self.num_shifts,
            'num_beds': self.num_beds,
            'beds_occupied_ratio': 0.8,
            'convert': False,
        }
        self.problem = Problem(**problem_kwargs)
        solution = self.generate_initial_solution()
        self.best = {'solution': solution, 'score': self.problem.evaluate(solution)}
        self.best_candidate = self.best
        self.tabu_list = [self.best]
        self.tabu_list_score = [self.best['score']]
        return None


    def generate_initial_solution(self):
        schedule_list = [
            '1000', '0100', '0010',
            '1001', '0101', '0011',
        ]
        solution = ''.join(random.choice(schedule_list) for _ in range(self.num_workers * self.period))
        # solution = ''.join("0" for _ in range(self.num_workers * self.period * self.num_shifts))
        return np.reshape(list(solution), (self.num_workers, self.period, self.num_shifts))


    def get_neighbor(self):
        self.neighbors = []
        for _ in range(self.num_neighbors):
            neighbor = copy.deepcopy(self.best['solution'])
            for _ in range(5):
                i = random.randint(0,self.num_workers-1)
                j = random.randint(0,self.period-1)
                k = random.randint(0,self.num_shifts-1)
                neighbor[i][j][k] = '1' if neighbor[i][j][k] == '0' else '0'
                    
            self.neighbors.append({
                'solution': neighbor,
                'score': self.problem.evaluate(neighbor),
            })
        return None

    
    def run_TS(self):
        self.init()

        avg_score = []
        best_score = []
        for i in tqdm(range(self.max_iteration)):
            self.get_neighbor()
            self.best_candidate = self.neighbors[0]
            
            for neighbor in self.neighbors:
                if neighbor['score'] > self.best_candidate['score'] and not neighbor['score'] in self.tabu_list_score:
                    self.best_candidate = neighbor

            if self.best_candidate['score'] > self.best['score']:
                self.best = self.best_candidate
            
            self.tabu_list.append(self.best_candidate)
            self.tabu_list_score.append(self.best_candidate['score'])

            self.tabu_list = self.tabu_list[-self.tabu_list_size:]
            self.tabu_list_score = self.tabu_list_score[-self.tabu_list_size:]
            avg = sum(self.tabu_list_score)/len(self.tabu_list_score)
            print(f"avg score: {avg:.3f}\tbest score: {self.best['score']:.3f}")
            avg_score.append(avg)
            best_score.append(self.best['score'])

        # print(f"Best Score: {self.best['score']}")
        # print(f"{''.join(list(itertools.chain.from_iterable(itertools.chain.from_iterable(self.best['solution'].tolist()))))}")
        print("avg score list: ", avg_score)
        print("best score list: ", best_score)
        return None