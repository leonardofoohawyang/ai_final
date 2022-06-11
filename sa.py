from dataclasses import dataclass
import random
from problem import Problem
import math
import numpy as np
import itertools

@dataclass
class SA:
    num_workers: int = 30
    period: int = 7 # period must be a multiple of 7
    num_shifts: int = 4
    max_iteration: int = 100
    target_score: float = 0
    T_max: float = 100
    T_min: float = 1e-7
    flip_rate: float = 0.15

    def generate_random_solution(self):
        schedule_list = [
            '1000', '0100', '0010',
            '1001', '0101', '0011',
        ]
        s = ''.join(random.choice(schedule_list) for _ in range(self.num_workers * self.period))
        # s = ''.join([random.choice(['0', '1']) for _ in range(self.num_workers * self.period * self.num_shifts)])
        # s = ''.join(['0' for _ in range(self.num_workers * self.period * self.num_shifts)])
        return s


    def init(self):
        problem_kwargs = {
            'num_workers': self.num_workers,
            'period': self.period,
            'num_shifts': self.num_shifts,
            'num_beds': self.num_workers,
            'beds_occupied_ratio': 0.8,   
        }
        self.problem = Problem(**problem_kwargs)
        self.best_x = self.generate_random_solution()
        self.best_y = self.problem.evaluate(self.best_x)
        self.T = self.T_max
        self.generation_best_X, self.generation_best_Y = [self.best_x], [self.best_y]
        self.best_x_history, self.best_y_history = self.generation_best_X, self.generation_best_Y
        return None
    

    def get_new_x(self, x :str):
        table = np.reshape(list(x), (self.num_workers, self.period, self.num_shifts))
        # flip_num = math.ceil(len(x) * self.flip_rate)
        flip_num = 5
        for _ in range(flip_num):
            i = random.randint(0,self.num_workers-1)
            j = random.randint(0,self.period-1)
            k = random.randint(0,self.num_shifts-1)
            table[i][j][k] = '1' if table[i][j][k] == '0' else '0'
        xnew = ''.join(list(itertools.chain.from_iterable(itertools.chain.from_iterable(table.tolist()))))
        return xnew

    
    def cool_down(self):
        self.T = self.T * 0.85
        return None


    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    

    def run_sa(self):

        self.init()

        x_current, y_current = self.best_x, self.best_y
        iteration = 0

        avg_score = []
        best_score = []
        while True:
            for i in range(self.max_iteration):
                x_new = self.get_new_x(x_current)
                y_new = self.problem.evaluate(x_new)

                # Metropolis
                if y_new > y_current:
                    x_current, y_current = x_new, y_new
                    if y_new > self.best_y:
                        self.best_x, self.best_y = x_new, y_new
                
            self.cool_down()
            self.generation_best_Y.append(self.best_y)
            self.generation_best_X.append(self.best_x)

            avg = sum(self.best_y_history[-10:])/len(self.best_y_history[-10:])
            print(f"avg score: {avg:.3f}\tbest score: {self.best_y:.3f}")
            avg_score.append(avg)
            best_score.append(self.best_y)

            # if best_y stay for max_iteration times, stop iteration
            if self.isclose(self.best_y_history[-1], self.best_y_history[-2]):
                iteration += 1
            else:
                iteration = 0

            if self.T < self.T_min:
                break
            
            if iteration > self.max_iteration:
                break

        print(f"Best score: {self.best_y}")
        print(self.best_x)

        print("avg score list: ", avg_score)
        print("best score list: ", best_score)
            
        return None