from dataclasses import dataclass
import random
import numpy as np

@dataclass
class Problem:
    num_workers: int
    period: int # period must be a multiple of 7
    num_shifts: int
    num_beds: int
    beds_occupied_ratio: float
    convert: bool = True
    
    def generate_random_solution(self):
        return ''.join([random.choice(['0', '1']) for _ in range(self.num_workers * self.period * self.num_shifts)])

    def evaluate(self, solution):
        HARD_CONSTRAINT_COST = 25
        SOFT_CONSTRAINT_COST = 2
        RATIO_IMPORTANCE = np.array([i*5 for i in [0.5, 0.25, 0.25]]) # the importance of the patients: workers ratio

        if self.convert:
            # convert the solution string to 3d array
            table = np.reshape(list(solution), (self.num_workers, self.period, self.num_shifts))
        else:
            table = solution
        
        fitness = 0
        for i in range(self.num_workers):
            monthly_working_hours, biweek_offday_count, weekly_shifts_count = 0, 0, np.zeros((3))
            for j in range(self.period):
                if j % 28 == 0: # new month
                    # monthly (4 weeks) working hours must not exceed 160 hours
                    if monthly_working_hours > 160:
                        fitness -= HARD_CONSTRAINT_COST
                    monthly_working_hours = 0
                if j % 14 == 0:
                    # at least 2 days off every 2 weeks
                    if biweek_offday_count < 2:
                        fitness -= HARD_CONSTRAINT_COST
                    biweek_offday_count = 0
                if j % 7 == 0: # new week
                    # only 1 type of shift will be assigned to a worker in the same week
                    fitness -= np.sum(np.delete(weekly_shifts_count, np.argmax(weekly_shifts_count))) * SOFT_CONSTRAINT_COST
                    weekly_shifts_count = [0, 0, 0]

                normal_shift_count = np.count_nonzero(table[i][j][0:3] == '1')
                overtime_count = int(table[i][j][3] == '1')
                # OT must follow a normal shift
                if overtime_count == 1 and normal_shift_count == 0:
                    fitness -= HARD_CONSTRAINT_COST

                # daily working hours must not exceed 10 hours (1 normal + 1 OT)
                if normal_shift_count >= 2:
                    fitness -= HARD_CONSTRAINT_COST
                
                working_hours = 8 * normal_shift_count + 2 * overtime_count
                fitness += working_hours

                monthly_working_hours += working_hours
                # print(f'worker {i}, day {j}, worked {monthly_working_hours} hours this month')
                weekly_shifts_count = np.add(weekly_shifts_count, table[i][j][0:3].astype(np.int))
                # print(f'worker {i}, day {j}, shifts {weekly_shifts_count} this week')
        
        num_patients = self.num_beds * self.beds_occupied_ratio

        for i in range(self.period):
            num_workers = {
                'regular_shift': np.count_nonzero(table[:, i, 0] == '1'),
                'night_shift': np.count_nonzero(table[:, i, 1] == '1'),
                'graveyard_shift': np.count_nonzero(table[:, i, 2] == '1')
            }
            intended_ratio = {
                'regular_shift': 6,
                'night_shift': 8,
                'graveyard_shift': 10
            }
            patients_workers_ratio = {shift: min(num_patients / (num_workers[shift] + 0.01), 100) for shift in num_workers.keys()}
            # fitness += RATIO_IMPORTANCE * np.sum([intended_ratio[shift] - patients_workers_ratio[shift] for shift in intended_ratio.keys()])
            fitness += np.inner(RATIO_IMPORTANCE, [intended_ratio[shift] - patients_workers_ratio[shift] for shift in intended_ratio.keys()])
        
        return fitness