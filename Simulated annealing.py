import numpy as np
import matplotlib.pyplot as plt


class Simulated_annealing:
    def __init__(self, n_samples=50, t0=10, t_min=0.00001, max_iter=100000):
        self.n_samples = n_samples
        self.data = self.generate_data(self.n_samples)
        self.state0 = np.random.permutation(np.arange(self.n_samples))
        self.t0 = t0
        self.t_min=t_min
        self.max_iter = max_iter
    
    
    def generate_data(self, a=1, b=-1):
        """Функция генерирования случайных координат"""
        data = np.random.sample((self.n_samples, 2)) * (a - b) - a
        return data
    
    
    def distance(self, state):
        """Функция вычисления растояния на основании состояния"""
        data = self.data[state].copy()
        distance = 0
        for i in range(data.shape[0]-1):
            distance += np.sqrt(np.power((data[i, 0] - data[i+1, 0]), 2) + np.power((data[i, 1] - data[i+1, 1]), 2))
        distance += np.sqrt(np.power((data[-1, 0] - data[0, 0]), 2) + np.power((data[-1, 1] - data[0, 1]), 2))
        return distance
    
    
    def generate_state_candidate(self, state):
        """
        Функция генерирования нового состояния-кандидата.
        state - текущее состояние.
        Генерация на основе обращения последовательности.
        """
        n = state.shape[0]
        i =  np.random.randint(1, n+1)
        j = np.random.randint(1, n+1)
        while (i == j):
            i =  np.random.randint(1, n+1)
            j = np.random.randint(1, n+1)
        new_state = state.copy()
        if (i > j):
            new_state[j:i] = new_state[j:i][::-1]
        else:
            new_state[i:j] = new_state[i:j][::-1]
        return new_state
    
    
    def get_transcation_probality(self, delta_e, t):
        return np.exp(-delta_e/t)


    def is_transaction(self, probality):
        p = np.random.rand()
        if (p <= probality):
            return True
        else:
            return False
        
    
    def decrease_temperature(self, i):
        return self.t0 * 0.1 / i
    
    
    def simuleated(self):
        """
        Имитация отжига.
        """
        state = self.state0
        current_distance = self.distance(state)
        t = self.t0
        for i in range(1, self.max_iter):
            new_state = self.generate_state_candidate(state)
            candidate_distance = self.distance(new_state)
            
            if (candidate_distance < current_distance):
                current_distance = candidate_distance
                state = new_state
            else:
                p = self.get_transcation_probality(candidate_distance - current_distance, t)
                if (self.is_transaction(p)):
                    current_distance = candidate_distance
                    state = new_state
            
            t = self.decrease_temperature(i)
            if (t <= self.t_min):
                print('Достигнута минимальная температура на шаге: ', i)
                break
        return new_state

if __name__ == '__main__':
    
    SA = Simulated_annealing(n_samples=50)
    plt.title('Исходная выборка. {} точек'.format(SA.n_samples))
    plt.scatter(SA.data[:, 0], SA.data[:, 1])
    plt.show()
    
    plt.title('Начальное состояние. Дистанция: {}'.format(SA.distance(SA.state0)))
    plt.plot(SA.data[SA.state0, 0], SA.data[SA.state0, 1])
    plt.show()
    
    new_state = SA.simuleated()
    plt.title('Итоговое состояние. Дистанция: {}'.format(SA.distance(new_state)))
    plt.plot(SA.data[new_state, 0], SA.data[new_state, 1])
    plt.show()
    
    
    
    
