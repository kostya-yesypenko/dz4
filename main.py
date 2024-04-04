import graphviz
import numpy as np

# Значення інтенсивностей переходів
intensities = {
    ('S1', 'S2'): 2.2,
    ('S1', 'S3'): 1.4,
    ('S2', 'S1'): 1.2,
    ('S2', 'S4'): 0.02,
    ('S3', 'S1'): 1.5,
    ('S3', 'S4'): 0.01,
    ('S4', 'S1'): 0.25,
    ('S4', 'S2'): 0.25
}

# Побудова графу
dot = graphviz.Digraph()
for transition, intensity in intensities.items():
    start_state, end_state = transition
    dot.edge(start_state, end_state, label=str(intensity))

# Збереження та відображення графу
dot.render('microclimate_graph', format='png', cleanup=True)
dot.view()



def get_transition_matrix(number_states):
    transition_matrix = np.zeros((number_states, number_states))

    for i in range(number_states):
        for j in range(number_states):
            while True:
                value = input(f'Введіть інтенсивність переходу λ{i + 1}{j + 1} (або натисніть Enter для завершення): ')
                if not value.strip():
                    break
                try:
                    float_value = float(value)
                    transition_matrix[i][j] = float_value
                    break
                except ValueError:
                    print('Неправильне значення. Будь ласка, введіть число або залиште порожнім для пропуску')

    return transition_matrix


def get_kolmogorov_matrix(transition_matrix):
    row_sums = np.sum(transition_matrix, axis=1)
    kolmogorov_matrix = transition_matrix.T
    np.fill_diagonal(kolmogorov_matrix, -row_sums)
    return kolmogorov_matrix


def solve_linear_equations(A, b):
    solutions, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    return solutions


def print_kolmogorov_equations(k_m):
    print('Система диференціальних рівнянь Колмогорова:')
    for i, row in enumerate(k_m):
        equation = "(p{})' = ".format(i + 1)
        non_zero_entries = [(f"{coeff:.2f}p{j + 1}") for j, coeff in enumerate(row) if coeff != 0]
        equation += " + ".join(non_zero_entries)
        print(equation)


def print_linear_equations(A, b):
    print('Система лінійних алгебраїчних рівнянь СЛАР для обчислення граничних ймовірностей:')
    for i, row in enumerate(A):
        equation = ' + '.join([f"{coeff:.2f}p{j + 1}" for j, coeff in enumerate(row) if coeff != 0])
        print(f"{equation} = {b[i]}")


number_states = int(input('Введіть кількість станів системи: '))
transition_matrix = get_transition_matrix(number_states)

kolmogorov_matrix = get_kolmogorov_matrix(transition_matrix)
print_kolmogorov_equations(kolmogorov_matrix)

A = kolmogorov_matrix
A[-1] = [1] * number_states
b = np.array([0] * (number_states - 1) + [1])

print_linear_equations(A, b)

solutions = solve_linear_equations(A, b)
print("Розв'язок:")
for i, p in enumerate(solutions, 1):
    print(f'p{i} = {p:.4f}')

