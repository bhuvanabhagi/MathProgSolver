# views.py
from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy.optimize import linprog

def home(request):
    return render(request, 'home.html')
# Helper function to parse constraints
def parse_constraints(constraints_input):
    return [list(map(float, c.split())) for c in constraints_input.split(',')]

# Graphical Method
def graphical_method(request):
    if request.method == "POST":
        try:
            # Get optimization type (maximize/minimize)
            optimization_type = request.POST.get('optimization_type')
            obj_func = list(map(float, request.POST['objective_function'].split()))  # Objective function
            constraints = parse_constraints(request.POST['constraints'])  # Constraints

            # Setup A, b matrices for constraints
            A, b = np.array([c[:-1] for c in constraints]), np.array([c[-1] for c in constraints])

            feasible_vertices = []
            # Solve for vertices using constraints
            for i in range(len(A)):
                for j in range(i + 1, len(A)):
                    try:
                        vertex = np.linalg.solve(A[[i, j]], b[[i, j]])
                        if all(np.dot(A, vertex) <= b) and all(vertex >= 0):
                            feasible_vertices.append(vertex)
                    except:
                        pass

            feasible_vertices = np.array(feasible_vertices)

            # Maximize or minimize the objective function
            z_values = [np.dot(obj_func, v) for v in feasible_vertices]
            if optimization_type == "maximize":
                optimal_vertex = feasible_vertices[np.argmax(z_values)]
                optimal_value = np.max(z_values)
            else:  # Minimize
                optimal_vertex = feasible_vertices[np.argmin(z_values)]
                optimal_value = np.min(z_values)

            # Plot the result
            plt.figure()
            plt.plot(optimal_vertex[0], optimal_vertex[1], 'ro')
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("Graphical Solution")
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = base64.b64encode(buf.getvalue()).decode()

            return render(request, 'graphical.html', {'image': image, 'optimal_value': optimal_value, 'optimal_vertex': optimal_vertex})

        except Exception as e:
            return render(request, 'graphical.html', {'error': str(e)})

    return render(request, 'graphical.html')

# Simplex Method
def simplex_method(request):
    if request.method == "POST":
        try:
            # Get optimization type (maximize/minimize)
            optimization_type = request.POST.get('optimization_type')
            c = np.array(list(map(float, request.POST['objective_function'].split())))  # Objective function
            constraints = parse_constraints(request.POST['constraints'])  # Constraints

            A, b = np.array([c[:-1] for c in constraints]), np.array([c[-1] for c in constraints])

            # Change the optimization type: max or min
            if optimization_type == "minimize":
                result = linprog(c, A_ub=A, b_ub=b, method='highs')
            else:  # maximize (negate the objective function for maximization)
                result = linprog(-c, A_ub=A, b_ub=b, method='highs')

            return render(request, 'simplex.html', {'solution': result.x, 'optimal_value': result.fun, 'optimization_type': optimization_type})

        except Exception as e:
            return render(request, 'simplex.html', {'error': str(e)})

    return render(request, 'simplex.html')

# Transportation Method
def transportation_method(request):
    if request.method == "POST":
        try:
            # Parse inputs
            cost_matrix = np.array([list(map(int, row.split())) for row in request.POST['cost_matrix'].split(';')])
            supply = np.array(list(map(int, request.POST['supply'].split())))
            demand = np.array(list(map(int, request.POST['demand'].split())))

            # Formulate the transportation problem
            c = cost_matrix.flatten()
            A_eq = []
            b_eq = []

            # Constraints for supply
            for i in range(len(supply)):
                row = [0] * len(c)
                for j in range(len(demand)):
                    row[i * len(demand) + j] = 1
                A_eq.append(row)
                b_eq.append(supply[i])

            # Constraints for demand
            for j in range(len(demand)):
                col = [0] * len(c)
                for i in range(len(supply)):
                    col[i * len(demand) + j] = 1
                A_eq.append(col)
                b_eq.append(demand[j])

            # Solve using linear programming
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

            return render(request, 'transportation.html', {'solution': result.x.reshape(cost_matrix.shape), 'total_cost': result.fun})

        except Exception as e:
            return render(request, 'transportation.html', {'error': str(e)})

    return render(request, 'transportation.html')
