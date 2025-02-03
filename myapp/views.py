import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
from scipy.optimize import linprog

def home(request):
    return render(request, 'home.html')

import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull  # Add this import

import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from django.shortcuts import render

import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from django.shortcuts import render

def graphical_method(request):
    if request.method == 'POST':
        try:
            # Get objective function coefficients
            c1 = float(request.POST['c1'])
            c2 = float(request.POST['c2'])
            opt_type = request.POST['optimization_type']

            # Check for negative inputs
            if c1 < 0 or c2 < 0:
                raise ValueError("Objective function coefficients must be non-negative.")

            # Get constraints
            a = list(map(float, request.POST.getlist('a[]')))
            b = list(map(float, request.POST.getlist('b[]')))
            rhs = list(map(float, request.POST.getlist('rhs[]')))
            ineq = request.POST.getlist('ineq[]')

            # Check for negative inputs in constraints
            if any(x < 0 for x in a + b + rhs):
                raise ValueError("Constraint values must be non-negative.")

            constraints = zip(a, b, rhs, ineq)  # Store constraints to pass back to the template

            # Create plot
            x = np.linspace(0, max([rhs_i/a_i if a_i != 0 else 0 for a_i, rhs_i in zip(a, rhs)]) + 5, 100)
            
            # Define colors for constraint lines
            colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#1abc9c', '#34495e']
            
            # Plot constraints
            plt.figure(figsize=(10, 8))
            for i in range(len(a)):
                if b[i] != 0:
                    y = (rhs[i] - a[i] * x) / b[i]
                    y = np.clip(y, 0, None)  # Clip y values to be non-negative (first quadrant only)
                    plt.plot(x, y, 
                            color=colors[i % len(colors)],  # Cycle through colors
                            linewidth=2,  # Make lines thicker
                            label=f'{a[i]}x + {b[i]}y {ineq[i]} {rhs[i]}')

            # Plot non-negativity constraints
            plt.axhline(y=0, color='#2c3e50', linestyle='-', linewidth=1.5)
            plt.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=1.5)

            # Find intersection points
            intersection_points = [(0, 0)]
            for i in range(len(a)):
                if b[i] != 0:
                    intersection_points.append((0, rhs[i] / b[i]))
                if a[i] != 0:
                    intersection_points.append((rhs[i] / a[i], 0))
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    if b[i] != 0 and b[j] != 0:
                        A = np.array([[a[i], b[i]], [a[j], b[j]]])
                        b_vec = np.array([rhs[i], rhs[j]])
                        try:
                            x_intersect = np.linalg.solve(A, b_vec)
                            if x_intersect[0] >= 0 and x_intersect[1] >= 0:
                                intersection_points.append((x_intersect[0], x_intersect[1]))
                        except np.linalg.LinAlgError:
                            pass

            # Find feasible points
            feasible_points = []
            for point in intersection_points:
                if all((a[i] * point[0] + b[i] * point[1] <= rhs[i] if ineq[i] == '<=' else a[i] * point[0] + b[i] * point[1] >= rhs[i]) for i in range(len(a))):
                    feasible_points.append(point)

            # Find optimal point
            if feasible_points:
                z_values = [c1 * point[0] + c2 * point[1] for point in feasible_points]
                if opt_type == 'max':
                    optimal_index = np.argmax(z_values)
                else:
                    optimal_index = np.argmin(z_values)
                optimal_point = feasible_points[optimal_index]
                optimal_value = z_values[optimal_index]

                # Create a convex hull from feasible points and shade feasible region
                if len(feasible_points) >= 3:
                    feasible_points_array = np.array(feasible_points)
                    hull = ConvexHull(feasible_points_array)
                    hull_points = feasible_points_array[hull.vertices]
                    plt.fill(hull_points[:, 0], hull_points[:, 1], 
                            color='red', alpha=0.2, 
                            label='Feasible Region')
                
                # Plot optimal point
                plt.scatter([optimal_point[0]], [optimal_point[1]], 
                          color='#2c3e50', 
                          s=100,
                          zorder=5,
                          label=f'Optimal point: ({optimal_point[0]:.2f}, {optimal_point[1]:.2f})\nZ={optimal_value:.2f}')

                # Enhance grid appearance
                plt.grid(True, linestyle=':', alpha=0.3)
                
                # Customize the appearance of the plot
                plt.gca().set_facecolor('white')
                for spine in plt.gca().spines.values():
                    spine.set_color('#2c3e50')

                plt.legend(framealpha=0.9, facecolor='white', edgecolor='#bdc3c7')
                plt.xlabel('x₁', fontsize=12)
                plt.ylabel('x₂', fontsize=12)
                plt.title('Linear Programming Solution (Graphical Method)', 
                         fontsize=14, pad=20)

                # Convert plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                plt.close()

                graph = base64.b64encode(image_png).decode()

                # Prepare results
                results = {
                    'graph': graph,
                    'x1': f"{optimal_point[0]:.2f}",
                    'x2': f"{optimal_point[1]:.2f}",
                    'z': f"{optimal_value:.2f}",
                    'optimization_type': opt_type,
                    'c1': c1,
                    'c2': c2,
                    'constraints': list(zip(a, b, rhs, ineq))
                }

                return render(request, 'graphical.html', results)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render(request, 'graphical.html', {
                'error': error_message,
                'optimization_type': request.POST.get('optimization_type'),
                'c1': request.POST.get('c1'),
                'c2': request.POST.get('c2'),
                'constraints': list(zip(request.POST.getlist('a[]'), 
                                     request.POST.getlist('b[]'), 
                                     request.POST.getlist('rhs[]'), 
                                     request.POST.getlist('ineq[]')))
            })

    return render(request, 'graphical.html', {
        'optimization_type': request.POST.get('optimization_type'),
        'c1': request.POST.get('c1'),
        'c2': request.POST.get('c2'),
        'constraints': list(zip(request.POST.getlist('a[]'), 
                              request.POST.getlist('b[]'), 
                              request.POST.getlist('rhs[]'), 
                              request.POST.getlist('ineq[]')))
    })



import numpy as np
from django.shortcuts import render

def simplex_method(request):
    if request.method == 'POST':
        try:
            # Get the number of variables and constraints
            num_vars = int(request.POST.get('num_vars', 2))
            num_constraints = int(request.POST.get('num_constraints', 2))
            
            # Get objective function coefficients
            obj_func = [float(request.POST.get(f'obj_coeff_{i}', 0)) for i in range(num_vars)]
            
            # Get constraints
            constraints = []
            rhs_values = []
            for i in range(num_constraints):
                constraint = [float(request.POST.get(f'constraint_{i}_{j}', 0)) for j in range(num_vars)]
                constraints.append(constraint)
                rhs_values.append(float(request.POST.get(f'rhs_{i}', 0)))
            
            # Convert to numpy arrays
            c = np.array(obj_func)
            A = np.array(constraints)
            b = np.array(rhs_values)
            
            # Add slack variables
            num_total = num_vars + num_constraints
            tableau = np.zeros((num_constraints + 1, num_total + 1))
            
            # Fill tableau
            tableau[0, 0:num_vars] = -c
            tableau[1:, 0:num_vars] = A
            tableau[1:, num_vars:num_total] = np.eye(num_constraints)
            tableau[1:, -1] = b
            
            # Iteration variables
            iterations = []
            iteration = 0
            max_iterations = 100  # Prevent infinite loop
            
            # Simplex iterations
            while iteration < max_iterations:
                iterations.append(tableau.copy())
                
                # Find entering variable (most negative coefficient)
                entering_col = np.argmin(tableau[0, :-1])
                if tableau[0, entering_col] >= 0:
                    break  # Optimal solution found
                
                # Find leaving variable (minimum ratio test)
                ratios = [(i, tableau[i, -1] / tableau[i, entering_col]) 
                          if tableau[i, entering_col] > 0 else (i, float('inf')) 
                          for i in range(1, tableau.shape[0])]
                
                if all(r[1] == float('inf') for r in ratios):
                    return render(request, 'simplex.html', {'error': 'Problem is unbounded'})
                
                leaving_row = min(ratios, key=lambda x: x[1])[0]
                
                # Perform row operations
                pivot = tableau[leaving_row, entering_col]
                tableau[leaving_row] /= pivot
                for i in range(tableau.shape[0]):
                    if i != leaving_row:
                        tableau[i] -= tableau[i, entering_col] * tableau[leaving_row]
                
                iteration += 1
            
            # Extract solution
            solution = np.zeros(num_vars)
            for j in range(num_vars):
                col = tableau[:, j]
                if np.sum(col) == 1 and np.sum(col == 1) == 1:
                    solution[j] = tableau[np.where(col == 1)[0][0], -1]
            
            optimal_value = -tableau[0, -1]
            
            # Pass results to template
            context = {
                'optimal_value': f"{optimal_value:.2f}",
                'solution': [f"{x:.2f}" for x in solution],
                'iterations': iteration,
                'num_vars': num_vars,
                'num_constraints': num_constraints,
                'objective_coefficients': obj_func,
                'constraints': constraints,
                'rhs_values': rhs_values,
                'var_range': range(num_vars),  # Pass range for template
                'constraint_range': range(num_constraints),
            }
            
            return render(request, 'simplex.html', context)
            
        except Exception as e:
            return render(request, 'simplex.html', {'error': f"An error occurred: {str(e)}"})
    
    return render(request, 'simplex.html', {'num_vars': 2, 'num_constraints': 2, 'var_range': range(2), 'constraint_range': range(2)})
