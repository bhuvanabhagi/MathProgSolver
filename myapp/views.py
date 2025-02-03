import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

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

            # Check for negative inputs in constraints
            if any(x < 0 for x in a + b + rhs):
                raise ValueError("Constraint values must be non-negative.")

            constraints = zip(a, b, rhs)  # Store constraints to pass back to the template

            # Create plot
            x = np.linspace(0, max([rhs_i/a_i if a_i != 0 else 0 for a_i, rhs_i in zip(a, rhs)]) + 5, 100)
            
            # Plot constraints
            plt.figure(figsize=(10, 8))
            for i in range(len(a)):
                if b[i] != 0:
                    y = (rhs[i] - a[i] * x) / b[i]
                    plt.plot(x, y, label=f'{a[i]}x + {b[i]}y ≤ {rhs[i]}')

            # Plot non-negativity constraints
            plt.axhline(y=0, color='k', linestyle='-')
            plt.axvline(x=0, color='k', linestyle='-')

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
                if all(a[i] * point[0] + b[i] * point[1] <= rhs[i] for i in range(len(a))):
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

                # Plot feasible points and optimal point
                x_coords, y_coords = zip(*feasible_points)
                plt.scatter(x_coords, y_coords, color='blue', label='Feasible points')
                plt.scatter([optimal_point[0]], [optimal_point[1]], color='red', s=100,
                          label=f'Optimal point: ({optimal_point[0]:.2f}, {optimal_point[1]:.2f})\nZ={optimal_value:.2f}')

            plt.grid(True)
            plt.legend()
            plt.xlabel('x₁')
            plt.ylabel('x₂')
            plt.title('Linear Programming Solution (Graphical Method)')

            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
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
                'constraints': list(zip(a, b, rhs))  # Include constraints to pass back to the template
            }

            return render(request, 'graphical.html', results)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render(request, 'graphical.html', {
                'error': error_message,
                'optimization_type': request.POST.get('optimization_type'),
                'c1': request.POST.get('c1'),
                'c2': request.POST.get('c2'),
                'constraints': list(zip(request.POST.getlist('a[]'), request.POST.getlist('b[]'), request.POST.getlist('rhs[]')))
            })

    return render(request, 'graphical.html', {
        'optimization_type': request.POST.get('optimization_type'),
        'c1': request.POST.get('c1'),
        'c2': request.POST.get('c2'),
        'constraints': list(zip(request.POST.getlist('a[]'), request.POST.getlist('b[]'), request.POST.getlist('rhs[]')))
    })
