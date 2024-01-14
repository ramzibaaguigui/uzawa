import sys

sys.path.append('./../.')


import numpy as np
import matplotlib.pyplot as plt

from typing import List
from utils.utils import Function

class Plotter(object):
    def __init__(self, f, constraints):
        self.f = f
        self.constraints = constraints

    def plot(self, constraint_list):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)

        used_constraints = [(name, con) for name, con in self.constraints if name in constraint_list]

        # Stack the X and Y grids into a single 2D array
        XY = np.column_stack((X.flatten(), Y.flatten()))
        # Evaluate the function on the grid
        Z = np.array([self.f.compute(point) for point in XY]).reshape(X.shape)
        constraints_values = [(name, con, np.array([con.compute(point) for point in XY]).reshape(X.shape)) for name, con in used_constraints]
        # Create a contour plot
        plt.figure(figsize=(30, 24.6))
        contour_plot = plt.contour(X, Y, Z, levels=20, cmap='viridis')  # Change levels and cmap as needed
        
        constraints_contours = [plt.contour(X, Y, constraint_values, levels=[0], cmap='viridis') for  
                                name, constraint, constraint_values in constraints_values]
        

        plt.colorbar(contour_plot, label='Function Values')
        plt.title('Contour Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()


plotter = Plotter(f=None, constraints=[])

    