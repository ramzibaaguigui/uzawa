
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List
from matplotlib.colors import LinearSegmentedColormap



# here we consider defining the general framework that we will work with

class Function(object):

    # init the function, pass the way to compute hessian and all that staff

    def __init__(self, compute: Callable, gradient: Callable, hessian: Callable, constraints=[]) -> None:
        self.compute = compute 
        self.gradient = gradient
        self.hessian = hessian
    
    def __add__(self, other):
        return Function(lambda x: self.__compute__(x) + other.__compute__(x),
                        lambda x: self.__gradient__(x) + other.__gradient__(x),
                        lambda x: self.__hessian__(x) + other.__hessian__(x))

    def __rmul__(self, value):
        return Function(lambda x: value * self.__compute__(x),
                        lambda x: value * self.__gradient__(x),
                        lambda x: value * self.__hessian__(x))

      
    
    

    def __getitem__(self, x):
        return self.compute(x)
    
    def __compute__(self, x):
        xx = x.reshape(-1)
        return self.compute(xx)
    
    def __gradient__(self, x):
        xx = x.reshape(-1)
        return self.gradient(xx)
    
    def __hessian__(self, x):
        xx = x.reshape(-1)
        return self.hessian(xx)

    def add_constraint(self, f: 'Function'):
        self.constraints.append(f)

    def plot_contour(self):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)

        # Stack the X and Y grids into a single 2D array
        XY = np.column_stack((X.flatten(), Y.flatten()))

        # Evaluate the function on the grid
        Z = np.array([self.compute(point) for point in XY]).reshape(X.shape)

        # Create a contour plot
        plt.figure(figsize=(30, 24.6))
        contour_plot = plt.contour(X, Y, Z, levels=20, cmap='viridis')  # Change levels and cmap as needed
        plt.colorbar(contour_plot, label='Function Values')
        plt.title('Contour Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()


    def plot_as_constraint(self):
        print('hello')





class UzawaPlotter(object):
    def __init__(self, solver: "UzawaSolver"):
        self.solver = solver

    # overall, there are many vizualization choice that we shoule take into consideration
    # the contour in the first place, 
    # whther to plot the function, the constraints...

    # this class should allow the different plottings possible for the uzawa solver

    
    def plot(self, plot_f=True, plot_constraints=True, plot_gradients=True):
        # there are different choices that we can make here
        # whether it is possible to plot f, constraints, gradients
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)


        # used_constraints = [(name, con) for name, con in self.constraints if name in constraint_list]
        
        constraint_list = self.solver.constraints
        
        # Stack the X and Y grids into a single 2D array
        XY = np.column_stack((X.flatten(), Y.flatten()))

        # Create a contour plot
        plt.figure(figsize=(30, 24.6))

        if plot_f:
            # Evaluate the function on the grid
            Z = np.array([self.solver.f.compute(point) for point in XY]).reshape(X.shape)
            # plot the contours of f
            contour_plot = plt.contour(X, Y, Z, levels=50, cmap='viridis')  # Change levels and cmap as needed
        

        if plot_constraints:
            
            constraints_values = [(con, np.array(
                [
                    con.__compute__(point) for 
                    point in XY]).reshape(X.shape)) for
                                                    con in constraint_list
                                                    ]
            # plot the contours of the constraints
            constraints_contours = [plt.contour(X, Y, constraint_values, levels=[0], cmap='viridis') for  
                                constraint, constraint_values in constraints_values]
            

            cmap = LinearSegmentedColormap.from_list('black_cmap', [(0, 0, 0), (0, 0, 0)], N=256)

            # plotting the non valid side of the constraints
            constraints_contours_invalid = [plt.contour(X, Y, constraint_values, levels=[0.05], 
                                                  cmap=cmap, linestyles='dashed', linewidths=10) for  
                                constraint, constraint_values in constraints_values]
        
        if plot_gradients:
            pass
            x_history = self.solver.x_history
            
            # Example data
            # x = [1, 2, 3, 4, 5]
            # y = [2, 4, 1, 3, 5]
            labels = [("x({})".format(str(i))) for i in range(len(x_history))]

            # Plot dots
            plt.scatter([x[0] for x in x_history], [x[1] for x in x_history], marker='x', color='red', s=200)

            # Add labels to each point
            for i, label in enumerate(labels):
                plt.text(x_history[i][0] - 0.05, x_history[i][1] + 0.1, label, ha='right', va='bottom', fontsize=18)

            # plot the gradients
            # gradients = self.solver.gradient

        plt.colorbar(contour_plot, label='Function Values')
        plt.title('Contour Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
        
    # we should also make a global function that plots the changes in the lambdas
    # we should track the changes in the increments as well

    def plot_lambda_history(self):
        # consider here
        norms = [np.linalg.norm(x) for x in self.solver.lambda_history]
        plt.stem(norms)

    def plot_f_increments(self, use_log=False):
        func = lambda x: np.log(1+x) if use_log else x
        norms = [func(np.linalg.norm(x)) for x in self.solver.f_increment_history]
        plt.stem(norms)
        plt.show()
    
    def plot_lagrangian_increments(self, use_log=False):
        func = lambda x: np.log(1+x) if use_log else x
        norms = [func(np.linalg.norm(x)) for x in self.solver.lagrangian_increment_history]
        plt.stem(norms)
        plt.show()

    def plot_tau_history(self):
        plt.stem(self.solver.tau_history)
        plt.show()

    
class GradientDescent(object):

    def __init__(self, f) -> None:
        self.f = f
    def continue_condition(self, current_iteration, max_iteration, current_norm, epsilon, use_epsilon):
        if use_epsilon:
            return current_norm > epsilon
        else:
            return current_iteration < max_iteration
        # return the minimal point

    def update_x(self, current_x, current_gradient, alpha):
    # current_x and current_gradient have the same length
        # print('current x: {}'.format(current_x))
        # print('its type: {}'.format(type(current_x)))
        current_x = current_x.reshape(-1)
        current_gradient = current_gradient.reshape(-1)
        delta = alpha * current_gradient
        new_x = current_x - delta
        return new_x.reshape(-1), delta.reshape(-1)  
    
    def solve(self, x0: np.array, max_iter=50, alpha=0.1, epsilon=0.01, use_epsilon=False):
        x_history = []
        gradient_history = []
        value_history = []
        lr_history = []

        current_x = x0
        current_value = self.f[current_x]
        current_gradient = self.f.__gradient__(current_x)
        current_alpha = alpha
        
        # the first setup
        x_history.append(current_x.tolist())
        value_history.append(current_value)
        gradient_history.append(current_gradient.tolist())
        lr_history.append(current_alpha)


        current_iteration = 1

        
        can_continue = self.continue_condition(current_iteration, max_iter, # iteration logic
                                                np.linalg.norm(current_gradient), epsilon, # epsilon logic
                                                use_epsilon=use_epsilon
                                                )
    
        # print("continue: {}".format(can_continue))
        # for i in range(max_iter):
        
        while can_continue:
            current_x, _ = self.update_x(
                current_x=current_x,
                current_gradient=current_gradient,
                alpha=current_alpha)
            
            current_gradient = self.f.__gradient__(current_x)
            current_value = self.f.__compute__(current_x)

            x_history.append(current_x.tolist())
            gradient_history.append(current_gradient.tolist())
            value_history.append(current_value)
            lr_history.append(current_alpha)

            current_iteration = current_iteration + 1
            
            # whether to continue to the next iteration or not

            can_continue = self.continue_condition(current_iteration, max_iter,
                                                    np.linalg.norm(current_gradient), epsilon, 
                                                    use_epsilon=use_epsilon)
        
            
        return np.array(x_history), np.array(gradient_history), np.array(lr_history), np.array(value_history), current_iteration        



# the end of the previous class is


    
    class MyFunction(object):
        def __init__(self, lambda_) -> None:
            self.lambda_ = lambda_
        
        def generate(self, f):
            return 


class UzawaSolver(object):
    def __init__(self, f: Function, constraints):
        
        # whether the optimzer is done optimizing before
        self.done = False
        # the parameters x and a history
        self.x_history = []
        self.lambda_history = []
        self.tau_history = []

        # the f value/gradient history
        self.f_gradient_history = []
        self.f_value_history = []
        
        # the lagrangian value and gradient history
        self.lagrangian_value_history = []
        self.lagrangian_gradient_history = []

        # f and the constraints
        self.f = f
        self.constraints = constraints


    def solve_min(self, x0_internal, _lambda: np.array, tau=0.01, alpha=0.01, 
                  max_iter=50, use_epsilon=False, epsilon=0.01, decay_type=None, decay_param=None):
        
        def continue_condition(current_iteration, max_iteration, current_norm, epsilon, use_epsilon):
            if use_epsilon:
                return current_norm > epsilon
            else:
                return current_iteration < max_iteration
        
        # SOME DECLARATIONS FOR THE DECAY
        # the decay param is a numeric value that is greater than 0
        # the decay type should be in ['linear', 'linear-param', 'quadratic', 'exponential']
        # if another value is specified out of this interval, we throw an error

        DECAY_TYPES = ['linear', 'linear-param', 'quadratic', 'exponential']


        def update_tau(tau, current_iteration, decay_type, decay_param):
            if decay_type is None:
                return tau 
            elif decay_type == 'linear':
                return tau / (current_iteration + 1)
            elif decay_type == 'linear-param':
                return tau / (decay_param * current_iteration + 1)
            elif decay_type == 'quadratic':
                return tau / (current_iteration + 1)**2
            elif decay_type == 'exponential':
                return tau * np.exp(-decay_param * current_iteration)
        
        
        # check decay type, assert the values are allowed
        if decay_type != None:
            # check for decay type validity
            if decay_type not in DECAY_TYPES:
                raise ValueError('decay_type {} not valid, you should enter a type in : {}'.format(decay_type, DECAY_TYPES))
            
            # check for decay param validity as well
            if decay_param <= 0:
                raise ValueError('decay param should be > 0')
            
        
            
        # history
        tau_history = []
        x_history = []
        f_value_history = []
        f_gradient_history = []

        lagrangian_value_history = []
        lagrangian_gradient_history = []

        # increment history
        f_increment_history = []
        lagrangian_increment_history = []

        # gradient_history = []
        lambda_history = []
        
        current_iteration = 1
        # we have been here
        
        # last gradient to compute the increments
        dim = x0_internal.shape[0]
        lagrangian_gradient_last = np.zeros(dim)
        f_gradient_last = np.zeros(dim)

        

        can_continue = True
        
        # for i in range(max_iter):
        while can_continue:
            # gradient descent on the lagrange function f to find x
            # X solver for the current lambda
            # f + lambda * constraints
            lagrangian = self.f
            for i in range(len(_lambda)):
                current_lambda = _lambda[i]
                lagrangian += current_lambda * self.constraints[i]
            

            x_solver = GradientDescent(lagrangian)
            x_history_local, gradient_history_local, \
            lr_history_local, value_history_local, last_iteration_local = x_solver.solve(x0=x0_internal)
            #         return np.array(x_history), np.array(gradient_history),
            #  np.array(lr_history), np.array(value_history), current_iteration        


            # add to the history of the variables 
            # we need the history of lambda, x, learning rate and all related stuff


            # last_x = x_history[-1]
            last_x = x_history_local[-1]
            
            # add the value of x and lambda to the history
            x_history.append(last_x)
            lambda_history.append(_lambda)
            
            # track value and gradient of f
            f_value_history.append(self.f.__compute__(last_x))
            f_gradient_history.append(self.f.gradient(last_x))
            
            # track value and gradient of lagrangian
            lagrangian_value_history.append(lagrangian.__compute__(last_x))
            lagrangian_gradient_history.append(lagrangian.__gradient__(last_x))

            # compute the next value for lambda
            for i in range(len(_lambda)):
                _lambda[i] = max(0, _lambda[i] + tau * self.constraints[i].__compute__(last_x))


            # save and update the tau 
            tau_history.append(tau)
            tau = update_tau(tau, current_iteration, 
                       decay_type=decay_type, decay_param=decay_param)
            
            # we compute the increments of f and the lagrangian
            # for that, we need the gradients
            f_current_gradient = self.f.__gradient__(last_x)
            lagrangian_current_gradient = lagrangian.__gradient__(last_x)

            # and then the increments
            f_increment = f_current_gradient - f_gradient_last
            lagrangian_increment = lagrangian_current_gradient - lagrangian_gradient_last
            
            f_increment_history.append(f_increment)
            lagrangian_increment_history.append(lagrangian_increment)    
        
            # update the last gradients
            lagrangian_gradient_last = lagrangian_current_gradient
            f_gradient_last = f_current_gradient

            can_continue = continue_condition(current_iteration, max_iter, # iteration logic
                                            np.linalg.norm(f_increment), epsilon, # epsilon logic
                                            use_epsilon=use_epsilon
                                            )

            current_iteration = current_iteration + 1

        # apply the current changed to entire solver
        # the solver should have the following attributes
        self.f_value_history = f_value_history
        self.f_gradient_history = f_gradient_history
        self.lagrangian_value_history = lagrangian_value_history
        self.lagrangian_gradient_history = lagrangian_gradient_history
        self.f_increment_history = f_increment_history
        self.lagrangian_increment_history = lagrangian_increment_history

        self.x_history = x_history
        self.lambda_history = lambda_history
        
        self.tau_history = tau_history
        # save the number of iterations in the last optimization execution
        self.iters = current_iteration - 1
        
            # we have found the new lambda
        #
        self.done = True
        return last_x