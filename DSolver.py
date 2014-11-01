from sympy import *
import numpy as np
import matplotlib.pyplot as plt

class DSolver:

    def __init__(self, yd_expression_str, phi_expr_str = None):
        self.x_symb, self.y_symb, self.yd_symb = symbols("x y yd")
        self.yd_expr = sympify(yd_expression_str)
        
        self.yd_func = lambdify((self.x_symb,self.y_symb),self.yd_expr,"numpy")

        self.y = np.zeros
        self.phi = np.zeros
        self.error = np.zeros

        self.phi_func = None
        if( phi_expr_str != None ):
            self.phi_expr = sympify(phi_expr_str)
            self.phi_func = lambdify(self.x_symb,self.phi_expr)

    # Initialization
    def __initialize__(self,y0,n):
        
        self.y = np.zeros(n)
        self.phi = np.zeros(n)
        self.error = np.zeros(n)

        self.y[0] = y0
        self.error[0] = 0

        if self.phi_func != None:
            self.phi[0] = self.phi_func(0)
            self.error[0] = self.phi[0] - self.y[0]

    # Default Euler method
    def __solve_euler__(self,x0,y0,h,n):

        self.__initialize__(y0,n)

        for i in range(1,n):
            self.y[i] = self.y[i-1] + self.yd_func((i-1)*h,self.y[i-1])*h
            

            if self.phi_func != None:
                self.phi[i] = self.phi_func(i*h)
                self.error[i] = self.phi[i] - self.y[i]

    # Modified Euler method
    def __solve_improved_euler__(self,x0,y0,h,n):

        self.__initialize__(y0,n)

        for i in range(1,n):
            self.y[i] = self.y[i-1] + (self.yd_func((i-1)*h,self.y[i-1])+self.yd_func(i*h,self.y[i-1] + self.y[i-1]*h))*h*0.5
            

            if self.phi_func != None:
                self.phi[i] = self.phi_func(i*h)
                self.error[i] = self.phi[i] - self.y[i]

    # Backward Euler method
    def __solve_backward_euler__(self,x0,y0,h,n):
        self.__initialize__(y0,n)

        for i in range(1,n):
            self.y[i] = self.y[i-1] + self.yd_func(i*h,self.y[i-1] + self.y[i-1]*h)*h
            

            if self.phi_func != None:
                self.phi[i] = self.phi_func(i*h)
                self.error[i] = self.phi[i] - self.y[i]

    # Modified Euler method
    def __solve_runge_kutta__(self,x0,y0,h,n):

        self.__initialize__(y0,n)

        kn1 = kn2 = kn3 = kn4 = 0

        for i in range(1,n):

            kn1 = self.yd_func((i-1)*h,self.y[i-1])
            kn2 = self.yd_func(((i-1)+0.5)*h,self.y[i-1] + 0.5*h*kn1)
            kn3 = self.yd_func(((i-1)+0.5)*h,self.y[i-1] + 0.5*h*kn2)
            kn4 = self.yd_func(i*h,self.y[i-1] + h*kn3)

            self.y[i] = self.y[i-1] + h*(kn1 + 2*kn2 + 2*kn3 + kn4)/6
            

            if self.phi_func != None:
                self.phi[i] = self.phi_func(i*h)
                self.error[i] = self.phi[i] - self.y[i]

    def solve(self,x0,y0,h,n, method="Euler"):

        print "---------------------Solving for method: ", method, "---------------------"

        if( method == "Euler" ):
            self.__solve_euler__(x0,y0,h,n)
        elif (method == "BackEuler"):
            self.__solve_backward_euler__(x0,y0,h,n)
        elif (method == "ImpEuler"):
            self.__solve_improved_euler__(x0,y0,h,n)
        elif (method == "RungeKutta"):
            self.__solve_runge_kutta__(x0,y0,h,n)
        


# y' = g(x) - p(x)*y
yd_expression = "1-x + 4*y"

# phi(x): analytical solution for error comparisson
phi_expression = "(x/4) - (3/16) + exp(4*x)*(19/16)"




ds = DSolver(yd_expression, phi_expression)

ds.solve(0,1,0.1,7,"BackEuler")


print "Y: ", ds.y
print "Phi: ", ds.phi
print "Error: ", ds.error
