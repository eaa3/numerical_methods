from sympy import *
from functools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import subprocess
import copy
import math

import sys



"""
Implementation of Numerical Methods
Center for Informatics, Federal University of Pernambuco (CIn/UFPE)


@author: Ermano A. Arruda <eaa3@cin.ufpe.br>



Implemented methods:

----One-step/Stepwise/Starting Methods----

1) Euler (error ~ h^2)
2) Improved Euler Method (Modified Euler) (error ~ h^3)
3) Backward Euler Method (error ~ h^2)
4) Runge-Kutta Method (error ~ h^4)
5) Three Term Taylor Series Method (error ~ h^3)


----Multistep or Continuing Methods----

6) Adams-Bashforth[1,2,3,4] (error = h^2, h^3, h^4, h^5 - respectively ) [TODO: Check error correctness]
7) Adams-Multon[1,2,3,4] (error = h^2, h^3, h^4, h^5 - respectively ) [TODO: Check error correctness]
8) Preditor-Corrector[1,2,3,4] (error = h^2, h^3, h^4, h^5 - respectively ) [TODO: Check error correctness]
9) Backward Differentiation (BackDiff[1,2,3,4]) (error = h^2, h^3, h^4, h^5 - respectively ) [TODO: Check error correctness]




"""

class DVSolver:


    def __init__(self, A, x0, g_expressions_str,phi_expr_str = None):

        self.A = A
        self.x0 = x0
        self.t = [0]

        self.A_symb, self.x_symb, self.g_symb, self.t_symb = symbols("A x g t")

        self.phi_func = None

        if phi_expr_str != None:
            self.phi_expr = sympify(phi_expr_str)

            # Initializing phi func
            self.phi_func = lambdify(self.t_symb,self.phi_expr)



        # Initializing vector function g
        g_expressions = []
        for g_expr_str in g_expressions_str:
            g_expressions.append( [sympify(g_expr_str)] )

        g_expression = Matrix(g_expressions)

        # Initializing expressiong for xd
        self.xd_expr = sympify("A*x + g(t)")       

        # Defining lambda vector function g
        self.g_func = lambdify(self.t_symb, g_expression)

        # Defining xd (derivative of x) in matricial form xd = A*x + g(t) (where xd and x are vectors, and g is a vector function)
        self.xd_func = lambdify((self.A_symb, self.x_symb, self.t_symb), self.xd_expr, {"g": self.g_func})





    # Initialization
    def __initialize__(self,x0,n):
        # Initializing data recording structures 
        self.x = [x0]
        self.error = []
        self.accerror = []
        self.phi = [0]

        if self.phi_func != None:
            self.phi.append(self.phi_func(0))
            self.error.append(abs(self.phi[0] - self.x[0][0,0]))



    # Default Euler method

    def __euler__(self,h,i):

        xi = self.x[i-1] + self.xd_func(self.A,self.x[i-1],(i-1)*h)*h

        self.x.append( xi )

        return self.x[i]



    def __select_method__(self,method="Euler"):
        
        method_func = self.__euler__

        if( method == "Euler" ):
            method_func = self.__euler__




        return method_func

    def __solve__(self,x0,h,n, method_func):

        self.h = h
        self.__initialize__(x0,n)

        for i in range(1,n+1):

            self.t.append( self.t[i-1] + self.h )
            method_func(self.h,i)
            

            if self.phi_func != None:
                self.phi.append(self.phi_func(self.h*i))
                self.error.append(abs(self.phi[i] - self.x[i][0,0]))
                self.accerror.append(abs(self.error[i]) + abs(self.error[i-1]))

    def solve(self,x0,h,n, method="Euler"):

        print "------------ Solving [ yd =",self.xd_expr,"] for method:", method, "-------------"

        self.method = method
        method_func = self.__select_method__(method)


        self.__solve__(x0,h,n,method_func)

    def plot(self,invert_yaxis = False ):


        mapped_x = map(lambda u: u[0,0], self.x)

        print mapped_x

        print len(mapped_x), len(self.t), len(self.phi)

        plt.subplot(2, 1, 1)
        plt.title(self.method + " (h = " + str(self.h) +")")
        p1, = plt.plot(self.t, mapped_x, 'b', linewidth=1, label='y')
        p2, = plt.plot(self.t, self.phi[0:len(self.phi)-1], 'g', linewidth=1, label='phi(x)')
        plt.legend( [p1, p2], ['y', 'phi(x)'] )

        ax = plt.subplot(2, 1, 2)
 

        p3, = plt.plot(self.t,self.error, 'r', linewidth=2, label='error')
        plt.legend( [p3], ['abs error'] )

        #verts = list(zip(self.x, abs(self.error)))

        #poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
        #ax.add_patch(poly)

        if invert_yaxis: 
            ax.invert_yaxis()
        plt.show()




def readInput(filename):
    return None


def main(argv=None):
    # phi expressionOK (Variacao de parametros): "(-2.0/5)*cos(t) - (4.0/5)*sin(t) + (77.0/65)*cos(2*t) - (49.0/130)*sin(2*t) + (81.0/40)*exp(t) + (1.0/20)*exp(-t) + (73.0/520)*exp(-3*t)"
    phi_expression_str = "(-2.0/5)*cos(t) - (4.0/5)*sin(t) + (77.0/65)*cos(2*t) - (49.0/130)*sin(2*t) + (81.0/40)*exp(t) + (1.0/20)*exp(-t) + (73.0/520)*exp(-3*t)"#"-(2.0/5)*cos(t) - (4.0/5)*sin(t) + (77.0/65)*cos(2*t) - (49.0/65)*sin(2*t) + (81.0/40)*exp(t)+ (1.0/20)*exp(-t) + (73.0/520)*exp(-3*t)"
    g_expressions_str = ["0", "0", "0", "12*sin(t) - exp(-t)"] # Cada bixo eh uma expressao
    x0 = np.mat("3; 0; -1; 2")
    A = np.mat("0 1 0 0; 0 0 1 0; 0 0 0 1; 12 -8 -1 -2")
    n = 10
    h = 0.1
    method = "Euler"

    ds = DVSolver(A,x0,g_expressions_str,phi_expression_str)

    ds.solve(x0,h,n)

    #for i in range(0,n):
    #    print "[", i, "] y = ", ds.x[i][0], " phi = ", ds.phi[i], " error = ", ds.error[i]
    ds.plot()


if __name__ == "__main__":
    sys.exit(main(sys.argv))

