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
        self.n = n
        self.x = [x0]
        self.error = []
        self.accerror = []
        self.phi = []
        self.t = [0]

        self.episolon = 0.1 # Not used yet
        

        if self.phi_func != None:
            self.phi.append(self.phi_func(0))
            self.error.append(abs(self.phi[0] - self.x[0][0,0]))



    # Default Euler method

    def __euler__(self,h,i):

        xi = self.x[i-1] + self.xd_func(self.A,self.x[i-1],(i-1)*h)*h

        self.x.append( xi )

        return self.x[i]

    def __improved_euler__(self,h,i):

        xi = self.x[i-1] + h*(self.xd_func(self.A,self.x[i-1],(i-1)*h)+self.xd_func(self.A,self.x[i-1] + self.x[i-1]*h,i*h))*0.5
        
        self.x.append( xi )

        return self.x[i]


    # Backward Euler method

    def __backward_euler__(self,h,i):
        
        m, n = self.A.shape

        Minv = np.linalg.inv(np.eye(m,n) - self.A*h)

        xi = Minv*(self.x[i-1] + h*self.g_func(i*h))

        self.x.append( xi )

        return self.x[i]
    

    # Runge-Kutta method

    def __runge_kutta__(self,h,i):
        kn1 = kn2 = kn3 = kn4 = 0
        
        kn1 = self.xd_func(self.A,self.x[i-1],(i-1)*h)
        kn2 = self.xd_func(self.A,self.x[i-1] + 0.5*h*kn1,((i-1)+0.5)*h)
        kn3 = self.xd_func(self.A,self.x[i-1] + 0.5*h*kn2,((i-1)+0.5)*h)
        kn4 = self.xd_func(self.A,self.x[i-1] + h*kn3,i*h)

        xi = self.x[i-1] + h*(kn1 + 2*kn2 + 2*kn3 + kn4)/6

        self.x.append( xi )

        return self.x[i]

    # Three term Adams-Bashforth Series method

    def __adams_bashforth__(self,p, h,i):

        integral = 0

        if i <= p:
            self.__runge_kutta__(h,i)
        else:

            if p == 1:

                    xdn = self.xd_func(self.A,self.x[i-1],(i-1)*h)
                    xdn_1 = self.xd_func(self.A,self.x[i-2],(i-2)*h)

                    integral = (3*xdn*0.5 - xdn_1*0.5)*h

            elif p == 2:
                    xdn = self.xd_func(self.A,self.x[i-1],(i-1)*h)
                    xdn_1 = self.xd_func(self.A,self.x[i-2],(i-2)*h)
                    xdn_2 = self.xd_func(self.A,self.x[i-3],(i-3)*h)

                    integral = h*((23.0/12.0)*xdn - (4.0/3.0)*xdn_1 + (5.0/12.0)*xdn_2)

            elif p == 3:

                    xdn = self.xd_func(self.A,self.x[i-1],(i-1)*h)
                    xdn_1 = self.xd_func(self.A,self.x[i-2],(i-2)*h)
                    xdn_2 = self.xd_func(self.A,self.x[i-3],(i-3)*h)
                    xdn_3 = self.xd_func(self.A,self.x[i-4],(i-4)*h)

                    integral = h*((55.0*xdn - 59.0*xdn_1 + 37.0*xdn_2 - 9.0*xdn_3)/24.0)

            elif p == 4:
                    xdn = self.xd_func(self.A,self.x[i-1],(i-1)*h)
                    xdn_1 = self.xd_func(self.A,self.x[i-2],(i-2)*h)
                    xdn_2 = self.xd_func(self.A,self.x[i-3],(i-3)*h)
                    xdn_3 = self.xd_func(self.A,self.x[i-4],(i-4)*h)
                    xdn_4 = self.xd_func(self.A,self.x[i-5],(i-5)*h)

                    integral = h*((1901.0/720)*xdn - (1387.0/360)*xdn_1 + (109.0/30)*xdn_2 - (637.0/360)*xdn_3 + (251.0/720)*xdn_4)


            xi = self.x[i-1] + integral

            self.x.append( xi )

        return self.x[i]

        # Prediction correction
    def __backward_diff__(self,p, h,i):



        m, n = self.A.shape

        


        if i < p:
            self.__runge_kutta__(h,i)
            return self.x[i]


        xi = 0

        if p == 1:

            #This is equal to the BackEuler
            Minv = np.linalg.inv(np.eye(m,n) - self.A*h)

            xi = Minv*(self.x[i-1] + h*self.g_func(i*h))

        elif p == 2:

            Minv = np.linalg.inv(np.eye(m,n) - (2.0*self.A*h)/3)
            xi = Minv*((4.0/3.0)*self.x[i-1] - self.x[i-2]/3.0 + (2.0/3.0)*h*self.g_func(i*h))

        elif p == 3:

            Minv = np.linalg.inv(np.eye(m,n) - (6.0*self.A*h)/11.0)
            xi = Minv*((18.0/11.0)*self.x[i-1] - (9.0/11.0)*self.x[i-2] + (2.0/11.0)*self.x[i-3] + (6.0/11.0)*h*self.g_func(i*h))

        elif p == 4:
            Minv = np.linalg.inv(np.eye(m,n) - (12.0*self.A*h)/25.0)

            xi = Minv*((48.0/25.0)*self.x[i-1] - (36.0/25.0)*self.x[i-2] + (16.0/25.0)*self.x[i-3] - (3.0/25.0)*self.x[i-4] + (12.0/25.0)*h*self.g_func(i*h))


        self.x.append( xi )

        return self.x[i]

    # Three term Adams-Multon Series method

    def __adams_multon__(self,p, h,i):

        m, n = self.A.shape


        if i < p:
            self.__runge_kutta__(h,i)
            return self.x[i]


        xi = 0


        #if polinomial degree 1
        if p == 1:

                # TODO: Check correctness (Not used in this project)
                Minv = np.linalg.inv(np.eye(m,n) - self.A*h*0.5)
                xdn = self.xd_func(self.A,self.x[i-1],(i-1)*h)
                xi = self.x[i-1] + (self.g_func(i*h) + xdn)*0.5*h

        #if polinomial degree 2
        elif p == 2:

                Minv = np.linalg.inv(np.eye(m,n) - (5.0*self.A*h)/12.0)

                xdn = self.xd_func(self.A,self.x[i-1],(i-1)*h)
                xdn_1 = self.xd_func(self.A,self.x[i-2],(i-2)*h)

                xi = Minv*(self.x[i-1] + h*((5.0*self.g_func(i*h) + 8.0*xdn - xdn_1)/12.0))

        #if polinomial degree 3
        elif p == 3:

                Minv = np.linalg.inv(np.eye(m,n) - (3.0*self.A*h)/8.0)
                
                xdn = self.xd_func(self.A,self.x[i-1],(i-1)*h)
                xdn_1 = self.xd_func(self.A,self.x[i-2],(i-2)*h)
                xdn_2 = self.xd_func(self.A,self.x[i-3],(i-3)*h)

                xi = Minv*(self.x[i-1] + h*((9.0*self.g_func(i*h) + 19.0*xdn - 5.0*xdn_1 + xdn_2)/24.0))
        #if polinomial degree 4
        elif p == 4:
                Minv = np.linalg.inv(np.eye(m,n) - (251.0*self.A*h)/720.0)

                xdn  = self.xd_func(self.A,self.x[i-1],(i-1)*h)
                xdn_1 = self.xd_func(self.A,self.x[i-2],(i-2)*h)
                xdn_2 = self.xd_func(self.A,self.x[i-3],(i-3)*h)
                xdn_3 = self.xd_func(self.A,self.x[i-4],(i-4)*h)

                xi = Minv*(self.x[i-1] + ((251.0*self.g_func(i*h))/720.0 + (646.0*xdn)/720.0 - (264.0*xdn_1)/720.0 + (106.0*xdn_2)/720.0 - (19.0*xdn_3)/720.0)*h)


        self.x.append( xi )

        return self.x[i]

        # Prediction correction
    def __predictor_corrector__(self,p, h,i):

        integral = 0

        # Prediction step
        self.__adams_bashforth__(p,h,i)

        


        if i < p:
            return self.x[i]


        #Correction step

        #if polinomial degree 1
        if p == 1:

                xdn1 = self.xd_func(self.A,self.x[i],i*h)
                xdn = self.xd_func(self.A,self.x[i-1],(i-1)*h)

                integral = (xdn1 + xdn)*0.5*h

        #if polinomial degree 2
        elif p == 2:

                xdn1 = self.xd_func(self.A,self.x[i],i*h)
                xdn = self.xd_func(self.A,self.x[i-1],(i-1)*h)
                xdn_1 = self.xd_func(self.A,self.x[i-2],(i-2)*h)

                integral = h*((5*xdn1 + 8*xdn - xdn_1)/12.0)
        #if polinomial degree 3
        elif p == 3:

                xdn1 = self.xd_func(self.A,self.x[i],i*h)
                xdn = self.xd_func(self.A,self.x[i-1],(i-1)*h)
                xdn_1 = self.xd_func(self.A,self.x[i-2],(i-2)*h)
                xdn_2 = self.xd_func(self.A,self.x[i-3],(i-3)*h)

                integral = h*((9.0*xdn1 + 19.0*xdn - 5.0*xdn_1 + xdn_2)/24.0)
        #if polinomial degree 4
        elif p == 4:

                xdn1 = self.xd_func(self.A,self.x[i],i*h)
                xdn  = self.xd_func(self.A,self.x[i-1],(i-1)*h)
                xdn_1 = self.xd_func(self.A,self.x[i-2],(i-2)*h)
                xdn_2 = self.xd_func(self.A,self.x[i-3],(i-3)*h)
                xdn_3 = self.xd_func(self.A,self.x[i-4],(i-4)*h)

                integral = ((251.0*xdn1)/720.0 + (646.0*xdn)/720.0 - (264.0*xdn_1)/720.0 + (106.0*xdn_2)/720.0 - (19.0*xdn_3)/720.0)*h


        x_tmp = self.x[i-1] + integral


        # Error very high! -> Decrease H
        #if abs(x_tmp - self.x[i]) > self.episolon:
            #print "Error very high! -> Decrease H"
            #self.h *= 0.9

            #self.h *= 0.8

        self.x[i] = x_tmp

        return self.x[i]





    def __select_method__(self,method="Euler"):
        
        method_func = self.__euler__

        if( method == "Euler" ):
            method_func = self.__euler__
        elif (method == "BackEuler"):
            method_func = self.__backward_euler__
        elif (method == "ImpEuler"):
            method_func = self.__improved_euler__
        elif (method == "RungeKutta"):
            method_func = self.__runge_kutta__
        elif (method == "Adams-Bashforth1"):

            method_func = partial(self.__adams_bashforth__,1)

        elif (method == "Adams-Bashforth2"):

            method_func = partial(self.__adams_bashforth__,2)
        elif (method == "Adams-Bashforth3"):

            method_func = partial(self.__adams_bashforth__,3)
        elif (method == "Adams-Bashforth4"):

            method_func = partial(self.__adams_bashforth__,4)
        elif (method == "Adams-Multon1"):

            method_func = partial(self.__adams_multon__,1)

        elif (method == "Adams-Multon2"):

            method_func = partial(self.__adams_multon__,2)
        elif (method == "Adams-Multon3"):

            method_func = partial(self.__adams_multon__,3)
        elif (method == "Adams-Multon4"):

            method_func = partial(self.__adams_multon__,4)
        elif (method == "Predictor-Corrector1"):

            method_func = partial(self.__predictor_corrector__,1)

        elif (method == "Predictor-Corrector2"):

            method_func = partial(self.__predictor_corrector__,2)
        elif (method == "Predictor-Corrector3"):

            method_func = partial(self.__predictor_corrector__,3)
        elif (method == "Predictor-Corrector4"):

            method_func = partial(self.__predictor_corrector__,4)
        elif (method == "BackDiff1"):

            method_func = partial(self.__backward_diff__,1)

        elif (method == "BackDiff2"):

            method_func = partial(self.__backward_diff__,2)
        elif (method == "BackDiff3"):

            method_func = partial(self.__backward_diff__,3)
        elif (method == "BackDiff4"):

            method_func = partial(self.__backward_diff__,4)




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

        plt.subplot(2, 1, 1)
        plt.title(self.method + " (h = " + str(self.h) +")")
        p1, = plt.plot(self.t, mapped_x, 'b', linewidth=2, label='y')
        p2, = plt.plot(self.t, self.phi, color='#00ff00', linewidth=1, label='phi(x)')
        plt.legend( [p1, p2], ['y', 'phi(x)'] )

        ax = plt.subplot(2, 1, 2)
 

        p3, = plt.plot(self.t,self.error, 'r', linewidth=2, label='error')
        plt.legend( [p3], ['abs error'] )

        #verts = list(zip(self.x, abs(self.error)))

        #poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
        #ax.add_patch(poly)

        if invert_yaxis: 
            ax.invert_yaxis()
        plt.savefig("plots/"+self.method+ ".png")

        out_file = open("plots/reports/"+self.method+"_experiment_report.txt",'w')

        out_file.write("H: " + str(self.h) + " N: " + str(self.n) + "\n")
        mapped_x = map(lambda u: u[0,0], self.x)

        out_file.write("t\t\tY\t\tPhi\t\tError\n")
        for i in range(0,len(mapped_x)):

            out_file.write(str(self.t[i]) + "\t\t" + str(mapped_x[i]) + "\t\t"+ str(self.phi[i])  + "\t\t" + str(self.error[i]) + "\n" )
        

        out_file.close()
        plt.show()

    def plotExperiment(self, line_color = 'b', invert_yaxis = False ):
        #plt.clf()

        

        plt.subplot(2, 1, 1)
        plt.title(self.method)
        self.h = 0.1
        self.solve(self.x0,self.h,self.n,self.method)
        mapped_x = map(lambda u: u[0,0], self.x)
        print len(self.t), len(mapped_x)
        p1, = plt.plot(self.t, mapped_x, 'b', linewidth=1, label="y (h = " + str(0.1) +")")
        ax = plt.subplot(2, 1, 2)
        p51, = plt.plot(self.t,self.error, 'b', linewidth=2, label='error')


        plt.subplot(2, 1, 1)
        self.h = 0.05
        self.solve(self.x0,self.h,self.n*2,self.method)
        mapped_x = map(lambda u: u[0,0], self.x)
        p2, = plt.plot(self.t, mapped_x, 'c', linewidth=1, label="y (h = " + str(0.05) +")")
        ax = plt.subplot(2, 1, 2)
        p52, = plt.plot(self.t,self.error, 'c', linewidth=2, label='error')


        plt.subplot(2, 1, 1)
        self.h = 0.025
        self.solve(self.x0,self.h,self.n*4,self.method)
        mapped_x = map(lambda u: u[0,0], self.x)
        p3, = plt.plot(self.t, mapped_x, 'm', linewidth=1, label="y (h = " + str(0.025) +")")
        ax = plt.subplot(2, 1, 2)
        p53, = plt.plot(self.t,self.error, 'm', linewidth=2, label='error')
        plt.legend( [p51, p52, p53], ['abs error (h = 0.1)','abs error (h = 0.05)', 'abs error (h = 0.025)'], loc = 0 )


        plt.subplot(2, 1, 1)
        p4, = plt.plot(self.t, self.phi, 'g', linewidth=2, label='phi(x)')
        plt.legend( [p1, p2,p3, p4], ["y (h = 0.1)","y (h = 0.05)","y (h = 0.025)", 'phi(x)'], loc=0 )


        #verts = list(zip(self.x, abs(self.error)))

        #poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
        #ax.add_patch(poly)

        if invert_yaxis: 
            ax.invert_yaxis()
        #plt.show()
        plt.savefig("plots/"+self.method+ ".png")

def mainExperiment(argv=None):

    starting_methods = ["Euler", "BackEuler", "ImpEuler", "RungeKutta"]
    multistep_methods = ["Adams-Bashforth", "Adams-Multon","Predictor-Corrector","BackDiff"]
    colors = ['b','m','c']
    hs = [0.1, 0.05, 0.025]
    phi_expression_str = "(-2.0/5)*cos(t) - (4.0/5)*sin(t) + (77.0/65)*cos(2*t) - (49.0/130)*sin(2*t) + (81.0/40)*exp(t) + (1.0/20)*exp(-t) + (73.0/520)*exp(-3*t)"#"-(2.0/5)*cos(t) - (4.0/5)*sin(t) + (77.0/65)*cos(2*t) - (49.0/65)*sin(2*t) + (81.0/40)*exp(t)+ (1.0/20)*exp(-t) + (73.0/520)*exp(-3*t)"
    g_expressions_str = ["0", "0", "0", "12*sin(t) - exp(-t)"] # Cada bixo eh uma expressao
    x0 = np.mat("3; 0; -1; 2")
    A = np.mat("0 1 0 0; 0 0 1 0; 0 0 0 1; 12 -8 -1 -2")
    n = 100
    h = 0.01
    method = "Euler"

    #yd_expression,phi_expression,y0,n,h,episolon,method = readInput(argv[1] if len(argv)>1 else "inputFile.txt")
    
    ds = DVSolver(A,x0,g_expressions_str,phi_expression_str)

    out_file = open("experiment_report.txt",'w')

    
    for method in starting_methods:
        plt.clf()
        out_file.write("----------------------------------------------------------------\n")
        out_file.write("METHOD: " + method + "\n")


        x = y = phi = error = accerror = []
        for j in range(0,len(hs)):
            h = hs[j]
            ds.solve(x0,h,n,method)
            #out_file.write("H: " + str(h) + " N: " + str(int((1.0/h))) + "\n")
            #out_file.write("X: " + str(ds.x) + "\n")
            #out_file.write("Y: " + str(ds.y) + "\n")
            #out_file.write("Phi: " + str(ds.phi)+ "\n")
            #out_file.write("Error: " + str(ds.error)+ "\n")
            #out_file.write("Acumulated Error: " + str(sum(abs(ds.error))) + "\n")

            


        ds.plotExperiment(colors[j])

    for k in range(0,len(multistep_methods)):
        for i in range(1,5):
            plt.clf()
            method = multistep_methods[k] + str(i)
            out_file.write("METHOD: " + method + "\n")
            for j in range(0,len(hs)):
                h = hs[j]
                ds.solve(x0,h,n,method)
                #out_file.write("H: " + str(h) + " N: " + str(int((1.0/h))) + "\n")
                #out_file.write("X: " + str(ds.x) + "\n")
                #out_file.write("Y: " + str(ds.y) + "\n")
                #out_file.write("Phi: " + str(ds.phi)+ "\n")
                #out_file.write("Error: " + str(ds.error)+ "\n")
                #out_file.write("Acumulated Error: " + str(sum(abs(ds.error)))+ "\n")


            ds.plotExperiment(colors[j])
            #out_file.write('\n')


        #out_file.write('\n')
    #out_file.close()




    #print "Y: ", ds.y
    #print "Phi: ", ds.phi
    #print "Error: ", ds.error
    #print "Acumulated Error: ", sum(abs(ds.error))

    #ds.plot()

    out_file.close()




def readInput(filename):
    return None


def main(argv=None):
    # phi expressionOK (Variacao de parametros): "(-2.0/5)*cos(t) - (4.0/5)*sin(t) + (77.0/65)*cos(2*t) - (49.0/130)*sin(2*t) + (81.0/40)*exp(t) + (1.0/20)*exp(-t) + (73.0/520)*exp(-3*t)"
    phi_expression_str = "(-2.0/5)*cos(t) - (4.0/5)*sin(t) + (77.0/65)*cos(2*t) - (49.0/130)*sin(2*t) + (81.0/40)*exp(t) + (1.0/20)*exp(-t) + (73.0/520)*exp(-3*t)"#"-(2.0/5)*cos(t) - (4.0/5)*sin(t) + (77.0/65)*cos(2*t) - (49.0/65)*sin(2*t) + (81.0/40)*exp(t)+ (1.0/20)*exp(-t) + (73.0/520)*exp(-3*t)"
    g_expressions_str = ["0", "0", "0", "12*sin(t) - exp(-t)"] # Cada bixo eh uma expressao
    x0 = np.mat("3; 0; -1; 2")
    A = np.mat("0 1 0 0; 0 0 1 0; 0 0 0 1; 12 -8 -1 -2")
    n = 100
    h = 0.01
    method = "BackDiff4"

    ds = DVSolver(A,x0,g_expressions_str,phi_expression_str)

    ds.solve(x0,h,n,method)

    #for i in range(0,n):
    #    print "[", i, "] y = ", ds.x[i][0], " phi = ", ds.phi[i], " error = ", ds.error[i]
    ds.plot()


if __name__ == "__main__":
    sys.exit(main(sys.argv))

