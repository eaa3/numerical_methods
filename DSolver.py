from sympy import *
from functools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import subprocess



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
8) Prevision-Correction[1,2,3,4] (error = h^2, h^3, h^4, h^5 - respectively ) [TODO: Check error correctness]
10) Backward Differentiation (BackDiff[1,2,3,4]) (error = h^2, h^3, h^4, h^5 - respectively ) [TODO: Check error correctness]




"""

class DSolver:

    def __init__(self, yd_expression_str, phi_expr_str = None):
        self.x_symb, self.y_symb, self.yd_symb = symbols("x y yd")
        self.yd_expr = sympify(yd_expression_str)
        
        self.yd_func = lambdify((self.x_symb,self.y_symb),self.yd_expr,"numpy")

        self.x = np.zeros
        self.y = np.zeros
        self.yd = np.zeros
        self.phi = np.zeros
        self.error = np.zeros

        self.iyd = 0

        self.phi_func = None
        if( phi_expr_str != None ):
            self.phi_expr = sympify(phi_expr_str)
            self.phi_func = lambdify(self.x_symb,self.phi_expr)

    # Initialization
    def __initialize__(self,x0,y0,n):
        
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.yd = np.zeros(n)
        self.phi = np.zeros(n)
        self.error = np.zeros(n)

        self.iyd = 0

        self.x[0] = x0
        self.y[0] = y0
        self.error[0] = 0

        if self.phi_func != None:
            self.phi[0] = self.phi_func(0)
            self.error[0] = self.phi[0] - self.y[0]

    # Yd (derivative of y) -- Unused
    def __yd__(self,h,i):
        if self.iyd < i:
            self.iyd = i
            self.yd[i] = self.yd_func(i*h,self.y[i])*h
        
        return self.yd[i]


    # Default Euler method

    def __euler_integral__(self,h,i):
        self.y[i] = self.y[i-1] + self.yd_func((i-1)*h,self.y[i-1])*h
        return self.y[i]

    # Modified Euler method

    def __improved_euler_integral__(self,h,i):
        self.y[i] = self.y[i-1] + (self.yd_func((i-1)*h,self.y[i-1])+self.yd_func(i*h,self.y[i-1] + self.y[i-1]*h))*h*0.5
        return self.y[i]


    # Backward Euler method

    def __backward_euler_integral__(self,h,i):
        
        yi = symbols("yi")
   
        exp = self.y[i-1] + self.yd_expr.subs({self.y_symb:yi, self.x_symb:i*h})*h
        equ = Eq(exp,yi)
        
        self.y[i] = solve(equ,yi)[0]

        return self.y[i]
    

    # Runge-Kutta method

    def __runge_kutta_integral__(self,h,i):
        kn1 = kn2 = kn3 = kn4 = 0
        
        kn1 = self.yd_func((i-1)*h,self.y[i-1])
        kn2 = self.yd_func(((i-1)+0.5)*h,self.y[i-1] + 0.5*h*kn1)
        kn3 = self.yd_func(((i-1)+0.5)*h,self.y[i-1] + 0.5*h*kn2)
        kn4 = self.yd_func(i*h,self.y[i-1] + h*kn3)

        self.y[i] = self.y[i-1] + h*(kn1 + 2*kn2 + 2*kn3 + kn4)/6

        return self.y[i]

    
    # Three term Taylor Series method

    def __taylor_series_integral__(self,f_x,f_y,h,i):


        yd = self.yd_func((i-1)*h,self.y[i-1])
        
        ydd = f_x((i-1)*h, yd) + f_y((i-1)*h, yd)*yd

        self.y[i] = self.y[i-1] + h*yd + ((h**2)/2)*ydd

        return self.y[i]


    # Three term Adams-Bashforth Series method

    def __ab_integral__(self,p, h,i):

        integral = 0

        if i <= p:
            self.__runge_kutta_integral__(h,i)
        else:

            if p == 1:

                    ydn = self.yd_func((i-1)*h,self.y[i-1])
                    ydn_1 = self.yd_func((i-2)*h,self.y[i-2])

                    integral = (3*ydn*0.5 - ydn_1*0.5)*h

            elif p == 2:
                    ydn = self.yd_func((i-1)*h,self.y[i-1])
                    ydn_1 = self.yd_func((i-2)*h,self.y[i-2])
                    ydn_2 = self.yd_func((i-3)*h,self.y[i-3])

                    integral = h*((23.0/12.0)*ydn - (4.0/3.0)*ydn_1 + (5.0/12.0)*ydn_2)

            elif p == 3:

                    ydn = self.yd_func((i-1)*h,self.y[i-1])
                    ydn_1 = self.yd_func((i-2)*h,self.y[i-2])
                    ydn_2 = self.yd_func((i-3)*h,self.y[i-3])
                    ydn_3 = self.yd_func((i-4)*h,self.y[i-4])

                    integral = h*((55*ydn - 59*ydn_1 + 37*ydn_2 - 9*ydn_3)/24.0)

            elif p == 4:
                    ydn = self.yd_func((i-1)*h,self.y[i-1])
                    ydn_1 = self.yd_func((i-2)*h,self.y[i-2])
                    ydn_2 = self.yd_func((i-3)*h,self.y[i-3])
                    ydn_3 = self.yd_func((i-4)*h,self.y[i-4])
                    ydn_4 = self.yd_func((i-5)*h,self.y[i-5])

                    integral = h*((1901/720)*ydn - (1387/360)*ydn_1 + (109/30)*ydn_2 - (637/360)*ydn_3 + (251/720)*ydn_4)


            self.y[i] = self.y[i-1] + integral

        return self.y[i]

    # Three term Adams-Multon Series method

    def __am_integral__(self,p, h,i):

        integral = 0


        if i < p:
            self.__runge_kutta_integral__(h,i)
            return self.y[i]



        yi = symbols("yi")

        ydn1 = self.yd_expr.subs({self.y_symb:yi, self.x_symb:i*h})


        #if polinomial degree 1
        if p == 1:

                ydn = self.yd_func((i-1)*h,self.y[i-1])

                integral = (ydn1 + ydn)*0.5*h

        #if polinomial degree 2
        elif p == 2:

                ydn = self.yd_func((i-1)*h,self.y[i-1])
                ydn_1 = self.yd_func((i-2)*h,self.y[i-2])

                integral = h*((5.0*ydn1 + 8.0*ydn - ydn_1)/12.0)
        #if polinomial degree 3
        elif p == 3:

                ydn = self.yd_func((i-1)*h,self.y[i-1])
                ydn_1 = self.yd_func((i-2)*h,self.y[i-2])
                ydn_2 = self.yd_func((i-3)*h,self.y[i-3])

                integral = h*((9.0*ydn1 + 19.0*ydn - 5.0*ydn_1 + ydn_2)/24.0)
        #if polinomial degree 4
        elif p == 4:

                ydn  = self.yd_func((i-1)*h,self.y[i-1])
                ydn_1 = self.yd_func((i-2)*h,self.y[i-2])
                ydn_2 = self.yd_func((i-3)*h,self.y[i-3])
                ydn_3 = self.yd_func((i-4)*h,self.y[i-4])

                integral = ((251.0*ydn1)/720.0 + (646.0*ydn)/720.0 - (264.0*ydn_1)/720.0 + (106.0*ydn_2)/720.0 - (19.0*ydn_3)/720.0)*h


        exp = self.y[i-1] + integral
        equ = Eq(exp,yi)

        self.y[i] = solve(equ,yi)[0]

        return self.y[i]


    # Prediction correction
    def __pc_integral__(self,p, h,i):

        integral = 0

        # Prediction step
        self.__ab_integral__(p,h,i)

        


        if i < p:
            return self.y[i]


        #Correction step

        #if polinomial degree 1
        if p == 1:

                ydn1 = self.yd_func(i*h,self.y[i])
                ydn = self.yd_func((i-1)*h,self.y[i-1])

                integral = (ydn1 + ydn)*0.5*h

        #if polinomial degree 2
        elif p == 2:

                ydn1 = self.yd_func(i*h,self.y[i])
                ydn = self.yd_func((i-1)*h,self.y[i-1])
                ydn_1 = self.yd_func((i-2)*h,self.y[i-2])

                integral = h*((5*ydn1 + 8*ydn - ydn_1)/12.0)
        #if polinomial degree 3
        elif p == 3:

                ydn1 = self.yd_func(i*h,self.y[i])
                ydn = self.yd_func((i-1)*h,self.y[i-1])
                ydn_1 = self.yd_func((i-2)*h,self.y[i-2])
                ydn_2 = self.yd_func((i-3)*h,self.y[i-3])

                integral = h*((9.0*ydn1 + 19.0*ydn - 5.0*ydn_1 + ydn_2)/24.0)
        #if polinomial degree 4
        elif p == 4:

                ydn1 = self.yd_func(i*h,self.y[i])
                ydn  = self.yd_func((i-1)*h,self.y[i-1])
                ydn_1 = self.yd_func((i-2)*h,self.y[i-2])
                ydn_2 = self.yd_func((i-3)*h,self.y[i-3])
                ydn_3 = self.yd_func((i-4)*h,self.y[i-4])

                integral = ((251.0*ydn1)/720.0 + (646.0*ydn)/720.0 - (264.0*ydn_1)/720.0 + (106.0*ydn_2)/720.0 - (19.0*ydn_3)/720.0)*h


        y_tmp = self.y[i-1] + integral


        if abs(y_tmp - self.y[i]) > 0.001:
            print "Error very high! -> Decrease H"

        self.y[i] = y_tmp

        return self.y[i]

    # Prediction correction
    def __backward_diff_integral__(self,p, h,i):


        y_tmp = 0


        if i < p:
            self.__runge_kutta_integral__(h,i)
            return self.y[i]



        yi = symbols("yi")

        ydn1 = self.yd_expr.subs({self.y_symb:yi, self.x_symb:(i*h)})

        if p == 1:
            y_tmp = self.y[i-1] + h*ydn1
        elif p == 2:


            y_tmp = ((4.0/3.0)*self.y[i-1] - self.y[i-2]/3.0 + (2.0/3.0)*h*ydn1)

        elif p == 4:

            y_tmp = ((48.0/25.0)*self.y[i-1] - (36.0/25.0)*self.y[i-2] + (16.0/25.0)*self.y[i-3] - (3.0/25.0)*self.y[i-4] + (12.0/25.0)*h*ydn1)

        exp = y_tmp
        equ = Eq(exp,yi)

        self.y[i] = solve(equ,yi)[0]

        return self.y[i]




    def __select_method__(self,method="Euler"):
        
        integral_func = self.__euler_integral__

        if( method == "Euler" ):
            integral_func = self.__euler_integral__
        elif (method == "BackEuler"):
            integral_func = self.__backward_euler_integral__
        elif (method == "ImpEuler"):
            integral_func = self.__improved_euler_integral__
        elif (method == "RungeKutta"):
            integral_func = self.__runge_kutta_integral__
        elif (method == "Taylor"):

             # Partial derivative of yd w.r.t to x
            f_x = lambdify((self.x_symb,self.y_symb),diff(self.yd_expr,self.x_symb),"numpy")
            # Partial derivative of yd w.r.t to y
            f_y = lambdify((self.x_symb,self.y_symb),diff(self.yd_expr,self.y_symb),"numpy")

            integral_func = partial(self.__taylor_series_integral__,f_x,f_y)

        elif (method == "Adams-Bashforth1"):

            integral_func = partial(self.__ab_integral__,1)

        elif (method == "Adams-Bashforth2"):

            integral_func = partial(self.__ab_integral__,2)
        elif (method == "Adams-Bashforth3"):

            integral_func = partial(self.__ab_integral__,3)
        elif (method == "Adams-Bashforth4"):

            integral_func = partial(self.__ab_integral__,4)

        elif (method == "Adams-Multon1"):

            integral_func = partial(self.__am_integral__,1)

        elif (method == "Adams-Multon2"):

            integral_func = partial(self.__am_integral__,2)
        elif (method == "Adams-Multon3"):

            integral_func = partial(self.__am_integral__,3)
        elif (method == "Adams-Multon4"):

            integral_func = partial(self.__am_integral__,4)

        elif (method == "Prediction-Correction1"):

            integral_func = partial(self.__pc_integral__,1)

        elif (method == "Prediction-Correction2"):

            integral_func = partial(self.__pc_integral__,2)
        elif (method == "Prediction-Correction3"):

            integral_func = partial(self.__pc_integral__,3)
        elif (method == "Prediction-Correction4"):

            integral_func = partial(self.__pc_integral__,4)

        elif (method == "BackDiff1"):

            integral_func = partial(self.__backward_diff_integral__,1)

        elif (method == "BackDiff2"):

            integral_func = partial(self.__backward_diff_integral__,2)
        elif (method == "BackDiff3"):
            print "TODO: BackDiff3"
            #integral_func = partial(self.__pc_integral__,3) # TODO: Change to backward_diff
        elif (method == "BackDiff4"):

            integral_func = partial(self.__backward_diff_integral__,4)




        return integral_func

    def __solve__(self,x0,y0,h,n, integral_func):

        self.h = h
        self.__initialize__(x0,y0,n)

        for i in range(1,n):

            self.x[i] = self.x[i-1] + self.h
            integral_func(self.h,i)
            

            if self.phi_func != None:
                self.phi[i] = self.phi_func(self.h*i)
                self.error[i] = self.phi[i] - self.y[i]

    def solve(self,x0,y0,h,n, method="Euler"):

        print "---------------------Solving for method: ", method, "---------------------"

        self.method = method
        integral_func = self.__select_method__(method)


        self.__solve__(x0,y0,h,n,integral_func)


    def plot(self,invert_yaxis = False ):
        plt.subplot(2, 1, 1)
        plt.title(self.method + " (h = " + str(self.h) +")")
        p1, = plt.plot(self.x, self.y, 'b', linewidth=1, label='y')
        p2, = plt.plot(self.x, self.phi, 'g', linewidth=1, label='phi(x)')
        plt.legend( [p1, p2], ['y', 'phi(x)'] )

        ax = plt.subplot(2, 1, 2)
        p3, = plt.plot(self.x,abs(self.error), 'r', linewidth=2, label='error')
        plt.legend( [p3], ['abs error'] )

        #verts = list(zip(self.x, abs(self.error)))

        #poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
        #ax.add_patch(poly)

        if invert_yaxis: 
            ax.invert_yaxis()
        plt.show()


        


# y' = g(x) - p(x)*y
yd_expression = "1-x + 4*y"

# phi(x): analytical solution for error comparisson
phi_expression = "(x/4) - (3/16) + exp(4*x)*(19/16)"




ds = DSolver(yd_expression, phi_expression)

ds.solve(0,1,0.1,10,"BackDiff1")


print "Y: ", ds.y
print "Phi: ", ds.phi
print "Error: ", ds.error
print "Acumulated Error: ", sum(abs(ds.error))

ds.plot()


#proc = subprocess.Popen(['gnuplot','-p'], 
#                        shell=True,
#                        stdin=subprocess.PIPE,
#                        )
#proc.stdin.write('set xrange [0:10]; set yrange [-2:2]\n')
#proc.stdin.write('plot sin(x)\n')
