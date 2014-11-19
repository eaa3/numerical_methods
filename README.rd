#---------------------------------------------------------------------------------
# 
# ABOUT THE PROJECT
#
#---------------------------------------------------------------------------------

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

6) Adams-Bashforth[1,2,3,4] (error = h^2, h^3, h^4, h^5 - respectively ) 
7) Adams-Multon[1,2,3,4] (error = h^2, h^3, h^4, h^5 - respectively ) 
8) Preditor-Corrector[1,2,3,4] (error = h^2, h^3, h^4, h^5 - respectively ) 
9) Backward Differentiation (BackDiff[1,2,3,4]) (error = h^2, h^3, h^4, h^5 - respectively ) 




#---------------------------------------------------------------------------------
# 
# DEPENDENCIES OF THE PROGRAM
#
#---------------------------------------------------------------------------------

The code needs numpy, matplotlib and sympy to run

Installing numpy:

1) pip install numpy

If you can check the general installation guide on http://www.numpy.org

Installing matplotlib:

1) matplotlib is a plotting library. 
   To install it you can simply follow the instructions on the official website:

   http://matplotlib.org/users/installing.html

Installing Sympy:

1) To install sympy you can simply follow the instructions on the official website:
	
	http://docs.sympy.org/latest/install.html


#---------------------------------------------------------------------------------
# 
# HOW TO EXECUTE THE PROGRAM
#
#---------------------------------------------------------------------------------

You can execute the code using the following command on the shell:

1) python DSolver.py [OptionalInputFileName.txt]

If the [OptionalInputFileName.txt] is not provided, provided the default inputFile.txt will be used.

#---------------------------------------------------------------------------------
# 
# THE INPUT FILE
#
#---------------------------------------------------------------------------------

The input file sets the first order linear diferential equation to be solved, as well as the numerical method to be used and its parameters.
It should have the format bellow (See also the default inputFile.txt already provided with the project source)

If for some reason you dont the inputFile.txt, create it and copy and paste the content below:
# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
#--------------------------------------------------------------------------------- 
# THE INPUT FILE FORMAT (default inputFile.txt)
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# First order differential linear equation
# y' = g(x) - p(x)*y
#
# (!!) -> YOU MUST USE THE VARIABLES x and y (IN LOWERCASE),
# (!!) -> NO OTHER VARIABLE SYMBOL IS ALLOWED.
#---------------------------------------------------------------------------------
yd = "1-x + 4*y"

#---------------------------------------------------------------------------------
# Exact solution (if you dont have phi, make phi = None)
#---------------------------------------------------------------------------------
phi = (x/4) - (3/16) + exp(4*x)*(19/16)

#---------------------------------------------------------------------------------
# Initial value
#---------------------------------------------------------------------------------
y0 = 1

#---------------------------------------------------------------------------------
# Step size
#---------------------------------------------------------------------------------
h = 0.1

#---------------------------------------------------------------------------------
# Number of evaluations
#---------------------------------------------------------------------------------
n = 10

#---------------------------------------------------------------------------------
# Error Correction Threshold (only used for Predictor-Corrector method)
# typically 0 < episolon < 1, but episolon may be > 1
#---------------------------------------------------------------------------------
episolon = 0.3

#---------------------------------------------------------------------------------
# The method must be one of the following values:
#
# 1) "Euler"
#
# 2) "BackEuler"
#
# 3) "ImpEuler"
#
# 4) "RungeKutta"
#
# 5) "Taylor"
#
# 6) "Adams-Bashforth[X]" where [X] is one of the following values [1,2,3,4]
#    (e.g. Adams-Bashforth1 is adams-bashforth method of degree one)
#
# 7) "Adams-Multon[X]" where [X] is one of the following values [1,2,3,4]
#    (e.g. Adams-Multon1 is the adams-multon method of degree one)
#
# 8) "Predictor-Corrector[X]" where [X] is one of the following values [1,2,3,4]
#    (e.g. Preditor-Corrector1 is the Preditor-Corrector method of degree one)
#
# 9) "BackDiff[X]" where [X] is one of the following values [1,2,3,4]
#    (e.g. BackDiff1 is the Backward differentiation formula method of degree one)
#--------------------------------------------------------------------------------- 
Predictor-Corrector3

#--------------------------------------------------------------------------------- 
# END OF FILE
#---------------------------------------------------------------------------------



