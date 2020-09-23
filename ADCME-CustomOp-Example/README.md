This repo illustrates how to implement a custom operator in ADCME for an implicit operator using Physics Constrained Learning (PCL). 


**input**:

vector x

**output**

vector y

The forward operator is expressed implicitly with 

y^3 + y = x


To derive the gradient back-propagation, we use the following implicit function theorem:

3 y^2 y' + y' = 1 


y' = 1/(3y^2 + 1)
