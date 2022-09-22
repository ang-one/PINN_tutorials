"""
TEST PARTIAL DIFFERENTIAL EQUATION resoltuion using PINN
Dynamics from: Non-equilibrium theory of the allele frequency spectrum
Theoretical Population Biology 71 (2007) 109–119. Using an exponetial growing pop.
as an example
"""

# loading packages
#Packages for NN
using NeuralPDE
using ModelingToolkit
using Optimization
using DiffEqFlux
using Lux
using OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum
# Packages for plots

using Plots
using DelimitedFiles

@parameters t,x # parameters of the problems. here declare both variables and parameters to be inferred
@variables g(..)  [bounds = (0.0,Inf)] # Unknow function/s of the problem
"symbolyc functions"
Dt = Differential(t)
Dxx = Differential(x) ^ 2
Dx = Differential(x)

"constant parameters"
s = 2 # Fisher selection coefficient
α = 1.0 # growth rate of the population
max_t = 10.0 # maximum time of the simulation

"Symbolic PDE that we want to solve. Note that the Allele Freq. Spectrum
(AFS) is defined as f(x,t) = g(x,t) * ( x * ( 1 - x ))"
eqs = Dt(g(t,x)) ~  - 1 * s * x * (1 - x ) * Dx(g(t,x)) + ((x *(1-x)) /( 2 * exp(α * t)) ) *  Dxx(g(t,x))

"Imposing intial condition"

##  points where the inital condition is evaluated
array_of_x = [0.0:0.01:1;]

#### EQUILIBRIUM BCS
if s > 0
    bcs = [g(0,array_of_x[k]) ~ (exp(2 * s) * (1 - exp( - 2 * s * (1 - array_of_x[k] )))) / (exp(2 * s) - 1) for k in 1:length(array_of_x) ]

end

if s == 0
    bcs = [g(0,array_of_x[k]) ~ 1-array_of_x[k] for k in 1:length(array_of_x) ]

end


# VF and time domains
domains = [t ∈ Interval(0.0,max_t) , x ∈ Interval(0.0,1.0)]

" setting symbolic problem"

@named pde_system = PDESystem(eqs,bcs,domains,[t,x],[g(t,x)])

"Architecture of the NN"
# Neural network
## Size of input layer MUST have the same dimension of the problem
input_  = length(domains)
# defining the NN
chain = Lux.Chain(Lux.Dense(input_,16,Lux.σ),Lux.Dense(16,16,Lux.σ),Lux.Dense(16,1))

# strategy of training, here I use the simplest case where the integration mesh
# is fixed
dx = 0.05 # size of mesh step
_strategy = GridTraining(dx)

# definig the Architecture combined with the training strategy
discretization = PhysicsInformedNN(chain, _strategy)

# combining the Architecture with the mathematical problem . I.e. function to
# transform the integration problem as a symbolic optimization
prob = discretize(pde_system , discretization)

##


### callback function used during the optimization
callback = function (p,l)
    println("Current loss is: $l")
    return false
end


# selecting optimizer
# starting optimization with ADAM
res = Optimization.solve(prob, Adam(10^-3); callback = callback, maxiters=300)
# second  optimization with BFGS
prob = remake(prob, u0 = res.u)
opt = OptimizationOptimJL.BFGS()
res = solve(prob,opt; callback = callback, maxiters=500)

# taking solutions

phi = discretization.phi # sybolic
minimizers_g =  res.u # weigths of the neurons with minimal loss
# You can also call them with minimizers_n  = res.u

# solutions plots

# number of points of the plot
dplot = [0.1, 0.05]
ts = [0.0 : dplot[1] : max_t;]
xs = [0.01 : dplot[2] : 0.99;]

times_points = Any[]
x_points = Any[]

for t in ts
    times_points = push!(times_points,t)
end

for x in xs
    x_points = push!(x_points,x)
end




### GENERATION OF GIF of the plot
anim = @animate for t ∈ times_points
    @info "Time $t..."
    sol_p = [ first(phi([t,x],minimizers_g)) / ( x * ( 1 - x ))    for x in x_points]
    title= t
    plot!(sol_p,title=title , legend = false)
end
gif(anim, "test_vf.gif", fps=200)

)
