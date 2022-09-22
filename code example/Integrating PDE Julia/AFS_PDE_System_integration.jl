"""
TEST PARTIAL DIFFERENTIAL EQUATION resoltuion using PINN
Dynamics from: Non-equilibrium theory of the allele frequency spectrum
Theoretical Population Biology 71 (2007) 109–119. Using an logistic growing pop.
but now also pop growth is defined by a differental equation
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
@variables n_pop(..) [bounds = (0.0, Inf)] # Unknow function of the pop size
"symbolyc functions"
Dt = Differential(t)
Dxx = Differential(x) ^ 2
Dx = Differential(x)

"constant parameters"
s = 2.0 # Fisher selection coefficient
α = 1.4 # growth rate of the population
cc = 100.0 # carrying capacity
max_t = 10.0 # maximum time of the simulation

"Symbolic PDE that we want to solve. Note that the Allele Freq. Spectrum
(AFS) is defined as f(x,t) = g(x,t) * ( x * ( 1 - x ))"
eqs =[ Dt(g(t,x)) ~  - 1 * s * x * (1 - x ) * Dx(g(t,x)) + ((x *(1-x)) /( 2 * exp(α * t)) ) *  Dxx(g(t,x)),
    Dt(n_pop(t)) ~ α * n_pop(t) * (1- n_pop(t)/cc ) ]

"Imposing intial condition"

##  points where the inital condition is evaluated
array_of_x = [0.0:0.01:1;]

#### EQUILIBRIUM BCS for AFS
if s > 0
    bcs = [g(0,array_of_x[k]) ~ (exp(2 * s) * (1 - exp( - 2 * s * (1 - array_of_x[k] )))) / (exp(2 * s) - 1) for k in 1:length(array_of_x) ]

end

if s == 0
    bcs = [g(0,array_of_x[k]) ~ 1-array_of_x[k] for k in 1:length(array_of_x) ]

end


# starting with one element in the pop

bcs = push!(bcs, n_pop(0) ~ 5.0 )

# VF and time domains
domains = [t ∈ Interval(0.0,max_t) , x ∈ Interval(0.0,1.0)]

" setting symbolic problem"

@named pde_system = PDESystem(eqs,bcs,domains,[t,x],[g(t,x),n_pop(t)])

"Architecture of the NN"
# Neural network
## Size of input layer MUST have the same dimension of the problem
input_  = length(domains)
# defining the NN
#NN for the g(x,t) first layer should have 2 neurons of input one for x and one for t
chain_1 = Lux.Chain(Lux.Dense(input_,16,Lux.σ),Lux.Dense(16,16,Lux.σ),Lux.Dense(16,1))
#NN for the n_pop(t) first layer should have 1 neuron as input for t

chain_2 = Lux.Chain(Lux.Dense(1,16,Lux.σ),Lux.Dense(16,16,Lux.σ),Lux.Dense(16,1))

# merging the two NN
fast_chain =[chain_1,chain_2]

# strategy of training, here I use the simplest case where the integration mesh
# is fixed
dx = 0.05 # size of mesh step
_strategy = GridTraining(dx)

# definig the Architecture combined with the training strategy
discretization = PhysicsInformedNN(fast_chain, _strategy)

# combining the Architecture with the mathematical problem . I.e. function to
# transform the integration problem as a symbolic optimization
prob = discretize(pde_system , discretization)




### callback function used during the optimization
callback = function (p,l)
    println("Current loss is: $l")
    return false
end


# selecting optimizer
# starting optimization with ADAM
res = Optimization.solve(prob, Adam(10^-3); callback = callback, maxiters = 1000)
# second  optimization with BFGS
prob = remake(prob, u0 = res.minimizer)

opt = OptimizationOptimJL.BFGS()

res = solve(prob,opt; callback = callback, maxiters = 2000)

# taking solutions



"solutions plots"
### Plot of the population growth

phi = discretization.phi # symbolic
minimizers_n_pop = res.u.depvar[:n_pop] #minimal weights for the part of the NN associated to n_pop
minimizers_g = res.u.depvar[:g] #minimal weights for the part of the NN associated to g


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
    sol_p = [ first(phi[1]([t,x],minimizers_g)) / ( x * ( 1 - x ))    for x in x_points]
    title= t
    plot!(sol_p,title=title , legend = false)
end
gif(anim, "test_vf.gif", fps=200)

)

# number of elements in the model

test_sol = [phi[2]( [t_s], minimizers_n_pop ) for t_s in times_data]
