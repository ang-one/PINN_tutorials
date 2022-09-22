
"package for data"
using CSV,  DataFrames
using Plots


"package for inference"
using Lux, DiffEqFlux, NeuralPDE
using DifferentialEquations, Optimization
using OptimizationOptimJL, Random
using ModelingToolkit
using DiffEqCallbacks
import ModelingToolkit: Interval, infimum, supremum
"ode problem"
# t time
# mu_max max values of growth rate
"utility variable"

result_mu_max = Any[]

"Using Huang model 2013"
@parameters t
@parameters mu_max  [bounds = (0.0,10.0)]
@parameters lambda_  [bounds = (0.0,1000.0)]
@parameters alpha_ [bounds = (0.0001,Inf)]
@parameters n_max [bounds = (0.0001,Inf)]
@variables n_(..)

Dt = Differential(t)



eqs = Dt(n_(t)) ~ ( mu_max / (1.0 + exp(-1.0 * alpha_* ( t - lambda_ )))   ) * (1.0 -   exp( n_(t) -  n_max))

"ARCHITECTURE OF THE NN equal for each well"
#input layer the same dimension of the problem
input_  = 1
# neurons per internal layer
n_neurons = 15
#chain (I.E. THE  neural network. One for each unknown fuction)
chain_1 = Lux.Chain(Lux.Dense(input_,n_neurons,Lux.σ),Lux.Dense(n_neurons,n_neurons,Lux.σ),Lux.Dense(n_neurons,1))
fastchains = [chain_1]

dt=0.01
_strategy = NeuralPDE.GridTraining(dt)
  #_strategy = QuadratureTraining()
### IMPORTING DATA
cd("//Users//fabrizio.angaroni//Documents//dati-fitting")
filelist = readdir()
filelist = filelist[2:end]
counter = 0.0



for f in filelist # for over the files in the folder
    counter = counter + 1
    dfs = CSV.File(f)
    test_data = dfs
    names_of_cols = propertynames(test_data)
    ## transforming time in minutes from seconds
    times_data = test_data[names_of_cols[2]]./60.0



    for w in names_of_cols[3:end] ## loop over the wells

        od_values  = test_data[w]
        data = transpose(hcat(times_data,od_values))
        tsteps = times_data
        "Time span"

        max_t = tsteps[length(tsteps)]
        domains = [t ∈ Interval(0.0, max_t)]
        "Intial  condition"
        bcs = [n_(times_data[1]) ~ log(od_values[1]) ]

        "System of equations"


        @named prob_PDE = NeuralPDE.PDESystem(eqs, bcs, domains, [t], [n_(t)],
         [mu_max, lambda_, alpha_ , n_max],defaults = Dict([mu_max => 0.01, lambda_ => 4.0,  alpha_ => 4.0, n_max => 1.5]) )



        function additional_l(phi, θ, p)
            #print( θ)
            loss= sum(sum(abs2, [first(phi[1]([t_s], θ[:n_])) for t_s in times_data] .- log.(data[2,:]) ) / length(data[2,:]) )
            return loss
        end

        callback_loss = function (p,l)
            println("Current loss is: $l")
            return false
        end


        discretization = NeuralPDE.PhysicsInformedNN(fastchains, _strategy;   param_estim=true, additional_loss=additional_l)

        #phi_temp = NeuralPDE.get_phi([chain_1,chain_2],DiffEqBase.parameterless_type(initθs))

        prob = NeuralPDE.discretize(prob_PDE , discretization)

        res = Optimization.solve(prob, Adam(10^-3); callback =callback_loss , maxiters=250)
# isoutofdomain=(y,p,t)->any(x->x<0,y)
        prob = remake(prob, u0 = res.u)

        res = Optimization.solve(prob, BFGS();  callback =callback_loss , maxiters=350)


        "taking result of minimization"
        phi = discretization.phi
        minimizers_n = res.u.depvar[:n_]
        test_sol = [discretization.phi[1]([t_s],minimizers_n )[1] for t_s in times_data]
        pt1= Plots.scatter(tsteps,od_values,markersize=0.4,xlabel="Time (min)",ylabel= "OD")
        pt2= plot!(times_data,exp.(test_sol))
        png(string("//Users//fabrizio.angaroni//Documents//results_gc_fitter//huang_results_nn_well_",w,"_experiment_",counter))

        # mu_max, m_, Kp_, nu_, n_max
        "reading results of inferred parameters"
        inf_mu_max = res.minimizer[:p][1]
        inf_lambda = res.minimizer[:p][2]
        inf_alpha = res.minimizer[:p][3]
        inf_n_max = res.minimizer[:p][4]

        println(string("the inferred growth rate is ",inf_mu_max, " [1/min]" ))
        println(string("the inferred doubling time  is ", log(2.0)/inf_mu_max ), " Min.")
        println(string("the inferred lambda is ",inf_lambda ))
        println(string("the inferred alpha is ",inf_alpha ))

        write(string("//Users//fabrizio.angaroni//Documents//results_gc_fitter//huang_results_nn_well_",w,"_experiment_",counter,"mu_max.txt"),string(inf_mu_max))
        write(string("//Users//fabrizio.angaroni//Documents//results_gc_fitter//huang_results_nn_well_",w,"_experiment_",counter,"lambda.txt"),string(inf_lambda))
        write(string("//Users//fabrizio.angaroni//Documents//results_gc_fitter//huang_results_nn_well_",w,"_experiment_",counter,"alpha.txt"),string(inf_alpha))
        write(string("//Users//fabrizio.angaroni//Documents//results_gc_fitter//huang_results_nn_well_",w,"_experiment_",counter,"n_max.txt"),string(inf_n_max))
        result_mu_max = push!(result_mu_max,inf_mu_max)
        #test_mu = [discretization.phi[1]([t_s],minimizers_mu )[1] for t_s in times_data]
        #plot(times_data,test_mu )
        #CSV.write(string("C://Users//Fabrizio//Documents//growth_curves//mu_values_well_",w,"_experiment_",counter,".csv'"),hcat(times_data,test_mu))
        #CSV.write(string("C://Users//Fabrizio//Documents//growth_curves//mu_Max_values_well_",w,"_experiment_",counter,".csv'"),max(test_mu))
        #CSV.write(string("C://Users//Fabrizio//Documents//growth_curves//death_rate_values_well_",w,"_experiment_",counter,".csv'"),res.minimizer[:p][1])


    end    ## end of loop over the wells

end # end of the for over the files in the folder
write(string("//Users//fabrizio.angaroni//Documents//results_gc_fitter//huang_results_mu_max_summary.txt"),string(result_mu_max))


"##############################################################################"
"##############################################################################"
"##############################################################################"
