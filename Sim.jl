push!(LOAD_PATH, ".")
T = 0.1; H = 5; L = 10


import Pkg
using Pkg
Pkg.activate(@__DIR__)

using Optim, Random, Distributions, CSV, DataFrames, MAT
using Plots, Dates, Statistics, Colors, ColorSchemes, StatsPlots
using Ipopt, JuMP, GaussianProcesses, LinearAlgebra, OrdinaryDiffEq, Flux

include("objects.jl")
include("computing.jl")
include("MPC.jl")

ENV["GKSwstype"]="nul"

color =  cgrad(:turbo, 2, categorical = true, scale = :lin)


## Load data
# df            = CSV.read("SOM.csv", DataFrame, header = 0)
# dataOr        = Matrix{Float64}(df)
# loca_train    = Matrix(dataOr[:,1:2]') 
# obsr_train    = dataOr[:,end]
# inDim, numDa  = size(loca_train)


R = 40.
r = 3.
str_ang = 0.6
v_min   = -1.
v_max   =  2.
a_min   = -1.
a_max   =  1.
pBounds = polyBound(str_ang, v_min, v_max, a_min, a_max)


init  = [0., 0., 0., 0.]
A     = [1. -1. 0. 0.; 0. 0. 1. -1.]
b     = [2.5, 2.5, 1., 1.]


obs   = obstacle(2., [10., 0.], 3)
robo  = robot(T, H, R, r, pBounds, init, A, b)

obsGP = Vector{GPBase}(undef, 2) #MeanPoly(ones(1,10))
obsGP[1] = ElasticGPE(1, mean = MeanZero(), kernel = SEArd([1.], 1.), logNoise = -2.)
obsGP[2] = ElasticGPE(1, mean = MeanZero(), kernel = SEArd([1.], 1.), logNoise = -2.)



# Run simulation
println("Now start the simulation")
timer        = zeros(L)
Pred         = zeros(length(init), H)
[Pred[:,h]   = robo.pose for h in 1:H]


for k in 1:L
    println("Time instance $k")
    global Pred

    run_obs!(obs, k, T, 1)

    # Train
    t0 = time_ns()
    Detection!(robo, obs, obsGP)
    dt = (time_ns()-t0)/1e9
    println("Training time: $dt (s)")

end

mNN  = Chain(LSTM(1 => 32), Dense(32 => 2, identity))
trainNN(mNN, obs.tseri, obs.traj)
m_obsX, v_obsX = predictGP(mNN, obsGP[1], [2.2, 2.3, 2.4], 1)
m_obsY, v_obsY = predictGP(mNN, obsGP[2], [2.2, 2.3, 2.4], 2)
print("\n", m_obsY)
print("\n", v_obsY)


# # Create the RNN model
# mNN    = Chain(LSTM(1 => 32), Dense(32 => 2, identity))
# opt    = ADAM(5e-2)
# θ      = Flux.params(mNN)
# epochs = 400
# for epoch in 1:epochs
#     Flux.reset!(mNN)
#     ∇ = gradient(θ) do
#         mNN(X[1]) # Warm-up the model
#         sum(sum(Flux.Losses.mse.([mNN(x)[i] for x in X[2:end]], Y[i,2:end])) for i in 1:2)
#     end
#     Flux.update!(opt, θ, ∇)
# end



# gr(size=(1000,600))
# FigRMSE = errorline(1:L, RMSE[:,1:L], linestyles = :solid, linewidth=2, secondarylinewidth=2, xlims = (0,L+0.5), errorstyle=:stick, 
# secondarycolor=:blue,  legendfontsize = 24, tickfontsize = 30, framestyle = :box, label = "")
# # errorline!(1:L, RMSE, linestyles = :solid, linewidth=2, xlims = (0,L+0.5), errorstyle=:ribbon, label="")
# scatter!(1:L, [mean(RMSE[:,i]) for i in 1:L], label="Mean Errors")
# png(FigRMSE, "Figs-10robots-200/RMSE")

# pResE    = zeros(N,MAX_ITER)
# [pResE[i,:] = sum(ResE[:,i,k] for k in 1:L)/L for i in 1:N]
# id       = minimum([nonzero(pResE[i,:],0.) for i in 1:N]) - 7
# FigpResE = errorline(1:id, pResE[:,1:id], secondarylinewidth=2, secondarycolor=:blue, errorstyle=:stick, framestyle = :box, yticks = 10 .^(-3.:1.:2.), label = "",
#             legendfontsize = 24, tickfontsize = 30, xlims = (0, id-0.5), ylims = (1e-3, 2e2),  yscale=:log10, linestyles = :solid, linewidth=2)
# scatter!(1:id, [mean(pResE[:,k]) for k in 1:id], label="Mean Errors")
# png(FigpResE, "Figs-10robots-200/ResE")

                
# png(plot(1:L, Eig2[1:L], linestyles = :dot, linewidth=3, xlims = (0,L+0.5), ylims = (0, 1.1*maximum(Eig2)), legendfontsize = 24,
#                 tickfontsize = 30, markershape = :circle, markersize = 5, label="", framestyle = :box), "Figs-10robots-200/Eig2")

# png(plot(1:L, minD[1:L], linestyles = :dot, linewidth=3, xlims = (0,L+0.5), ylims = (0, 1.1*maximum(minD)), legendfontsize = 24,
#                 tickfontsize = 30, markershape = :circle, markersize = 5, label="", framestyle = :box), "Figs-10robots-200/minD")

# matwrite("Data10robo200.mat", Dict("RMSE" => RMSE, "ResE" => ResE, "Eig2" => Eig2, "PosX" => PosX, "PosY" => PosY))
