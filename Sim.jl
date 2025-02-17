push!(LOAD_PATH, ".")
dim = 2 
T   = 0.1
H   = 8
H₊  = 10
L   = 60
ns  = 3


import Pkg
using Pkg
Pkg.activate(@__DIR__)
# Pkg.instantiate()

using Optim, Random, CSV, DataFrames, MAT,  Distributions
using Plots, Dates, Statistics, Colors, ColorSchemes
using Ipopt, JuMP, GaussianProcesses, LinearAlgebra, OrdinaryDiffEq, Flux



include("objects.jl")
include("computing.jl")
include("MPC.jl")
include("mPlots.jl")

ENV["GKSwstype"]="nul"

color =  cgrad(:turbo, 2, categorical = true, scale = :lin)

R = 20.
ℓ = 1.9
str_ang =  0.6
v_min   = -10.
v_max   =  20.
a_min   = -2.
a_max   =  2.
pBounds = polyBound(str_ang, v_min, v_max, a_min, a_max)


init  = [0., 2., 0., 6.]
A     = [1. 0; -1. 0.; 0. 1.; 0. -1.]
b     = [2.3, 2.3, 1., 1.]
TypeOfObs = 1

test_points = [10, 20, 30, 40, 50]
obs_traj = zeros(2,L)
pre_traj = zeros(2,H,L)
mse_pre  = zeros(length(test_points))


if TypeOfObs == 1 #Sinusoid move
    obs   = obstacle(2., [22., 0.], ns)
elseif TypeOfObs == 2 #Circle move
    obs   = obstacle(2., [18., -10.], ns)
else #Straight move
    obs   = obstacle(2., [15., -15.], ns)
end

robo  = robot(T, H, H₊, R, ℓ, pBounds, init, A, b)

obsGP    = Vector{GPBase}(undef, 2) #MeanPoly(ones(1,10))
obsGP[1] = GPE(mean = MeanConst(1.), kernel = SEArd([1.], 1.), logNoise = -2.)
obsGP[2] = GPE(mean = MeanConst(1.), kernel = SEArd([1.], 1.), logNoise = -2.)
mNN      = Chain(RNN(1 => 16), Dense(16 => 2))


# Run simulation
println("Now start the simulation")
for k in 1:L
    println("Time instance $k")

    run_obs!(obs, k-1, T, TypeOfObs)

    obs_traj[:,k] = obs.posn

    Ref  = [[6(k+i-1)*T, 2.] for i in 1:H₊]

    if norm(robo.pose[1:2] - obs.posn) < robo.R
        @time Detection!(robo, obs, obsGP, mNN)
        δme = zeros(dim, ns*H)
        δva = zeros(dim, ns*H)
        for i in 1:dim
            δme[i,:], δva[i,:] = predictGP(mNN, obsGP[i], k*T .+ T*[x/ns for x in 1:H*ns], i)
        end

        # Uncertainty Propagation
        δme[:,1] += obs.posn
        for i in 1:H*ns-1
            δme[:,i+1] += δme[:,i]
            δva[:,i+1] += δva[:,i] + [exp(2obsGP[1].logNoise.value), exp(2obsGP[2].logNoise.value)]
        end

        Pmean = δme[:,[i*ns for i in 1:H]]
        Pvar  = δva[:,[i*ns for i in 1:H]]

        pre_traj[:,:,k] = Pmean

        # Prob = [0.95*0.9^(i-1)  for i in 1:H]
        Prob = [0.95  for i in 1:H]
        φ    = [quantile(Normal(0.,1.), (1 + Prob[i])/2)  for i in 1:H]

        @time x, y, θ, v, tα, a = nMPC_avoid(robo, obs, Ref, Pmean, Pvar, φ)
        robo.predPose = [x';y';θ';v']
        fig = plot_robot(robo, x, y, TypeOfObs,k)
        plot_obs(obs, Pmean, Pvar, φ)
        display(fig)
        png(fig, "$k")
        run!(robo, [tα[1], a[1]])
    else
        @time x, y, θ, v, tα, a = nMPC_free(robo, Ref)
        robo.predPose = [x';y';θ';v']
        fig = plot_robot(robo, x, y, TypeOfObs,k)
        plot_obs(obs)
        display(fig)
        png(fig, "$k")
        run!(robo, [tα[1], a[1]])
    end
end


for (i, v) in enumerate(test_points)
    print(v)
    mse = sqrt(1/H*sum(norm(obs_traj[:,v+h] - pre_traj[:,h,v])^2 for h in 1:H))
    mse_pre[i] = mse
end