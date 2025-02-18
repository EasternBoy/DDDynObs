push!(LOAD_PATH, ".")
dim = 2 
T   = 0.1
H   = 10
L   = 65
ns  = 3


import Pkg
using Pkg
Pkg.activate(@__DIR__)
# Pkg.instantiate()

using Optim, Plots, JuMP, GaussianProcesses, LinearAlgebra, OrdinaryDiffEq, Flux, Random, Ipopt, Distributions



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
a_min   = -5.
a_max   =  5.
pBounds = polyBound(str_ang, v_min, v_max, a_min, a_max)


init  = [5., 2., 0., 6.]
A     = [1. 0; -1. 0.; 0. 1.; 0. -1.]
b     = [2.3, 2.3, 1., 1.]


test_points = [10, 20, 30, 40, 50]
mse_pre     = zeros(length(test_points))


nobs = 2
obs  = [obstacle(2., [18., -10.], ns); obstacle(2., [30., 18.], ns)]
robo = robot(T, H, R, ℓ, pBounds, init, A, b)



obsGP = Vector{Vector{GPBase}}(undef, nobs)
#obstacle 1
obsGP[1] = [GPE(mean = MeanConst(1.), kernel = SEArd([1.], 1.), logNoise = -2.), 
                 GPE(mean = MeanConst(1.), kernel = SEArd([1.], 1.), logNoise = -2.)]
#obstacle 2
obsGP[2] = [GPE(mean = MeanConst(1.), kernel = SEArd([1.], 1.), logNoise = -2.), 
                 GPE(mean = MeanConst(1.), kernel = SEArd([1.], 1.), logNoise = -2.)]
#recurrentNN
ncell = [8,8]
mNN   = [Chain(RNN(1 => ncell[1]), Dense(ncell[1] => 2)); Chain(RNN(1 => ncell[1]), Dense(ncell[1] => 2)); 
         Chain(RNN(1 => ncell[2]), Dense(ncell[2] => 2)); Chain(RNN(1 => ncell[2]), Dense(ncell[2] => 2))]

TypeOfObs = [2,3] # Cirle + Straight
pre_traj  = [zeros(dim,H,L) for n in 1:nobs]
# Run simulation
println("Now start the simulation")
for k in 1:L
    println("Step $k:")

    Ref = [[6(k+i-1)*T, 2.] for i in 1:H]
    δme = [zeros(dim, ns*H) for n in 1:nobs]
    δva = [zeros(dim, ns*H) for n in 1:nobs]
    Pmean = [zeros(dim, H) for n in 1:nobs]
    Pvar  = [zeros(dim, H) for n in 1:nobs]


    flag = Int64[]


    for n in 1:nobs
        run_obs!(obs[n], k-1, T, TypeOfObs[n])
        if norm(robo.pose[1:2] - obs[n].posn) < robo.R
            flag = [flag; n]

            @time Detection!(robo, obs[n], obsGP[n], mNN[n])

            for i in 1:dim
                δme[n][i,:], δva[n][i,:] = predictGP(mNN[n], obsGP[n][i], k*T .+ T*[x/ns for x in 1:H*ns], i)
            end

            # Uncertainty Propagation
            δme[n][:,1] += obs[n].posn
            for i in 1:H*ns-1
                δme[n][:,i+1] += δme[n][:,i]
                δva[n][:,i+1] += δva[n][:,i] + [exp(2obsGP[n][1].logNoise.value), exp(2obsGP[n][2].logNoise.value)]
            end

            Pmean[n] = δme[n][:,[i*ns for i in 1:H]]
            Pvar[n]  = δva[n][:,[i*ns for i in 1:H]]

            pre_traj[n][:,:,k] = Pmean[n]
        end
    end

    if length(flag) != 0
        Prob = [0.95  for i in 1:H]
        φ    = [quantile(Normal(0.,1.), (1 + Prob[i])/2)  for i in 1:H]

        @time x, y, θ, v, tα, a = nMPC_avoid(robo, obs, Ref, Pmean, Pvar, φ, flag)
        robo.predPose = [x';y';θ';v']
        fig = plot_robot(robo, x, y, 5, k)

        [plot_obs(obs[n], Pmean[n], Pvar[n], φ) for n in flag]
        [plot_obs(obs[n]) for n in setdiff(1:nobs,flag)]

 
        display(fig)
        png(fig, "$k")
        run!(robo, [tα[1], a[1]])
    else
        @time x, y, θ, v, tα, a = nMPC_free(robo, Ref)
        robo.predPose = [x';y';θ';v']
        fig = plot_robot(robo, x, y, 5, k)
        for n in 1:nobs
            plot_obs(obs[n])
        end
        display(fig)
        png(fig, "$k")
        run!(robo, [tα[1], a[1]])
    end
end