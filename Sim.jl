push!(LOAD_PATH, ".")
dim = 2; T = 0.1; H = 5; L = 20; ns = 3


import Pkg
using Pkg
Pkg.activate(@__DIR__)

using Optim, Random, CSV, DataFrames, MAT,  Distributions
using Plots, Dates, Statistics, Colors, ColorSchemes
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
ℓ = 1.9
str_ang =  0.6
v_min   = -1.
v_max   =  2.
a_min   = -1.
a_max   =  1.
pBounds = polyBound(str_ang, v_min, v_max, a_min, a_max)


init  = [0., 3., 0., 0.]
A     = [1. 0; -1. 0.; 0. 1.; 0. -1.]
b     = [2.5, 2.5, 1., 1.]


obs   = obstacle(2., [20., 0.], ns)
robo  = robot(T, H, R, ℓ, pBounds, init, A, b)

obsGP    = Vector{GPBase}(undef, 2) #MeanPoly(ones(1,10))
obsGP[1] = ElasticGPE(1, mean = MeanConst(1.), kernel = SEArd([1.], 1.), logNoise = -2.)
obsGP[2] = ElasticGPE(1, mean = MeanConst(1.), kernel = SEArd([1.], 1.), logNoise = -2.)
mNN      = Chain(LSTM(1 => 15), LSTM(15 => 2))


# Run simulation
println("Now start the simulation")
timer        = zeros(L)
Pmean        = zeros(dim, H)
Pvar         = zeros(dim, H)


for k in 1:L
    println("Time instance $k")
    global Pmean

    TypeOfObs = 1
    run_obs!(obs, k-1, T, TypeOfObs)

    @time Detection!(robo, obs, obsGP, mNN)
    Ref  = [[5(k+i)*T, 3.] for i in 1:H]

    if k >= 10
        for i in 1:dim
            Pmean[i,:], Pvar[i,:] = predictGP(mNN, obsGP[i], k*T .+ T*[x for x in 1:H], i)
        end

        Prob = [0.95*0.8^(i-1)  for i in 1:H]
        φ    = [quantile(Normal(0.,1.), (1 + Prob[i])/2)  for i in 1:H]
        
        fig = plot(obs.traj[1,:], obs.traj[2,:], label = "Past trajactory")
        plot!([obs.posn[1]; Pmean[1,:]], [obs.posn[2]; Pmean[2,:]], label = "Predicted mean trajectory")
        xlims!((2.,20.)); ylims!((-5.,10.))

        for i in 1:H
            δ₁  = φ[i]*sqrt(Pvar[1,i]) 
            δ₂  = φ[i]*sqrt(Pvar[2,i])
            rec = Shape([Pmean[1,i] + δ₁, Pmean[1,i] + δ₁, Pmean[1,i] - δ₁, Pmean[1,i] - δ₁],
                        [Pmean[2,i] + δ₂, Pmean[2,i] - δ₂, Pmean[2,i] - δ₂, Pmean[2,i] + δ₂])
            CI = Int64(round(Prob[i]*100))
            plot!(rec, label="$CI% Con. In. at $i step ahead")
        end

        θ = robo.pose[3]
        p = robo.pose[1:2]
        R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        RU = R*b[1]
        RD = R*b[2:3]
        LU = 
        LD = 
        rec = Shape([p[1] + δ₁, p[1] + δ₁, p[1] - δ₁, p[1] - δ₁],
                    [p[2] + δ₂, p[2] - δ₂, p[2] - δ₂, p[2] + δ₂])

        png(fig, "Predict after $k-th step (collect $ns data each step)")

        @time x, y, θ, v, tα, a = nMPC_avoid(robo, obs, Ref, Pmean, Pvar, φ)

        run!(robo, [tα[1], a[1]])
    else
        @time x, y, θ, v, tα, a = nMPC_free(robo, Ref)
        run!(robo, [tα[1], a[1]])
    end

end
