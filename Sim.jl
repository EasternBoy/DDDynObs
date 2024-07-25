push!(LOAD_PATH, ".")
dim = 2 
T   = 0.1
H   = 5
H₊  = 8
L   = 25 
ns  = 3


import Pkg
using Pkg
Pkg.activate(@__DIR__)

using Optim, Random, CSV, DataFrames, MAT,  Distributions, NLopt
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
v_min   = -20.
v_max   =  20.
a_min   = -40.
a_max   =  10.
pBounds = polyBound(str_ang, v_min, v_max, a_min, a_max)


init  = [0., 3., 0., 0.]
A     = [1. 0; -1. 0.; 0. 1.; 0. -1.]
b     = [2.3, 2.3, 1., 1.]


obs   = obstacle(2., [20., 0.], ns)
robo  = robot(T, H, H₊, R, ℓ, pBounds, init, A, b)

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
    Ref  = [[5(k+i-1)*T, 3.] for i in 1:H₊]

    if k >= 10
        for i in 1:dim
            Pmean[i,:], Pvar[i,:] = predictGP(mNN, obsGP[i], k*T .+ T*[x for x in 1:H], i)
        end

        # Prob = [0.95*0.8^(i-1)  for i in 1:H]
        Prob = [0.95  for i in 1:H]
        φ    = [quantile(Normal(0.,1.), (1 + Prob[i])/2)  for i in 1:H]
        
        fig = plot(obs.traj[1,:], obs.traj[2,:], label = "Past trajectory")
        plot!([obs.posn[1]; Pmean[1,:]], [obs.posn[2]; Pmean[2,:]], label = "Predicted mean trajectory") 
        xlims!((0.,20.)); ylims!((-2.,8.))

        for i in 1:H
            δ₁  = φ[i]*sqrt(Pvar[1,i]) 
            δ₂  = φ[i]*sqrt(Pvar[2,i])
            rec = Shape([Pmean[1,i] + δ₁, Pmean[1,i] + δ₁, Pmean[1,i] - δ₁, Pmean[1,i] - δ₁],
                        [Pmean[2,i] + δ₂, Pmean[2,i] - δ₂, Pmean[2,i] - δ₂, Pmean[2,i] + δ₂])
            CI = Int64(round(Prob[i]*100))
            plot!(rec, label="$CI% Con. In. at $i step ahead") #
        end

        θ = robo.pose[3]
        p = robo.pose[1:2]
        Rot = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        RU = Rot*[b[1], b[3]]
        RD = Rot*[b[1],-b[3]]
        LU = -RD
        LD = -RU
        rec = Shape([p[1]+RU[1], p[1]+RD[1], p[1]+LD[1], p[1]+LU[1]],
                    [p[2]+RU[2], p[2]+RD[2], p[2]+LD[2], p[2]+LU[2]])
        plot!(rec, label="Robo at step $k")

        png(fig, "Predict after $k-th step (collect $ns data each step)")

        @time x, y, θ, v, tα, a = nMPC_avoid(robo, obs, Ref, Pmean, Pvar, φ)

        run!(robo, [tα[1], a[1]])
    else
        @time x, y, θ, v, tα, a = nMPC_free(robo, Ref)
        run!(robo, [tα[1], a[1]])
    end

end
