push!(LOAD_PATH, ".")
N  = 2; T = 0.1; H = 5; L = 100


import Pkg
using Pkg
Pkg.activate(@__DIR__)

using Optim, Random, Distributions, CSV, DataFrames, MAT
using Plots, Dates, Statistics, Colors, ColorSchemes, StatsPlots
using Ipopt, JuMP, GaussianProcesses, LinearAlgebra, OrdinaryDiffEq

include("objects.jl")
include("computing.jl")
include("pxadmm.jl")
include("connectivity.jl")

ENV["GKSwstype"]="nul"

color =  cgrad(:turbo, N, categorical = true, scale = :lin)


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
pBounds = polyBound(str, v_min, v_max, a_min, a_max)


init    = [0, 0, 0, 0]
A = [1  -1  0 0; 0 0 1 -1]
b = [2.5 2.5 1. 1.]

robo    = robot(T, H, R, r, pBounds, init, A, b)
mGP     = Vector{GPBase}(undef, N)

for i in 1:N
    robo[i].meas   = measure!(robo[i].posn, GPtruth)  
    mGP[i]         = ElasticGPE(inDim, mean = MeanConst(robo[i].β), kernel = SEArd(ones(inDim), 0.), logNoise = -2., capacity = 3000, stepsize = 1000)
end



## Run simulation
println("Now start the simulation")
timer        = zeros(L)
Pred         = zeros(inDim, H, N)
PosX         = zeros(N,L)
PosY         = zeros(N,L)
[Pred[:,h,i] = robo[i].posn for h in 1:H, i in 1:N]

mesh = 2; ndef = -10
dataTot = [ndef*ones(3) for k in 1:N, i in 1:Int64(round((x_max-x_min)/mesh)), j in 1:Int64(round((y_max-y_min)/mesh))]


for k in 1:L

    for i in 1:N
        for j in (i+1):N
            dist = norm(robo[i].posn - robo[j].posn)
            if minD[k] >= dist
                minD[k] = dist
            end
        end
    end

    println("Time instance $k")
    global Pred, J, ResE, NB

    # Train
    t0 = time_ns()
    dstbRetrain!(robo, mGP, NB, dataTot)
    dt = (time_ns()-t0)/1e9
    println("Training time: $dt (s)")

    NB      = find_nears(robo, N)
    Eig2[k] = Index!(NB)
    pserSet = pserCon(robo, J[:, (k>1) ? k-1 : 1])
    # print(pserSet)

    Fig, RMSE[:,k] = myPlot(robo, mGP, vectemp, testSize, NB, color)
    png(Fig, "Figs-4robots-200/step $k"); #display(Fig)

    # Execute PxADMM
    t0 = time_ns()
    Pred, J[:,k], ResE[:,:,k] = dstbProxADMM!(robo, Pred, NB, pserSet, mGP; MAX_ITER = MAX_ITER)
    dt = (time_ns()-t0)/1e9
    println("Predicting time: $dt (s)")

    # Robots move to new locations and take measurement
    for i in 1:N
        robo[i].posn = Pred[:,1,i]
        PosX[i,k] = Pred[1,1,i]
        PosY[i,k] = Pred[2,1,i]
        robo[i].meas = measure!(robo[i].posn, GPtruth)
    end
end


gr(size=(1000,600))
FigRMSE = errorline(1:L, RMSE[:,1:L], linestyles = :solid, linewidth=2, secondarylinewidth=2, xlims = (0,L+0.5), errorstyle=:stick, 
secondarycolor=:blue,  legendfontsize = 24, tickfontsize = 30, framestyle = :box, label = "")
# errorline!(1:L, RMSE, linestyles = :solid, linewidth=2, xlims = (0,L+0.5), errorstyle=:ribbon, label="")
scatter!(1:L, [mean(RMSE[:,i]) for i in 1:L], label="Mean Errors")
png(FigRMSE, "Figs-10robots-200/RMSE")

pResE    = zeros(N,MAX_ITER)
[pResE[i,:] = sum(ResE[:,i,k] for k in 1:L)/L for i in 1:N]
id       = minimum([nonzero(pResE[i,:],0.) for i in 1:N]) - 7
FigpResE = errorline(1:id, pResE[:,1:id], secondarylinewidth=2, secondarycolor=:blue, errorstyle=:stick, framestyle = :box, yticks = 10 .^(-3.:1.:2.), label = "",
            legendfontsize = 24, tickfontsize = 30, xlims = (0, id-0.5), ylims = (1e-3, 2e2),  yscale=:log10, linestyles = :solid, linewidth=2)
scatter!(1:id, [mean(pResE[:,k]) for k in 1:id], label="Mean Errors")
png(FigpResE, "Figs-10robots-200/ResE")

                
png(plot(1:L, Eig2[1:L], linestyles = :dot, linewidth=3, xlims = (0,L+0.5), ylims = (0, 1.1*maximum(Eig2)), legendfontsize = 24,
                tickfontsize = 30, markershape = :circle, markersize = 5, label="", framestyle = :box), "Figs-10robots-200/Eig2")

png(plot(1:L, minD[1:L], linestyles = :dot, linewidth=3, xlims = (0,L+0.5), ylims = (0, 1.1*maximum(minD)), legendfontsize = 24,
                tickfontsize = 30, markershape = :circle, markersize = 5, label="", framestyle = :box), "Figs-10robots-200/minD")

matwrite("Data10robo200.mat", Dict("RMSE" => RMSE, "ResE" => ResE, "Eig2" => Eig2, "PosX" => PosX, "PosY" => PosY))