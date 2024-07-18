function Detection!(robo::robot, obs::obstacle, obsGP::Vector{GPBase}, mNN::Chain)
    robo_loca = robo.pose[1:2]
    obs_loca  = obs.posn
    R         = robo.R

    if norm(robo_loca - obs_loca) < R
        for i in 1:length(obsGP)
            append!(obsGP[i], reshape(obs.time,1,length(obs.time)), obs.mulpos[i,:])
            if length(obsGP[i].x) >= 6
                method = NelderMead(;   parameters = Optim.AdaptiveParameters(),
                                        initial_simplex = Optim.AffineSimplexer())
                obsGP[i].logNoise.value = -2.
                obsGP[i].kernel.σ2      =  1.

                obsGP[i].kernel.iℓ2     = [10.]
                GaussianProcesses.optimize!(obsGP[i], domean = true, kern = true, noise = true, 
                                            noisebounds = [-3., -1.], kernbounds = [[-5.,-5.],[5.,5.]]; method)
            end
        end
        
        if length(obsGP[1].x) >= 10
            t = [x for x in obsGP[1].x][1,:]
            Y = transpose([[y for y in obsGP[1].y] [y for y in obsGP[2].y]])
            trainNN(mNN, t, Y)
        end
    end
end


function trainNN(mNN::Chain, dataIn::AbstractVector, dataOut::AbstractArray; epochs = 400) #time series

    X = [[Float32.(x)] for x in dataIn]
    Y = Float32.(dataOut)

    bestloss        = Inf
    bestmodel       = mNN
    numincreases    = 0
    maxnumincreases = 50

    θ     = Flux.params(mNN)
    opt   = ADAM(1e-2)

    for epoch in 1:epochs
        Flux.reset!(mNN)
        ∇ = gradient(θ) do
            mNN(X[1]) # Warm-up the model
            sum(sum(Flux.Losses.mse.([mNN(x)[i] for x in X[1:end]], Y[i,1:end])) for i in 1:2)
        end
        Flux.update!(opt, θ, ∇)

        loss = sum(sum(Flux.Losses.mse.([mNN(x)[i] for x in X[1:end]], Y[i,1:end])) for i in 1:2)

        if loss < 0.95bestloss
            bestloss  = loss
            bestmodel = deepcopy(mNN)
        else numincreases +=1
        end
        numincreases > maxnumincreases ? break : nothing
    end
    mNN = bestmodel
end

function predictLSTM(mNN::Chain, dataIn::Vector{Float64}; recal = 20) #time series
    X = [[Float32.(x)] for x in dataIn]
    for i in 1:recal
        for x in X
            mNN(x)
        end
    end

    res = zeros(2,0)
    for x in X
        res = [res Float64.(mNN(x))]
    end

    return res
end


function SEkern!(mGP::GPBase, X1::Vector{Float64}, X2::Vector{Float64})
    row   = length(X1)
    col   = length(X2)
    K     = zeros(row, col)
    σκ2   = mGP.kernel.σ2
    iℓ2   = mGP.kernel.iℓ2

    for i in 1:row
        for j in 1:col
            K[i,j] = σκ2*exp(-0.5*iℓ2[1]*(X1[i] - X2[j])^2)
        end
    end
    return K
end

function predictGP(mNN::Chain, mGP::GPBase, dataIn::Vector{Float64}, id::Int64)
    H = length(dataIn)
    postMean = ones(H)
    postVar  = zeros(H)

    t     = [x for x in mGP.x][1,:]
    ndata = length(t)
    Ktt   = SEkern!(mGP, t, t)
    σω2   = exp(2*mGP.logNoise.value)
    Cyy   = inv(Ktt + σω2*Matrix(I, ndata, ndata))

    Y = [y for y in mGP.y]  

    for i in 1:H
        h   = dataIn[i]
        Kth = SEkern!(mGP, t, [h])
        Khh = SEkern!(mGP, [h], [h])

        postMean[i] = predictLSTM(mNN, [h])[id] + (Kth'*Cyy*(Y - predictLSTM(mNN, t)[id,:]))[1]
        postVar[i]  = (Khh - Kth'*Cyy*Kth)[1]
    end

    return postMean, postVar
end
