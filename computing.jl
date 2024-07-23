function Detection!(robo::robot, obs::obstacle, obsGP::Vector{GPBase}, mNN::Chain; minData = 10)
    robo_loca = robo.pose[1:2]
    obs_loca  = obs.posn
    R         = robo.R

    if norm(robo_loca - obs_loca) < R
        for i in 1:length(obsGP)
            append!(obsGP[i], reshape(obs.time,1,length(obs.time)), obs.mulpos[i,:])
        end
        
        for i in 1:length(obsGP)
            bestmodel = obsGP[i]
            bestmll   = -Inf
            if length(obsGP[i].x) >= minData
                for il2 in [0.1, 1., 10.]
                    obsGP[i].kernel.iℓ2[1]  =  il2
                    obsGP[i].logNoise.value = -2.
                    obsGP[i].kernel.σ2      =  1.
                    GaussianProcesses.optimize!(obsGP[i], domean = false, kern = true, noise = true, noisebounds = [-3., -1.], 
                                                kernbounds = [[-5.,-5.],[5.,5.]], method = Optim.NelderMead())
                    if bestmll < obsGP[i].mll
                        bestmll   = obsGP[i].mll
                        bestmodel = deepcopy(obsGP[i])
                    end
                end
                obsGP[i] = bestmodel
                t = [x for x in obsGP[1].x][1,:]
                Y = transpose([[y for y in obsGP[1].y] [y for y in obsGP[2].y]])
                trainNN(mNN, t, Y)
            else
                GaussianProcesses.optimize!(obsGP[i], domean = true, kern = true, noise = true, 
                                            noisebounds = [-3., -1.], kernbounds = [[-5.,-5.],[5.,5.]])
            end
        end
    end
end


function trainNN(mNN::Chain, dataIn::AbstractVector, dataOut::AbstractArray; epochs = 400) #time series

    X   = [[Float32.(x)] for x in dataIn]
    Y   = Float32.(dataOut)
    dim = minimum(size(dataOut))

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
            sum(sum(Flux.Losses.mse.([mNN(x)[i] for x in X], Y[i,:])) for i in 1:dim)
        end
        Flux.update!(opt, θ, ∇)

        loss = sum(sum(Flux.Losses.mse.([mNN(x)[i] for x in X], Y[i,:])) for i in 1:dim)

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


function predictGP(mNN::Chain, mGP::GPBase, dataIn::Vector{Float64}, id::Int64; minData = 10)
    H = length(dataIn)
    postMean = ones(H)
    postVar  = zeros(H)

    t     = obsGP[1].x.data
    
    Ktt   = cov(mGP.kernel, mGP.x, mGP.x)
    σω2   = exp(2*mGP.logNoise.value)

    ndata = length(t)
    Cyy   = inv(Ktt + σω2*Matrix(I, ndata, ndata))

    Y = [y for y in mGP.y]  

    for i in 1:H
        if length(mGP.x) >= minData
            h   = dataIn[i]
            Kth = cov(mGP.kernel, mGP.x, [h;;])
            Khh = cov(mGP.kernel, [h;;], [h;;])

            postMean[i] = predictLSTM(mNN, [h])[id] + (Kth'*Cyy*(Y - predictLSTM(mNN, t)[id,:]))[1]
            postVar[i]  = (Khh - Kth'*Cyy*Kth)[1]
        else
            postMean[i], var = predict_f(mGP, [dataIn[i];;] ; full_cov = true)
            postVar[i] = var[1]
        end
    end

    return postMean, postVar
end
