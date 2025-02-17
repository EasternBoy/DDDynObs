function Detection!(robo::robot, obs::obstacle, obsGP::Vector{GPBase}, mNN::Chain; minData = 9, maxData = 30)
    robo_loca = robo.pose[1:2]
    obs_loca  = obs.posn
    R         = robo.R

    if norm(robo_loca - obs_loca) < R
        for i in 1:length(obsGP)
            if length(obsGP[i].x) <= maxData
                nIn = [obsGP[i].x[:]; obs.time]
                nOb = [obsGP[i].y; obs.δpos[i,:]]
                nGP = GPE([nIn';;], nOb, obsGP[i].mean, obsGP[i].kernel, obsGP[i].logNoise)
                obsGP[i] = deepcopy(nGP)
            else
                n = length(obs.time)
                nIn = [obsGP[i].x[n+1:end]; obs.time]
                nOb = [obsGP[i].y[n+1:end]; obs.δpos[i,:]]
                nGP = GPE([nIn';;], nOb, obsGP[i].mean, obsGP[i].kernel, obsGP[i].logNoise)
                obsGP[i] = deepcopy(nGP)
            end
        end
        
        for i in 1:length(obsGP)
            obsGP[i].logNoise.value = -2. #Initial
            obsGP[i].kernel.σ2      =  0.1 #Initial
            if length(obsGP[i].x) >= minData
                GaussianProcesses.optimize!(obsGP[i], domean = false, kern = true, noise = true, noisebounds = [-3., -1.], 
                                            kernbounds = [[-5.,-5.],[5.,5.]], method = Optim.NelderMead())

                t = [x for x in obsGP[1].x][1,:]
                Y = transpose([[y for y in obsGP[1].y] [y for y in obsGP[2].y]])
                trainNN(mNN, t, Y)
            else
                GaussianProcesses.optimize!(obsGP[i], domean = true, kern = true, noise = true, 
                                            noisebounds = [-3., -1.], kernbounds = [[-5.,-5.],[5.,5.]], method = Optim.NelderMead())
            end
        end
    end
end


function trainNN(mNN::Chain, dataIn::AbstractVector, dataOut::AbstractArray; epochs = 400) #time series

    X   = [[Float32.(x)] for x in dataIn]
    Y   = Float32.(dataOut)
    nd  = length(X)

    data = [(reshape(X[i], 1, 1), Y[:, i]) for i in 1:nd]

    loss(mNN, x, y) = norm(mNN(x) .- y)

    opt_state = Flux.setup(ADAM(0.1), mNN)

    bestloss        = Inf
    bestmodel       = mNN
    numincreases    = 0
    maxnumincreases = 20

    for epoch in 1:epochs
        Flux.train!(loss, mNN, data, opt_state)

        lss = sum(loss(mNN, X, Y) for (X,Y) in data)

        if lss < 0.95bestloss
            bestloss  = lss
            bestmodel = deepcopy(mNN)
        else numincreases +=1
        end

        numincreases > maxnumincreases ? break : nothing
    end
    mNN = bestmodel
end

function predictLSTM(mNN::Chain, dataIn::Vector{Float64}) #time series
    X = [[Float32.(x)] for x in dataIn]
    res = zeros(2,0)
    for x in X
        res = [res Float64.(mNN(reshape(x,1,1)))]
    end

    return res
end

function predictGP(mNN::Chain, mGP::GPBase, dataIn::Vector{Float64}, id::Int64; minData = 9)
    H = length(dataIn)
    postMean = ones(H)
    postVar  = zeros(H)
    
    Ktt   = cov(mGP.kernel, mGP.x, mGP.x)
    σω2   = exp(2mGP.logNoise.value)

    ndata = length(mGP.x)
    Cyy   = inv(Ktt + σω2*Matrix(I, ndata, ndata))

    Y = [y for y in mGP.y]  

    for i in 1:H
        if length(mGP.x) >= minData
            h   = dataIn[i]
            Kth = cov(mGP.kernel, mGP.x, [h;;])
            Khh = cov(mGP.kernel, [h;;], [h;;])

            postMean[i] = predictLSTM(mNN, [h])[id] + (Kth'*Cyy*(Y - predictLSTM(mNN, mGP.x[:])[id,:]))[1]
            postVar[i]  = (Khh - Kth'*Cyy*Kth)[1]
        else
            post, var   = predict_f(mGP, [dataIn[i];;] ; full_cov = true)
            postMean[i] = post[1]
            postVar[i]  = var[1]
        end
    end

    return postMean, postVar
end