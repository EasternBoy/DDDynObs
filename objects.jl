mutable struct polyBound
    str_ang::Float64 # constant of bound constraint
    v_min::Float64 
    v_max::Float64 # constant of bound constraint
    a_min::Float64 
    a_max::Float64 # constant of bound constraint

    function polyBound(str_ang::Float64, v_min::Float64, v_max::Float64, a_min::Float64, a_max::Float64)
        return new(str_ang, v_min, v_max, a_min, a_max)
    end
end

function robot_ode!(dx, x, u, t)
    ℓ = 1.9
    dx[1] = x[4]*cos(x[3])
    dx[2] = x[4]*sin(x[3])
    dx[3] = x[4]*u[1]/ℓ
    dx[4] = u[2]
end

mutable struct robot
    T::Float64             # Sampling time
    H::Integer             # Horizon length
    H₊::Integer            # Horizon length
    R::Float64             # Detection range
    ℓ::Float64             # distance between two wheels
    pBnd::polyBound        # Physical limit
    pose::Vector{Float64}  # Current states: x, y, θ, v
    A::Matrix{Float64}     # Shape
    b::Vector{Float64}

    predPose::Matrix{Float64}

    input::Vector{Float64} # measurement

    traj::Matrix{Float64}  # All state up to current time
    inarr::Matrix{Float64}
    
    # ODEProblem for solving the ODE
    function robot(T::Float64, H::Integer, H₊::Integer, R::Float64, ℓ::Float64, pBnd::polyBound, x0::AbstractVector, 
                    A::Matrix{Float64}, b::Vector{Float64})

        obj         = new(T, H, H₊, R, ℓ, pBnd, x0, A, b)
        obj.input   = [0., 0.]

        obj.traj    = Matrix{Float64}(undef, length(x0), 0)
        obj.traj    = [obj.traj x0]
        obj.inarr   = Matrix{Float64}(undef, length(obj.input), 0)
        obj.inarr   = [obj.inarr obj.input]

        obj.predPose = repeat(x0, 1, H₊)

        return obj
    end
end

mutable struct obstacle
    r::Float64 #radius
    posn::Vector{Float64}
    ns::Int64
    δpos::Matrix{Float64}
    time::Vector{Float64}

    traj::Matrix{Float64}
    tseri::Vector{Float64}

    function obstacle(r::Float64, posn::Vector{Float64}, ns::Int64)

        obj        = new(r, posn, ns)
        obj.δpos   = zeros(length(posn), ns)
        obj.time   = zeros(ns)

        obj.traj  = [posn;;]
        obj.tseri = [0.]
        return obj
    end
end


function run!(robo::robot, in::Vector{Float64})
    # Copy the values from u to a.u
    robo.input = in
    prob  = OrdinaryDiffEq.ODEProblem(robot_ode!, robo.pose, (0.0, robo.T), robo.input)
    sol   = OrdinaryDiffEq.solve(prob, Tsit5())

    # Update state with the solution, and return it
    robo.pose  = sol[end]
    robo.traj  = [robo.traj  robo.pose]
    robo.inarr = [robo.inarr robo.input]
end


function run_obs!(obs::obstacle, k::Int64, τ::Float64, scenario::Int64)
    ns  = obs.ns
    Δτ  = τ/ns
    

    rng       = Random.MersenneTwister(1234)
    obs.time  = [k*τ + i*Δτ for i in 1:ns]
    obs.tseri = [obs.tseri; obs.time]

    if scenario == 1 #Sinusoid move
        for i in 1:ns
            obs.δpos[:,i] = Δτ*[-5., 5sin(π*(k*τ+i*Δτ))] + Δτ*randn(rng, Float64, (2))/ns
            obs.traj      = [obs.traj   obs.posn+sum(obs.δpos[:,k] for k in 1:i)]
        end
    elseif  scenario == 2 #Circle move
        for i in 1:ns
            obs.δpos[:,i] = Δτ*[5cos(π/4*(k*τ+i*Δτ)), 5sin(π/4*(k*τ+i*Δτ))] + Δτ*randn(rng, Float64, (2))/ns
            obs.traj      = [obs.traj   obs.posn+sum(obs.δpos[:,k] for k in 1:i)]
        end
    else #Straight move
        for i in 1:ns
            obs.δpos[:,i] = Δτ*[0., 6.] + Δτ*randn(rng, Float64, (2))/ns
            obs.traj      = [obs.traj   obs.posn+sum(obs.δpos[:,k] for k in 1:i)]
        end
    end
    obs.posn = obs.traj[:,end]
end