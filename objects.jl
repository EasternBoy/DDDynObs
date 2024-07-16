
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

# mutable struct obstacle
#     A::Matrix{Float64}
#     b::Float64
# end


function robot_ode!(dx, u, x, t)
    ℓ = 1.8
    dx[1] = x[4]*cos(x[3])
    dx[2] = x[4]*sin(x[3])
    dx[3] = x[4]*tan(u[1])/ℓ
    dx[4] = u[2]
end

mutable struct robot
    T::Float64             # Sampling time
    H::Integer             # Horizon length
    R::Float64             # Detection range
    r::Float64             # Obstacle radius
    pBnd::polyBound        # Physical limit
    pose::Vector{Float64}  # Current states: x, y, θ, v
    A::Matrix{Float64}     # Shape
    b::Vector{Float64}

    input::Vector{Float64} # measurement

    traj::Matrix{Float64}  # All state up to current time
    inarr::Matrix{Float64}



    # ODEProblem for solving the ODE
    opti
    ODEprob

    function robot(T::Float64, H::Integer, R::Float64, r::Float64, pBnd::polyBound, x0::AbstractVector, 
                    A::Matrix{Float64}, b::Vector{Float64})

        obj         = new(T, H, R, r, pBnd, x0, A, b)
        obj.input   = [0., 0.]

        obj.traj   = Matrix{Float64}(undef, length(x0), 0)
        obj.traj   = [obj.traj x0]
        obj.inarr  = Matrix{Float64}(undef, length(obj.input), 0)
        obj.inarr  = [obj.inarr obj.input]

        obj.opti = JuMP.Model(Ipopt.Optimizer)
        obj.ODEprob  = OrdinaryDiffEq.ODEProblem(robot_ode!, obj.pose, (0.0, T), obj.input)

        set_silent(obj.opti)
        return obj
    end
end

mutable struct obstacle
    r::Float64 #radius
    posn::Vector{Float64}
    ns::Int64
    mulpos::Matrix{Float64}
    time::Vector{Float64}

    traj::Matrix{Float64}
    tseri::Vector{Float64}

    function obstacle(r::Float64, posn::Vector{Float64}, ns::Int64)

        obj        = new(r, posn, ns)
        obj.mulpos = zeros(length(posn), ns)
        obj.time   = zeros(ns)
        for i in 1:ns
            obj.mulpos[:,i] = posn
        end
        # obj.traj   = Matrix{Float64}(undef, length(posn), 0)
        obj.traj   = obj.mulpos
        obj.tseri  = obj.time
        return obj
    end
end


function run!(robo::robot, in::Vector{Float64})
    # Copy the values from u to a.u
    robo.input = in
    sol = OrdinaryDiffEq.solve(robo.ODEprob, Tsit5(), reltol = 1e-5, abstol = 1e-5)

    # Update state with the solution, and return it
    robo.pose  = sol[end]
    robo.traj  = [robo.traj  robo.pose]
    robo.inarr = [robo.inarr robo.input]
end


function run_obs!(obs::obstacle, s::Int64, st::Float64, sce::Int64)
    ns = obs.ns
    τ = st/ns

    rng = Random.MersenneTwister(1234)
    for i in 1:ns
        obs.time[i] = s*st + i*τ
    end

    if sce == 1 # Straight impact
        vx = -3.
        vy = 2sin(2(s*st + τ))
        obs.mulpos[:,1] = obs.posn + τ*[vx, vy] + τ*randn(rng, Float64, (2))
        for i in 1:ns-1
            obs.mulpos[:,i+1] = obs.mulpos[:,i] + τ*[vx, vy] + τ*randn(rng, Float64, (2))
        end
        obs.posn = obs.mulpos[:,ns]

    elseif sce == 2 # Circle vehicle
        vx = 2cos(5(s*st + τ))
        vy = 2sin(5(s*st + τ))
        obs.mulpos[:,1] = obs.posn + τ*[vx, vy] + τ*randn(rng, Float64, (2))
        for i in 1:ns-1
            obs.mulpos[:,i+1] = obs.mulpos[:,i] + τ*[2cos(5(s*st + i*τ)), 2sin(5(s*st + i*τ))] + τ*randn(rng, Float64, (2))
        end
        obs.posn = obs.mulpos[:,ns]
    else
    end
    obs.traj  = [obs.traj   obs.mulpos]
    obs.tseri = [obs.tseri; obs.time]
end
