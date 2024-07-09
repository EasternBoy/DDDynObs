
mutable struct polyBound
    str_ang::Float64 # constant of bound constraint
    v_min::Float64 
    v_max::Float64 # constant of bound constraint
    v_min::Float64 
    v_max::Float64 # constant of bound constraint

    function polyBound(s_max::Float64, x_min::Float64, x_max::Float64, y_min::Float64, y_max::Float64)
        return new(s_max, x_min, x_max, y_min, y_max)
    end
end

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
        obj.inarr  = Matrix{Float64}(undef, length(input), 0)

        obj.opti = JuMP.Model(Ipopt.Optimizer)
        ODEprob  = OrdinaryDiffEq.ODEProblem(robot_ode!, obj.pose, (0.0, T), obj.input)

        set_silent(obj.opti)
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


mutable struct obstacle
     A::Matrix{Float64}
     b::Float64
end

# Circle obstacle

function run_obs!(s::Int64, st::Float64, sce::Int64)
    if sce == 1 #Straight obs
    elseif sce == 2
    else
    end
end