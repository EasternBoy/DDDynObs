function nMPC_avoid(robo::robot, obs::obstacle, ref::Vector{Vector{Float64}}, Pmean::Matrix{Float64}, Pvar::Matrix{Float64}, φ::Vector{Float64})
    
    opti  = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    H     = robo.H
    H₊    = robo.H₊
    τ     = robo.T
    A     = robo.A
    b     = robo.b
    E     = A
    ℓ     = robo.ℓ
    r     = obs.r

    d = [φ[i]*[sqrt(Pvar[1,i]), sqrt(Pvar[1,i]), sqrt(Pvar[2,i]), sqrt(Pvar[2,i])] for i in 1:H]

    bds   = robo.pBnd
    state = robo.pose

    dimR = length(robo.b)
    dimO = length(d[1])

    @variable(opti, x[1:H₊])
    @variable(opti, y[1:H₊])
    @variable(opti, θ[1:H₊])
    @variable(opti, v[1:H₊])
    @variable(opti, tα[1:H₊])
    @variable(opti, a[1:H₊])
    @variable(opti, λ[1:dimR,1:H₊])
    @variable(opti, μ[1:dimO,1:H₊])

    J = sum(dot([x[i], y[i]] - ref[i], [x[i], y[i]] - ref[i]) for i in 1:H₊)
    @objective(opti, Min, J)


    @constraints(opti, begin    bds.v_max  .>= v .>= bds.v_min
                                bds.a_max  .>= a .>= bds.a_min
                                tan(bds.str_ang) .>= tα .>= -tan(bds.str_ang) end)

    @constraint(opti, λ .>= 0)
    @constraint(opti, μ .>= 0)

    for i in 1:H₊
        if i == 1
            @constraint(opti, x[i] == state[1] + τ*state[4]*cos(state[3]))
            @constraint(opti, y[i] == state[2] + τ*state[4]*sin(state[3]))
            @constraint(opti, θ[i] == state[3] + τ*tα[i]/ℓ)
            @constraint(opti, v[i] == state[4] + τ*a[i])
        else
            @constraint(opti, x[i] == x[i-1] + τ*v[i-1]*cos(θ[i-1]))
            @constraint(opti, y[i] == y[i-1] + τ*v[i-1]*sin(θ[i-1]))
            @constraint(opti, θ[i] == θ[i-1] + τ*tα[i]/ℓ)
            @constraint(opti, v[i] == v[i-1] + τ*a[i])
        end
    end

    for i in 1:H
        @constraint(opti, dot(E'*μ[:,i], E'*μ[:,i]) <= 1.)
        @constraint(opti, μ[:,i]'*(E*([x[i], y[i]] - Pmean[:,i]) - d[i])- λ[:,i]'*b >= r)
        @constraint(opti, λ[:,i]'*A + μ[:,i]'*E*[cos(θ[i]) -sin(θ[i]); sin(θ[i]) cos(θ[i])] .== 0)
    end



    JuMP.optimize!(opti)

    x  = JuMP.value.(x)
    y  = JuMP.value.(y)
    θ  = JuMP.value.(θ)
    v  = JuMP.value.(v)
    tα = JuMP.value.(tα) 
    a  = JuMP.value.(a)
    # λ  = JuMP.value.(λ)
    # μ  = JuMP.value.(μ)

    return x, y, θ, v, tα, a
end

function nMPC_free(robo::robot, ref::Vector{Vector{Float64}})

    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    H     = robo.H
    τ     = robo.T
    ℓ     = robo.ℓ
    bds   = robo.pBnd
    state = robo.pose

    @variable(opti, x[1:H])
    @variable(opti, y[1:H])
    @variable(opti, θ[1:H])
    @variable(opti, v[1:H])
    @variable(opti,tα[1:H])
    @variable(opti, a[1:H])

    J = sum((x[i] - ref[i][1])^2 + (y[i] - ref[i][2])^2 for i in 1:H)
    @objective(opti, Min, J)

    @constraints(opti, begin    bds.v_max  .>= v .>= bds.v_min
                                bds.a_max  .>= a .>= bds.a_min
                                tan(bds.str_ang) .>= tα .>= -tan(bds.str_ang) end)
    for i in 1:H
        if i == 1
            @constraint(opti, x[i] == state[1] + τ*state[4]*cos(state[3]))
            @constraint(opti, y[i] == state[2] + τ*state[4]*sin(state[3]))
            @constraint(opti, θ[i] == state[3] + τ*tα[i]/ℓ)
            @constraint(opti, v[i] == state[4] + τ*a[i])
        else
            @constraint(opti, x[i] == x[i-1] + τ*v[i-1]*cos(θ[i-1]))
            @constraint(opti, y[i] == y[i-1] + τ*v[i-1]*sin(θ[i-1]))
            @constraint(opti, θ[i] == θ[i-1] + τ*tα[i]/ℓ)
            @constraint(opti, v[i] == v[i-1] + τ*a[i])
        end
    end

    JuMP.optimize!(opti)

    print(JuMP.value.(J))

    return JuMP.value.(x), JuMP.value.(y), JuMP.value.(θ), JuMP.value.(v), JuMP.value.(tα), JuMP.value.(a)
    
end
