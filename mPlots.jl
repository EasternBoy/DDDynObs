function plot_robot(robo::robot, x::Vector{Float64}, y::Vector{Float64})
    θ = robo.pose[3]
    p = robo.pose[1:2]
    b = robo.b
    Rot = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    RU = Rot*[b[1], b[3]]
    RD = Rot*[b[1],-b[3]]
    LU = -RD
    LD = -RU
    rec = Shape([p[1]+RU[1], p[1]+RD[1], p[1]+LD[1], p[1]+LU[1]],
                [p[2]+RU[2], p[2]+RD[2], p[2]+LD[2], p[2]+LU[2]])
    fig = plot(rec, label="", fillcolor = plot_color(:red, 0.3))
    plot!([p[1]; x], [p[2];y], label="Predicted trajectory of robot", linewidth = 2, linecolor = :red)
    return fig
end


function plot_obs(obs::obstacle, Pmean::Matrix{Float64}, Pvar::Matrix{Float64}, φ::Vector{Float64})

    plot!([obs.posn[1]; Pmean[1,:]], [obs.posn[2]; Pmean[2,:]], label = "Predicted trajectory of obstacle", linewidth = 2, linecolor = :orange)
    plot!(obs.traj[1,:], obs.traj[2,:], label = "Trajectory of obstacle", aspect_ratio=1.0, legend= :topright, linewidth = 2, linecolor = :blue)
    # xlims!((-3.,25.)); ylims!((-6.,8.))

    pts  = Plots.partialcircle(0, 2π, 100, obs.r)
    X, Y = Plots.unzip(pts)
    X    = obs.posn[1] .+ X
    Y    = obs.posn[2] .+ Y
    pts  = collect(zip(X, Y))
    plot!(Shape(pts), fillcolor = plot_color(:blue, 0.3), label="")


    for i in 1:H
        δ₁  = φ[i]*sqrt(Pvar[1,i]) 
        δ₂  = φ[i]*sqrt(Pvar[2,i])
        rec = Shape([Pmean[1,i] + δ₁, Pmean[1,i] + δ₁, Pmean[1,i] - δ₁, Pmean[1,i] - δ₁],
                    [Pmean[2,i] + δ₂, Pmean[2,i] - δ₂, Pmean[2,i] - δ₂, Pmean[2,i] + δ₂])
        # CI = Int64(round(Prob[i]*100))
        plot!(rec, fillcolor = plot_color(:orange, 0.3), label="")
    end
end

function plot_obs(obs::obstacle)
    plot!(obs.traj[1,:], obs.traj[2,:], label = "Trajectory of obstacle", aspect_ratio=1.0, legend= :topright, linewidth = 2, linecolor = :blue)
    # xlims!((-3.,25.)); ylims!((-6.,8.))

    pts  = Plots.partialcircle(0, 2π, 100, obs.r)
    X, Y = Plots.unzip(pts)
    X    = obs.posn[1] .+ X
    Y    = obs.posn[2] .+ Y
    pts  = collect(zip(X, Y))
    plot!(Shape(pts), fillcolor = plot_color(:blue, 0.3), label="")
end
