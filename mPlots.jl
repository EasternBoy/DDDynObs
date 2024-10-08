function plot_robot(robo::robot, x::Vector{Float64}, y::Vector{Float64}, type::Int64, k::Int64)
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

    fig = plot(Array(0:45), 2ones(46), size=(1000,500), color=:brown, linestyle=:dash, linewidth = 2, label="")
    plot!(rec, label="", fillcolor = plot_color(:red, 0.5), fontsize = 30)

    pts  = Plots.partialcircle(0, 2π, 100, robo.R)
    X, Y = Plots.unzip(pts)
    X    = robo.pose[1] .+ X
    Y    = robo.pose[2] .+ Y
    pts  = collect(zip(X, Y))
    plot!(Shape(pts), fillcolor=plot_color(:yellow, 0.1), linestyle=:dash, label="")

    if type == 3
        xlims!((-3.,45.)); ylims!((-17.,25.))
    elseif type == 2
        xlims!((-5.,47.)); ylims!((-13,13))
        annotate!(6, -10, text("Step $k", :red, :right, 30))
    else
        xlims!((-10.,35.)); ylims!((-12.,10.))
        annotate!(0, -10, text("Step $k", :red, :right, 30))
    end
    plot!([p[1]; x], [p[2];y], label="", linewidth = 2, linecolor = :green, xtickfontsize=24,ytickfontsize=24)
    plot!(robo.traj[1,:], robo.traj[2,:], label="", linewidth = 2, linecolor = :red)
    return fig
end


function plot_obs(obs::obstacle, Pmean::Matrix{Float64}, Pvar::Matrix{Float64}, φ::Vector{Float64})

    plot!(obs.traj[1,:], obs.traj[2,:], label = "", legend= :bottomright, linewidth = 2, linecolor = :blue)

    pts  = Plots.partialcircle(0, 2π, 100, obs.r)
    X, Y = Plots.unzip(pts)
    X    = obs.posn[1] .+ X
    Y    = obs.posn[2] .+ Y
    pts  = collect(zip(X, Y))
    plot!(Shape(pts), fillcolor = plot_color(:blue, 0.5), label="")


    for i in 1:H
        δ₁  = φ[i]*sqrt(Pvar[1,i]) 
        δ₂  = φ[i]*sqrt(Pvar[2,i])
        rec = Shape([Pmean[1,i] + δ₁, Pmean[1,i] + δ₁, Pmean[1,i] - δ₁, Pmean[1,i] - δ₁],
                    [Pmean[2,i] + δ₂, Pmean[2,i] - δ₂, Pmean[2,i] - δ₂, Pmean[2,i] + δ₂])
        # CI = Int64(round(Prob[i]*100))
        plot!(rec, fillcolor = plot_color(:orange, 0.5), label="")
    end
    plot!([obs.posn[1]; Pmean[1,:]], [obs.posn[2]; Pmean[2,:]], label = "", linewidth = 2, linecolor = :orange)

end

function plot_obs(obs::obstacle)
    plot!(obs.traj[1,:], obs.traj[2,:], label = "", aspect_ratio=1.0, legend= :topright, linewidth = 2, linecolor = :blue)

    pts  = Plots.partialcircle(0, 2π, 100, obs.r)
    X, Y = Plots.unzip(pts)
    X    = obs.posn[1] .+ X
    Y    = obs.posn[2] .+ Y
    pts  = collect(zip(X, Y))
    plot!(Shape(pts), fillcolor = plot_color(:blue, 0.5), label="")
end