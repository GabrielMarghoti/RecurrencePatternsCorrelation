# transition_time.jl

using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions
using Plots 
using DelimitedFiles

using JLD2

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs


function main()
    # Parameters
    
    rr = 0.01

    Nf = 40000
    
    dt = 0.1

    # Define the dictionary: Int → Vector of (Float64, String)
    upo_data = Dict{Int, Vector{Tuple{Float64,String}}}(
                    2 => [(1.55865, "LR")],
                    3 => [(2.30591, "LLR")],
                    4 => [(3.02358, "LLLR"), (3.08428, "LLRR")],
                    5 => [(3.72564, "LLLLR"), (3.82025, "LLLRR"), (3.86953, "LLRLR")],
                    6 => [
                        (4.41776, "LLLLLR"),
                        (4.53410, "LLLLRR"),
                        (4.56631, "LLLRRR"),
                        (4.59381, "LLLRLR"),
                        (4.63714, "LLRLRR")
                    ],
                    7 => [
                        (5.10303, "LLLLLLR"),
                        (5.23419, "LLLLLRR"),
                        (5.28634, "LLLLLRRR"),
                        (5.30120, "LLLLRLR"),
                        (5.33091, "LLLRLLR"),
                        (5.36988, "LLLRLRR"),
                        (5.37052, "LLRRLR")
                    ],
                    8 => [
                        (5.78341, "LLLLLLLR"),
                        (5.92499, "LLLLLLRR"),
                        (5.99044, "LLLLLRRR"),
                        (5.99732, "LLLLRLR"),
                        (6.01003, "LLLLRRRR"),
                        (6.03523, "LLLLRLLR"),
                        (6.08235, "LLLLRLRR"),
                        (6.08382, "LLLLRRLR"),
                        (6.10805, "LLLRLRRR"),
                        (6.12145, "LLLRLLRR"),
                        (6.12233, "LLLRRLLR"),
                        (6.13512, "LLLRRLRR"),
                        (6.15472, "LLLRLRLR"),
                        (6.17587, "LLRLLRLR"),
                        (6.18751, "LLRLRRLR"),
                        (6.19460, "LLRLRLRR")
                    ]
                )
    # Build new dict: p → Vector of (n_steps::Int, code)
    upo_steps = Dict{Int, Vector{Tuple{Int,String}}}()

    for (p, entries) in upo_data
        upo_steps[p] = [(round(Int, T/dt), s) for (T, s) in entries]
    end


    # Output directories
    
    data_path    = "/home/gabrielm/projects/RPMotifs/data/lorenz_RPC_basis/"
    figures_path = "figures/lorenz_RPC_basis_$(today())/Nf$(Nf)_rr$(rr)_dt$(dt)/"

    mkpath(data_path)
    mkpath(figures_path)
    
    time_series= generate_time_series("Lorenz",  [[10.0, 28.0, 8 / 3], dt], Nf, [1,2,3]) # Generate time series

    RP = Matrix(RecurrenceMatrix(StateSpaceSet(time_series), GlobalRecurrenceRate(rr); metric = Euclidean(), parallel = true))

    plot_recurrence_matrix(RP[1:1000,1:1000], "Lorenz System", figures_path; filename="recurrence_plot.png")


    for p_i=2:4
        for p_j=0:0
            if p_i == 0 && p_j == 0
                continue # Skip the case where both di and dj are zero
            end
            di = upo_steps[p_i][1][1]
            if p_j == 0
                dj = 0
            elseif p_j ==1
                dj = 1
            else
                dj = upo_steps[p_j][1][1]
            end

            RPC_name = "RPC_pi_$(p_i)_pj_$(p_j)_$(di)_dj$(dj)"



            RPC = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)
                
            circ = Shape(Plots.partialcircle(0, 2π))   


            #x-y-z
            plot(time_series[:, 1], time_series[:, 2], time_series[:, 3], color=:gray, lw=0.5, label="", alpha=0.1) # Smooth gray lines
                scatter!(time_series[:, 1], time_series[:, 2], time_series[:, 3], 
                marker_z=RPC, color=:amp, label="",ms=2, alpha=0.5,
                strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
                xlabel="x", ylabel="y", zlabel="z", title="RPC w($(di), $(dj)) = 1",
                size=(800, 600), dpi=200, frame_style=:box, grid=false)
            savefig(joinpath(figures_path, "lorenz_trajectory_xyz"*RPC_name*".png"))


            #x-y
            plot(time_series[:, 1], time_series[:, 2], color=:gray, lw=0.5, label="", alpha=0.1) # Smooth gray lines
                scatter!(time_series[:, 1], time_series[:, 2], 
                marker_z=RPC, color=:amp, label="",ms=2, alpha=0.9,
                strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
                xlabel="x", ylabel="y", title="RPC w($(di), $(dj)) = 1",
                size=(800, 600), dpi=200, frame_style=:box, grid=false)
            savefig(joinpath(figures_path, "lorenz_trajectory_xy"*RPC_name*".png"))


            #x-z
            plot(time_series[:, 1], time_series[:, 3], color=:gray, lw=0.5, label="", alpha=0.1) # Smooth gray lines
                scatter!(time_series[:, 1], time_series[:, 3], 
                marker_z=RPC, color=:amp, label="",ms=2, alpha=0.9,
                strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
                xlabel="x", ylabel="z", title="RPC w($(di), $(dj)) = 1; UPO $(upo_steps[p_i][1][2])",
                size=(800, 600), dpi=200, frame_style=:box, grid=false)
            savefig(joinpath(figures_path, "lorenz_trajectory_xz"*RPC_name*".png"))

        end
    end
end


main()