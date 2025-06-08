# transition_time.jl

using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions
using Plots 
using Colors
using DelimitedFiles
using LaTeXStrings

using JLD2

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs

function main()
    # Parameters
    
    rr = 0.01

    Nf = 10000
    
    dt = 0.02

    # Define the dictionary: Int → Vector of (Float64, String)
    upo_data = Dict{Int, Vector{Tuple{Float64,String}}}(
                    1 => [(1.55865/2, "L or R")],
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
    upo_colors = Dict(
        1 => :black,
        2 => :purple,
        3 => :blue,
        4 => :green,
        5 => :red,
        6 => :orange,
    )
    # Build new dict: p → Vector of (n_steps::Int, code)
    upo_steps = Dict{Int, Vector{Tuple{Int,String}}}()

    for (p, entries) in upo_data
        upo_steps[p] = [(round(Int, T/dt), s) for (T, s) in entries]
    end


    # Output directories
    
    data_path    = "/home/gabrielm/projects/RPMotifs/data/lorenz_RPC_x_time_shift/"
    figures_path = "figures/lorenz_RPC_x_time_shift_$(today())/Nf$(Nf)_rr$(rr)_dt$(dt)/"

    mkpath(data_path)
    mkpath(figures_path)
    
    time_series= generate_time_series("Lorenz",  [[10.0, 28.0, 8 / 3], dt], Nf, [1,2,3]) # Generate time series

    RP = Matrix(RecurrenceMatrix(StateSpaceSet(time_series), GlobalRecurrenceRate(rr); metric = Euclidean(), parallel = true))

    plot_recurrence_matrix(RP[1:1000,1:1000], "Lorenz System", figures_path; filename="recurrence_plot.png")

    time_shifts = 0:1:260
    tshifts_len = length(time_shifts)

    RPC = zeros(tshifts_len)
    Threads.@threads for dj_idx = 1:tshifts_len
        
        di = 0
        dj = time_shifts[dj_idx]
        RPC[dj_idx] = morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)

    end

    color_grad = cgrad([:gray, :red])

    #
    plot(time_shifts*dt, RPC, xlabel=L"\Delta j \ dt", ylabel=L"RPC \ (\Delta i = 0)", label="",
    lc=:black, lw=2,
     size=(500, 300), dpi=300,
     frame_style=:box, grid=false,
     )

    vline!([upo_data[1][1][1]], label=L"Period \ 1", ls=:dash, lc=upo_colors[1])
    vline!([upo_data[2][1][1]], label=L"Period \ 2", ls=:dash, lc=upo_colors[2])
    vline!([upo_data[3][1][1]], label=L"Period \ 3", ls=:dash, lc=upo_colors[3])
    vline!([upo_data[4][1][1]; upo_data[4][2][1]], label=L"Period \ 4", ls=:dash, lc=upo_colors[4])
    vline!([upo_data[5][1][1]; upo_data[5][2][1]; upo_data[5][3][1]], label=L"Period \ 5", ls=:dash, lc=:upo_colors[5])
    vline!([upo_data[6][1][1]; upo_data[6][2][1]; upo_data[6][3][1]; upo_data[6][4][1]; upo_data[6][5][1]], label=L"Period \ 6", ls=:dash, lc=upo_colors[6])



    savefig(joinpath(figures_path, "lorenz_RPC_x_time_shift_xz.png"))
end


main()