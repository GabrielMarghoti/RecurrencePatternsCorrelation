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
    
    rr = 0.1


    data_file = "orbit_manifold_1e-14_pi"# "orbit_random_0.2_pi" #


    # Output directories
    
    data_path    = "/home/gabrielm/projects/RPMotifs/data/orbits_standard_map_k=2.5/"
    figures_path = "figures/orbits_standard_map_k=2.5_rr$(rr)_$(today())"


    mkpath(data_path)
    mkpath(figures_path)
    

    time_series = readdlm(joinpath(data_path, data_file*".dat"))

    Nf = size(time_series, 1)
    # Adjust array to store results for each system and component

    RPC_di4_dj0  = zeros(Nf) 

    RPC_di3_dj0  = zeros(Nf) 

    RPC_di2_dj0  = zeros(Nf) 

    RPC_diag  = zeros(Nf) 

    RPC_anti_diag = zeros(Nf) 

    RPC_vert_line = zeros(Nf) 
    
    RPC_4sides = zeros(Nf) 

    RPC_8sides = zeros(Nf) 


    RP = Matrix(RecurrenceMatrix(StateSpaceSet(time_series), GlobalRecurrenceRate(rr); metric = Euclidean(), parallel = true))

    RPC_di4_dj0 = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == 4 && Δj == 0 ? 1 : 0), Δi_range = 4:4, Δj_range = 0:0)

    RPC_di3_dj0 = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == 3 && Δj == 0 ? 1 : 0), Δi_range = 0:3, Δj_range = 0:0)

    RPC_di2_dj0 = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == 2 && Δj == 0 ? 1 : 0), Δi_range = 0:2, Δj_range = 0:0)

    RPC_diag = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == Δj ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)

    RPC_anti_diag = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == -Δj ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)

    RPC_vert_line = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δj == 0 ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)

    RPC_4sides = local_morans_I(RP; weight_function = (Δi, Δj) -> (abs(Δi) + abs(Δj) == 1 ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)

    RPC_8sides = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi in [-1, 0, 1] && Δj in [-1, 0, 1] ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)

    plot_recurrence_matrix(RP, "Standard Map", figures_path; filename="recurrence_plot_"*data_file*".png")


    circ = Shape(Plots.partialcircle(0, 2π))

        # di =4, dj = 0
    scatter(time_series[:, 1], time_series[:, 2], 
             marker_z=RPC_di4_dj0, color=:viridis, label="", ms=4, alpha=0.9,
             strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
             xlabel="x", ylabel="y", title="Standard Map RPC for di=4 and dj = 0 patterns",
             size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path,  data_file*"_w_di4_dj0.png"))


    # di =3, dj = 0
    scatter(time_series[:, 1], time_series[:, 2], 
             marker_z=RPC_di3_dj0, color=:viridis, label="", ms=4, alpha=0.9,
             strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
             xlabel="x", ylabel="y", title="Standard Map RPC for di=3 and dj = 0 patterns",
             size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path,  data_file*"_w_di3_dj0.png"))


    # di =2, dj = 0
    scatter(time_series[:, 1], time_series[:, 2], 
             marker_z=RPC_di2_dj0, color=:viridis, label="", ms=4, alpha=0.9,
             strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
             xlabel="x", ylabel="y", title="Standard Map RPC for di=2 and dj = 0 patterns",
             size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path,  data_file*"_w__di2_dj0.png"))

    # Diagonal
    scatter(time_series[:, 1], time_series[:, 2], 
             marker_z=RPC_diag, color=:viridis, label="", ms=4, alpha=0.9,
             strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
             xlabel="x", ylabel="y", title="Standard Map RPC for Diagonal patterns",
             size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path,  data_file*"_w_diagonal.png"))


    # Anti-Diagonal
    scatter(time_series[:, 1], time_series[:, 2], 
             marker_z=RPC_anti_diag, color=:viridis, label="", ms=4, alpha=0.9,
             strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
             xlabel="x", ylabel="y", title="Standard Map RPC for Anti-Diagonal patterns",
             size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path,  data_file*"_w_anti_diagonal.png"))

    # Vertical Line
    scatter(time_series[:, 1], time_series[:, 2], 
             marker_z=RPC_vert_line, color=:viridis, label="", ms=4, alpha=0.9,
             strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
             xlabel="x", ylabel="y", title="Standard Map RPC for Vertical patterns",
             size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path,  data_file*"_w_vertical.png"))

    # 4-Sides
    scatter(time_series[:, 1], time_series[:, 2], 
             marker_z=RPC_4sides, color=:viridis, label="", ms=4, alpha=0.9,
             strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
             xlabel="x", ylabel="y", title="Standard Map RPC for Vertical and Horizontal patterns",
             size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path,  data_file*"_w_vert+horz.png"))

    # 8-Sides
    scatter(time_series[:, 1], time_series[:, 2], 
             marker_z=RPC_8sides, color=:viridis, label="", ms=4, alpha=0.9,
             strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
             xlabel="x", ylabel="y", title="Standard Map RPC for Block patterns",
             size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path,  data_file*"_w_block.png"))







end


main()