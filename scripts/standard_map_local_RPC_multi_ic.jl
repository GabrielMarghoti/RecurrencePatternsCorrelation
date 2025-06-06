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
    
    rr = 0.02


    data_file = "orbit_manifold_1e-14_pi"# "orbit_random_0.2_pi" #


    # Output directories
    
    data_path    = "/home/gabrielm/projects/RPMotifs/data/orbits_standard_map_k=2.5_x0_y0/orbits/"
    figures_path = "figures/orbits_standard_map_k=2.5_multi_IC_rr$(rr)_$(today())"

    # Load all .dat files in data_path
    data_files = filter(f -> endswith(f, ".dat"), readdir(data_path))
    data_files = data_files[1:end]
    println("Found $(length(data_files)) data files in $data_path")

    n_ics = length(data_files)

    Nf = size(readdlm(joinpath(data_path, data_files[1])), 1)

    time_series = zeros(n_ics, Nf, 2)  # Adjust array to store results for each system and component

    # For demonstration, just print the file names
    for (i, f) in enumerate(data_files)
        println("Data file: ", f)
        time_series[i, :, :] = readdlm(joinpath(data_path, f))
    end

    mkpath(data_path)
    mkpath(figures_path)
    


    for di=-1:4
        for dj=-1:4
            if di == 0 && dj == 0
                continue # Skip the case where both di and dj are zero
            end
            RPC_name = "RPC_di$(di)_dj$(dj)"

            RPC = zeros(n_ics, Nf)

            for ic_idx = 1:n_ics
                
                RP = Matrix(RecurrenceMatrix(StateSpaceSet(time_series[ic_idx, :, :]), GlobalRecurrenceRate(rr); metric = Euclidean(), parallel = true))
            
                RPC[ic_idx, :] = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)
                
                plot_recurrence_matrix(RP, "Standard Map", figures_path; filename="recurrence_plot_"*data_files[ic_idx]*".png")

            end
            plot()
            for ic_idx = 1:n_ics
                circ = Shape(Plots.partialcircle(0, 2π))
                # Plot bifurcation diagram with Moran's I (Diagonal) as colors
                scatter!(time_series[ic_idx, :, 1], time_series[ic_idx, :, 2], 
                marker_z=RPC[ic_idx, :], color=:viridis, label="", ms=1.2, alpha=0.9,
                strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
                xlabel="x", ylabel="y",
                size=(800, 600), dpi=300, frame_style=:box, grid=false)
            end
            xlabel!("x")
            ylabel!("y")
            title!("RPC w($(di), $(dj)) = 1")
            savefig(joinpath(figures_path, "standard_map_eps$(eps)"*RPC_name*".png"))

        end
    end
end


main()