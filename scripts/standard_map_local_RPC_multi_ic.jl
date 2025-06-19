# transition_time.jl

using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions
using Plots 
using DelimitedFiles
using Distances
using JLD2

using LaTeXStrings

gr() 
default(fontfamily="Computer Modern", linewidth=2, label=nothing, grid=false, framestyle=:box)


include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs


function custom_recurrence_plot(data::Matrix{Float64}, epsilon::Float64; periodic::Bool=false, period::Float64=2π)
    N, D = size(data)
    RP = falses(N, N)  # preallocate recurrence matrix

    for i in 1:N
        for j in i:N  # symmetric, only compute upper triangle
            if periodic
                d = 0.0
                for k in 1:D
                    delta = abs(data[i, k] - data[j, k])
                    md = mod(delta, period)
                    mindist = min(md, period - md)
                    d += mindist^2
                end
                d = sqrt(d)
            else
                d = sqrt(sum((data[i, :] .- data[j, :]).^2))
            end

            RP[i, j] = d ≤ epsilon
            RP[j, i] = RP[i, j]  # symmetry
        end
    end

    return RP
end

function main()
    # Parameters
    
    eps = 1.0 # Recurrence threshold


    data_file = "orbit_random_0.2_pi" # "orbit_manifold_1e-14_pi"# 


    # Output directories
    
    data_path    = "/home/gabrielm/projects/RPMotifs/data/orbits_standard_map_k=2.5_x0_y0/orbits/"
    figures_path = "figures/orbits_standard_map_k=2.5_multi_IC/eps$(eps)/date_$(today())"

    # Load all .dat files in data_path
    data_files = filter(f -> endswith(f, ".dat"), readdir(data_path))
    data_files = data_files
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
    di_range=1:4
    dj_range=-4:4

    panels_letters = [
        "(aa)" "(ab)" "(ac)" "(ad)";
        "(ba)" "(bb)" "(bc)" "(bd)";
        "(ca)" "(cb)" "(cc)" "(cd)";
        "(da)" "(db)" "(dc)" "(dd)";
        "(ea)" "(eb)" "(ec)" "(ed)";
        "(fa)" "(fb)" "(fc)" "(fd)";
        "(ga)" "(gb)" "(gc)" "(gd)";
        "(ha)" "(hb)" "(hc)" "(hd)";
        "(ia)" "(ib)" "(ic)" "(id)"
    ]

    for di=di_range
        for dj=dj_range
            if di == 0 && dj == 0
                continue # Skip the case where both di and dj are zero
            end
            RPC_name = "RPC_di$(di)_dj$(dj)"

            # Save the RPC data for each (di, dj) combination
            save_path = joinpath(data_path, "RPC_eps$(eps)_di$(di)_dj$(dj).jld2")

            if isfile(save_path)
                println("Loading cache: $save_path")
                @load save_path RPC
            else
                RPC = zeros(n_ics, Nf)

                for ic_idx = 1:n_ics
                    
                    
                    RP = custom_recurrence_plot(time_series[ic_idx, :, :], eps; periodic=true, period=2π)
                
                    RPC[ic_idx, :] = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)
                    
                    #plot_recurrence_matrix(RP, "Standard Map", figures_path; filename="recurrence_plot_"*data_files[ic_idx]*".png")

                end
            end
            plot()
            for ic_idx = 1:n_ics
                circ = Shape(Plots.partialcircle(0, 2π))
                # Plot bifurcation diagram with Moran's I (Diagonal) as colors
                scatter!(time_series[ic_idx, 5:end-5, 1], time_series[ic_idx, 5:end-5, 2], 
                marker_z=RPC[ic_idx, 5:end-5], color=:vik, label="", ms=1.2, alpha=0.9,
                strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
                xlabel=L"x", ylabel=L"y", widen=false,
                xticks=([0, π/2, π, 3π/2, 2π], [L"0", L"π/2", L"π", L"3π/2", L"2π"]),
                yticks=([0, π/2, π, 3π/2, 2π], [L"0", L"π/2", L"π", L"3π/2", L"2π"]),
                colorbar_title="lRPC",
                clims=(-1.0, 1.0),
                size=(500, 400), dpi=200, frame_style=:box, grid=false)
            end
            xlabel!("x")
            ylabel!("y")
            annotate!(0, 2π-0.25, text("$(panels_letters[end-(dj-dj_range[1]+1)+1, di-di_range[1]+1])"*L" \ w_{Δi, Δj} = δ_{Δi, %$di} δ_{Δj, %$dj}", :left, 12, :black))
            savefig(joinpath(figures_path, "standard_map_eps$(eps)"*RPC_name*".png"))
            savefig(joinpath(figures_path, "standard_map_eps$(eps)"*RPC_name*".svg"))
            savefig(joinpath(figures_path, "standard_map_eps$(eps)"*RPC_name*".pdf"))

            @save save_path RPC
        end
    end
end


main()