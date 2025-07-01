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
circ = Shape(Plots.partialcircle(0, 2π))

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs



function main()
    # Parameters
    eps = 1.0 # Recurrence threshold
    
    # Output directories
    data_path = "/home/gabrielm/projects/RPMotifs/data/orbits_standard_map_k=2.5_x0_y0/orbits/"
    figures_path = "figures/orbits_standard_map_k=2.5_multi_IC/eps$(eps)/date_$(today())_PANEL" # New folder for panel plot

    mkpath(figures_path)
    
    # Load all .dat files in data_path
    data_files = filter(f -> endswith(f, ".dat"), readdir(data_path))
    println("Found $(length(data_files)) data files in $data_path")
    
    n_ics = length(data_files)
    Nf = size(readdlm(joinpath(data_path, data_files[1])), 1)
    
    time_series = zeros(n_ics, Nf, 2)
    for (i, f) in enumerate(data_files)
        # Assuming you've already printed file names, skipping for brevity
        time_series[i, :, :] = readdlm(joinpath(data_path, f))
    end
    
    # --- Panel Plot Setup ---
    di_range = 1:4
    dj_range = -1:4#1:4#
    
    # Calculate grid size for the plot layout
    n_rows = length(dj_range)
    n_cols = length(di_range)
    
    num_panels = n_rows * n_cols
    panel_labels = map(i -> "($(string('a' + i - 1)))", 1:min(num_panels, 26))

    # Initialize the panel plot with the correct layout
    # The `layout` grid respects the order of plotting commands
    panel_plot = plot(layout = grid(n_rows, n_cols, widths=[fill(0.66/(n_cols-1), n_cols-1)..., 0.34]), 
                      size = (220 * n_cols, 180 * n_rows), # Start size, may need tuning
                      dpi = 200,
                      link = :both,
                      framestyle = :box,
                    top_margin = 2Plots.mm, 
                    )
    # --- Loop to generate data and populate panels ---
    
    # --- Loop to generate data and populate panels ---
    for (row_idx, dj) in enumerate(reverse(dj_range)) # Get both index and value
        for (col_idx, di) in enumerate(di_range)
            
            p_idx = (row_idx - 1) * (n_cols) + col_idx # Calculate linear index manually
            if di == 0 && dj == 0
                plot!(panel_plot, subplot=p_idx, framestyle=:none)
                continue
            end
            
            # --- Data Caching (Good practice, kept from original) ---
            save_path = joinpath(data_path, "RPC_eps$(eps)_di$(di)_dj$(dj).jld2")
            RPC = if isfile(save_path)
                println("Loading cache for (di=$(di), dj=$(dj)) from: $save_path")
                jldopen(save_path, "r") do file
                    read(file, "RPC")
                end
            else
                println("Calculating RPC for (di=$(di), dj=$(dj))...")
                temp_RPC = zeros(n_ics, Nf)
                for ic_idx in 1:n_ics
                    RP = custom_recurrence_plot(time_series[ic_idx, :, :], eps; periodic=true, period=2π)
                    temp_RPC[ic_idx, :] = local_morans_I(RP; 
                                                        weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), 
                                                        Δi_range = di:di, 
                                                        Δj_range = dj:dj)
                end
                @save save_path RPC=temp_RPC
                temp_RPC
            end
            
            # --- Configure attributes for the current subplot ---
            is_top= (dj == dj_range[end])
            # Determine if this subplot is on the bottom row
            is_bottom = (dj == dj_range[1])
            # Determine if this subplot is on the leftmost column
            is_leftmost = (di == di_range[1])
            # Determine if this subplot is on the rightmost column (for colorbar)
            is_rightmost = (di == di_range[end])

            # Axes and Ticks
            # Show xticks and xlabel only on the bottom row
            show_xticks = is_bottom ? ([0, π, 2π], [L"0", L"π", L"2π"]) : ([], [])
            show_xlabel = is_bottom ? L"x" : ""
            # Show yticks and ylabel only on the leftmost column
            show_yticks = is_leftmost ? ([0, π, 2π], [L"0", L"π", L"2π"]) : ([], [])
            show_ylabel = is_leftmost ? L"y" : ""

            # Colorbar: show only on the rightmost column plots
            show_colorbar = is_rightmost

            # --- Plotting the Scatter Plot for the current panel ---
            # Loop over initial conditions and plot on the same subplot
            for ic_idx = 1:n_ics
                scatter!(panel_plot, subplot = p_idx,
                    time_series[ic_idx, 5:end-5, 1], time_series[ic_idx, 5:end-5, 2], 
                    marker_z = RPC[ic_idx, 5:end-5],
                    color = :vik,
                    markershape=circ,
                    strokewidth=0,
                    markerstrokealpha=0, 
                    ms = 0.8,
                    alpha = 0.8,
                )
            end

            # --- Applying subplot-specific attributes ---
            plot!(panel_plot, subplot = p_idx,
                xlabel = show_xlabel,
                ylabel = show_ylabel,
                xticks = show_xticks,
                yticks = show_yticks,
                colorbar = show_colorbar,
                colorbar_title = show_colorbar ? " lRPC " : "",
                clims = (-1.0, 1.0),
                top_margin = is_top ? 2Plots.mm : 0Plots.mm, # Make panels touch
                widen = false
            )
            
            # Add annotation for the motif
            # The coordinates for annotate! are relative to the subplot axes
            label = panel_labels[p_idx]

            annotate!(panel_plot, subplot=p_idx, 
                      0.0 * 2π, 1.05 * 2π, # position in top-left corner
                      text(label * L" \ w_{Δi, Δj} = δ_{Δi, %$di} δ_{Δj, %$dj}", :left, 8, :black))
            
            p_idx += 1
        end
    end    


    # --- Save the final panel plot ---
    final_figure_path = joinpath(figures_path, "standard_map_panel_plot_eps$(eps)_$(di_range[1])_$(di_range[end])_$(dj_range[1])_$(dj_range[end])")
    println("Saving final panel plot to $(final_figure_path)...")
    savefig(panel_plot, final_figure_path * ".png")
    savefig(panel_plot, final_figure_path * ".svg")
#savefig(panel_plot, final_figure_path * ".pdf")

    println("Done.")
end

main()