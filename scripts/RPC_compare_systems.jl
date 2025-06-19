# transition_time.jl

using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions
using Plots, Colors
using DelayEmbeddings
using JLD2
using LaTeXStrings

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs

function diag_marker(ratio=1.0)
    x = ratio*[-0.1, 1.0]
    y = ratio*[-0.1, 1.0]
    return Shape(x, y)
end


function anti_diag_marker(ratio=1.0)
    x = ratio*[-0.1, 1.0]
    y = ratio*[0.1, -1.0]
    return Shape(x, y)
end


function vert_marker(ratio=1.0)
    x = ratio*[0.0, 0.0]
    y = ratio*[-0.1, 1.0]
    return Shape(x, y)
end

function main()
    # Parameters
    Nf = 2000
    
    load_cache = true

    rr_resol = 100
    rrs = range(0.001, 0.999, rr_resol) #10 .^ range(-3, -0.001, rr_resol)
    rr_resol = length(rrs)
    
    # 3D autoregressive model connection matrix
    A = [0.35  0.2  0.25;
         0.2  0.45  0.2;
         0.25  0.5  0.2]
    
    # Systems to analyze
    systems = [
        #("Tipping point sde", [[0.1, 0.004], 0.1], 1),
        #("UWN", nothing, 1),
        ("GWN", nothing, 1),
        ("AR (0.8)", 0.8, 1),
        ("AR (0.99)", 0.99, 1),
       # ("NAR (0, 0.99)", [0.0; 0.99], 1),
        ("Logistic", 4.0, 1),
       # ("Logistic", 3.678, 1),
        ("Lorenz", [[10.0, 28.0, 8 / 3], 0.05], [1,2,3]),
       # ("Rossler", [[0.2, 0.2, 5.7], 0.5], [1,2,3]),
       # ("AR(2)", [[0.7, -0.2], 0.5], 1),  # AR(2) with noise variance
       # ("Lorenz (x)", [[10.0, 28.0, 8 / 3], 0.05], 1),
       # ("Lorenz (z)", [[10.0, 28.0, 8 / 3], 0.05], 3),
       # ("Logistic 3D", nothing, [3.711, 0.06], 1),
       # ("AR 0.3", nothing, 0.3, 1),
       # ("AR 0.8", nothing, 0.8, 1),
       # ("3D AR", A, 1),
       # ("Rossler (x)", [[0.2, 0.2, 5.7], 0.5], 1),
        ("Sine", 0.01, [1]),
        #("GARCH", nothing, 
        #[0.01,
        #[0.1, 0.05],  # ARCH(2)
        #[0.7, 0.2, 0.05]]  # GARCH(3)
        #, 1)  # ω, α, β
    ]
    
    n_systems = length(systems)

    # Output directories
    time_series_path = "data/time_series"
    data_path    = "data/RPC_compare_sys_global/Nf$(Nf)_rr_reso$(rr_resol)/"
    figures_path = "figures/RPC_compare_sys_global_$(today())/Nf$(Nf)_rr_reso$(rr_resol)/"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component

    det_results = zeros(length(systems), rr_resol) 

    lam_results = zeros(length(systems), rr_resol) 

    di_dj_tuples =[(0, 1), (1, -1), (1, 1), (0, 2), (2, -2), (2, 2), (0, 4), (4, -4), (4, 4)] 

    len_RPC = length(di_dj_tuples)
    didj_labels = [L" w_{\Delta i, \Delta j}=\delta _{\Delta i,%$(shift[1])} \delta_{\Delta j, %$(shift[2])}" for shift in di_dj_tuples]
    
    RPC = zeros(length(systems), rr_resol, len_RPC) # Initialize RPC array

    cache_path = joinpath(data_path, "data.jld2")

    if isfile(cache_path)
        @load cache_path rrs RPC systems di_dj_tuples didj_labels
    else
        for (i, system_tuple) in enumerate(systems)
            system_name, params, component = system_tuple
            
            println("Analyzing $system_name system...")

            system_path = figures_path*"$(system_name)_$(params)"
            mkpath(system_path)
            
            time_series = generate_time_series(system_name, params, Nf, component) # Generate time series

            if size(time_series, 2) == 1
                #println("   Performing Pecuzal embedding for 1D time series")
                embedded_time_series = pecuzal_embedding(time_series; max_cycles=7)[1]
            else
                embedded_time_series = time_series
            end
            
            file_path = data_path * "rr_resol$(rr_resol)_morans_I_$(system_name).jld2"
            #if isfile(file_path)
            #    @load  file_path rrs RPC_diag RPC_anti_diag RPC_vert_line RPC_4sides RPC_8sides  det_results lam_results
            for rr_idx in 1:rr_resol

                rr_fig_path = joinpath(system_path, "rr$(rrs[rr_idx])/")
                rr_data_path = joinpath(data_path, "rr$(rrs[rr_idx])/")
                mkpath(rr_data_path)

                RP = RecurrenceMatrix(StateSpaceSet(embedded_time_series), GlobalRecurrenceRate(rrs[rr_idx]); metric = Euclidean(), parallel = true)
                _rqa_sys = rqa(RP)

                det_results[i, rr_idx] = _rqa_sys[:DET]
                lam_results[i, rr_idx] = _rqa_sys[:LAM]
                for (RPC_idx, di_dj) in enumerate(di_dj_tuples)
                    di, dj = di_dj
                    RPC_name = "RPC_di$(di)_dj$(dj)"

                    plot_recurrence_matrix(RP, system_name, rr_fig_path; filename="recurrence_plot.png")
                    
                    RPC[i, rr_idx, RPC_idx] = morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)      

                end
            end

        end

        @save cache_path rrs RPC systems di_dj_tuples didj_labels
    end


    # 1. Group motifs by delay magnitude
    di_dj_tuples = [(0, 1), (1, -1), (1, 1), (0, 2), (2, -2), (2, 2), (0, 4), (4, -4), (4, 4)]
    delay_groups = [1:3, 4:6, 7:9] # Indices into di_dj_tuples for each delay magnitude
    n_delays = length(delay_groups)
    n_systems = length(systems)

    # Labels for the lines within each subplot
    motif_labels = [L"Vertical", L"Anti-diag.", L"Diagonal"]

    # Define colors and markers for the three motif types (Vertical, Anti-diag, Diagonal)
    # This will be repeated for each delay group.
    colors_per_motif = [:blue, :red, :green]
    markers_per_motif = [vert_marker(1.5), anti_diag_marker(1.5), diag_marker(1.5)]

    # 2. Define the new layout: n_systems rows, n_delays columns
    plt = plot(layout = grid(n_systems, n_delays),
            size = (400 * n_delays, 200 * n_systems),
            dpi = 300,
            link = :y) # Link y-axes for easier comparison

    letters_annotation = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    # 3. Create the nested plotting loop
    p_idx = 1
    for i in 1:n_systems # Loop over rows (systems)
        for j in 1:n_delays # Loop over columns (delay magnitudes)
            
            is_top_row = (i == 1)
            is_bottom_row = (i == n_systems)
            is_leftmost_col = (j == 1)
            
            # Get the indices for the current delay group
            current_group_indices = delay_groups[j]
            
            # Plot the 3 lines for the current system and delay group
            for (k_motif, k_rpc) in enumerate(current_group_indices)
                plot!(plt, subplot = p_idx,
                    rrs, RPC[i, :, k_rpc],
                    label = "", # We will create a custom legend
                    color = colors_per_motif[k_motif],
                    marker = markers_per_motif[k_motif],
                    markercolor = colors_per_motif[k_motif],
                    markersize = 2,
                    strokewidth = 1.5,
                    alpha = 0.9)
            end
            
            # Add a title for each column, only on the top row
            column_title = is_top_row ? "Delay $(2^(j-1))" : ""
            
            # Add labels only to the edges
            xlabel = is_bottom_row ? L"Recurrence \ Rate \ (rr)" : ""
            ylabel = is_leftmost_col ? "RPC" : ""
            
            # Get system name for row annotation
            system_name = systems[i][1]

            # Apply attributes to the subplot
            plot!(plt, subplot = p_idx,
                title = column_title,
                titlefontsize = 10,
                xlabel = xlabel,
                ylabel = ylabel,
                xtickfontsize = 8,
                ytickfontsize = 8,
                grid = false,
                ylims = (-0.4, 1.1),
                framestyle = :box)

            # Annotate with system name on the first column
            if is_leftmost_col
                annotate!(plt, subplot=p_idx,
                        -0.3, 1.3, text(letters_annotation[i] * " " * system_name, :left, 10))
            end

            p_idx += 1
        end
    end

    # --- Create a custom legend at the top of the entire plot ---
    # This is a bit of a "hack" in Plots.jl: create invisible series with the desired labels.
    # We add them to the first subplot and use the :outertopright legend position.
    for k in 1:length(motif_labels)
        plot!(plt, subplot=1, [NaN], [NaN], # Plot nothing
            label=motif_labels[k],
            color=colors_per_motif[k],
            marker=markers_per_motif[k],
            markercolor=colors_per_motif[k],
            lw=1.5
        )
    end
    plot!(plt, subplot=1, legend=:outertop, legendcolumns=3, legendfontsize=10, framestyle=:none)
    plot!(plt, top_margin = 15Plots.mm) # Add space for the legend and titles


    # Save the final plot
    mkpath(figures_path)
    savefig(plt, joinpath(figures_path, "systems_by_delay_columns.png"))
    savefig(plt, joinpath(figures_path, "systems_by_delay_columns.pdf"))

    #= Scatter plot for :DET
    plt_det = scatter(det_results', RPC_diag',
                    label = reshape([dat[1] for dat in systems], 1, length(systems)),
                    xlabel = "DET", ylabel = "C_diag", 
                    title = "Moran's I Diag vs DET", 
                    legend = :outerright, size = (600, 400), dpi = 200, grid = true, frame_style = :box)
    savefig(plt_det, joinpath(figures_path, "all_systems_RPC_vs_DET.png"))

    # Scatter plot for :LAM
    plt_lam = scatter(lam_results', RPC_vert_line',
                    label = reshape([dat[1] for dat in systems], 1, length(systems)),
                    xlabel = "LAM", ylabel = "C_vert_line", 
                    title = "Moran's I vert. line vs LAM", 
                    legend = :outerright, size = (600, 400), dpi = 200, grid = true, frame_style = :box)
    savefig(plt_lam, joinpath(figures_path, "all_systems_RPC_vs_LAM.png"))
    =#
end


main()