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
using StatsBase # --- NEW --- Added for autocorrelation

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs

gr() 
default(fontfamily="Computer Modern", linewidth=2, label=nothing, grid=false, framestyle=:box)
circ = Shape(Plots.partialcircle(0, 2π))

function main()
    # Parameters
    rr = 0.01
    Nf = 6000
    dt = 0.02

    # --- MODIFIED ---: Make time_shifts match a max lag for autocorrelation
    max_lag_steps = 240
    time_shifts = 0:1:max_lag_steps
    tshifts_len = length(time_shifts)

    # Define the dictionary: Int → Vector of (Float64, String)
    upo_data = Dict{String, Vector{Tuple{Float64,String}}}(
                    "T_2/2" => [(0.6854, "L or R")],
                    "2T_2/2" => [(1.32, "L or R")],
                    "T_2" => [(1.55865, "LR")],
                    "3T_2/2" => [(1.97, "L or R")],
                    "T_3" => [(2.30591, "LLR")],
                    "T_4" => [(3.02358, "LLLR"), (3.08428, "LLRR")],
                    "T_5" => [(3.72564, "LLLLR"), (3.82025, "LLLRR"), (3.86953, "LLRLR")],
                    "T_6" => [
                        (4.41776, "LLLLLR"),
                        (4.53410, "LLLLRR"),
                        (4.56631, "LLLRRR"),
                        (4.59381, "LLLRLR"),
                        (4.63714, "LLRLRR")
                    ],
                    "T_7" => [
                        (5.10303, "LLLLLLR"),
                        (5.23419, "LLLLLRR"),
                        (5.28634, "LLLLLRRR"),
                        (5.30120, "LLLLRLR"),
                        (5.33091, "LLLRLLR"),
                        (5.36988, "LLLRLRR"),
                        (5.37052, "LLRRLR")
                    ],
                    "T_8" => [
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
        "T_2/2" => :black,
        "2T_2/2" => :green,
        "3T_2/2" => :blue,
        "T_2" => :red,
        "T_3" => :cyan,
        "T_4" => :orange,
        "T_5" => :purple,
        "T_6" => :magenta,
    )

    upo_steps = Dict{String, Vector{Tuple{Int,String}}}()
    for (p, _) in upo_colors
        entries = upo_data[p]
        upo_steps[p] = [(round(Int, T/dt), s) for (T, s) in entries]
    end

    n_local_lags = length(upo_steps)
    panel_labels = map(i -> "($(string('a' + i - 1)))", 1:min(n_local_lags+2, 26)) # --- MODIFIED --- Increased index for new panel.

    # Output directories
    data_path    = "data/lorenz_Nf$(Nf)_rr$(rr)_dt$(dt)_RPC_autocorr_time_shift$(time_shifts[end])_$(length(time_shifts))/"
    figures_path = "figures/lorenz_RPC_x_time_shift_$(today())/Nf$(Nf)_rr$(rr)_dt$(dt)_RPC_autocorr_time_shift$(time_shifts[end])_$(length(time_shifts))/"

    mkpath(data_path)
    mkpath(figures_path)
    
    time_series = generate_time_series("Lorenz", [[10.0, 28.0, 8 / 3], dt], Nf, [1,2,3])
    RP = Matrix(RecurrenceMatrix(StateSpaceSet(time_series), GlobalRecurrenceRate(rr); metric = Euclidean(), parallel = true))
    plot_recurrence_matrix(RP[1:1000,1:1000], "Lorenz System", figures_path; filename="recurrence_plot.png")

    # --- NEW ---: Calculate autocorrelation of the x-component
    ac = autocor(time_series[:,1], time_shifts)

    # --- Data Caching (Good practice, kept from original) ---
    save_path = joinpath(data_path, "RPC.jld2")
    
    RPC = zeros(tshifts_len)
    lRPC = Dict{String, Vector{Float64}}()

    if isfile(save_path)
        @load save_path RPC lRPC
    else
        Threads.@threads for dj_idx = 1:tshifts_len
            di = 0
            dj = time_shifts[dj_idx]
            RPC[dj_idx] = morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)
        end
        
        for (p_j_name, entries) in upo_steps
            p_j_code = upo_steps[p_j_name][1][2]
            dj = upo_steps[p_j_name][1][1]
            di = 0
            lRPC[p_j_name] = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)
        end
        @save save_path RPC lRPC
    end

    # --- MODIFIED ---: Updated layout and sizing for clarity
    nrows = 3 
    ncols = 2
    
    # New layout: A single wide plot on top, then the grid
    l = @layout [
        a{0.3h} # Top plot takes 30% of the height
        grid(nrows, ncols)
    ]
    
    panel_plot = plot(
        layout = l,
        size = (600, 750), # Increased height for better spacing
        dpi = 400,
        fontfamily = "Computer Modern",
    )

    # --- MODIFIED ---: Create the TOP plot (Subplot 1) with dual y-axes
    # Plot RPC on the left Y-axis
    plot!(panel_plot, subplot=1,
          time_shifts * dt, RPC,
          label = "RPC", lc = :black, lw = 2,
          xlabel = "", # Remove xlabel from top plot to avoid overlap
          ylabel = "RPC",
          title = "(a) Global Measures", titlelocation = :left, titlefontsize = 10,
          left_margin=5Plots.mm,
          framestyle = :box,
          legend = :topright, # Add legend
          xtickfontsize=0 # Hide x-ticks to avoid clutter with bottom plots
    )
    
    # Activate the right Y-axis and plot autocorrelation
    plot!(panel_plot, subplot=1,
          time_shifts * dt, ac,
          label = "Autocorrelation", lc = :blue, ls = :dash, lw = 2,
          ylabel = "Autocorrelation"
    )

    # Add vertical lines for UPO periods to the top plot (referencing the main subplot)
    for (key, color) in upo_colors
        period_times = upo_data[key]
        for k in period_times
            vline!(panel_plot, subplot=1, [k[1]],
                label = "",
                ls = :dash, lw=1.2, lc = color, alpha = 0.7)
        end
    end

    # --- MODIFIED ---: Create the BOTTOM panels
    plot_order = ["T_2/2", "2T_2/2", "T_2", "3T_2/2", "T_3"]
    
    # We now have 6 total panels to fill
    # Let's adjust has_xlabel and has_ylabel for a 3x2 grid below the main plot
    # The panels start from subplot index 2
    # Grid positions:
    # 2 | 3
    # 4 | 5
    # 6 | 7 (if it exists)
    has_xlabel = [false, false, false, false, true, true]
    has_ylabel = [true, false, true, false, true, false]

    p_idx = 2
    for p_j_name in plot_order
        if p_idx > (nrows * ncols + 1); break; end

        dj_steps = upo_steps[p_j_name][1][1]
        dj_time = upo_data[p_j_name][1][1]
        panel_label = panel_labels[p_idx]

        base_color = upo_colors[p_j_name]
        color_grad = cgrad([RGBA(0.95, 0.95, 0.95, 0.5), base_color])
        
        scatter!(panel_plot, subplot = p_idx,
            time_series[400:end-400, 1], time_series[400:end-400, 3],
            marker_z = lRPC[p_j_name][400:end-400],
            color = color_grad,
            ms = 1.0, alpha = 0.9, strokewidth = 0, markershape = circ,
            markerstrokealpha = 0,
            xlabel = has_xlabel[p_idx-1] ? L"x" : "",
            ylabel = has_ylabel[p_idx-1] ? L"z" : "",
            framestyle = :box, grid = false,
            colorbar = true, colorbar_title = "",
            label="",
            title = "$panel_label  Δt = $(round(dj_time, digits=2))",
            titlelocation = :left, titlefontsize = 10,
            clims = (minimum(lRPC[p_j_name][400:end-400]), maximum(lRPC[p_j_name][400:end-400]))
        )
        
        p_idx += 1
    end

    # --- NEW ---: Add a shared x-axis label for the bottom grid
    # This is a bit of a workaround, we annotate the plot space below the figure
    plot!(panel_plot, 
        xlabel = "\n" * L"x", # Add newlines to push it down
        bottom_margin=10Plots.mm, # Give it space
        grid=false, framestyle=:none, # Make it invisible
        inset = (1, bbox(0.5, -0.1, 0.1, 0.1)), subplot=2 # Place it
    )

    savefig(panel_plot, joinpath(figures_path, "lorenz_upo_analysis_panel.png"))
    savefig(panel_plot, joinpath(figures_path, "lorenz_upo_analysis_panel.pdf"))
    
end

main()