# transition_time.jl

using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions
using Plots, Colors
using DelayEmbeddings
using JLD2
using LaTeXStrings
using Statistics # Added for mean()

gr() 
default(fontfamily="Computer Modern", linewidth=2, label=nothing, grid=false, framestyle=:box)

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs

# --- NEW HELPER FUNCTION ---
"""
Finds indices where the time series crosses a `threshold` value from below.
Returns a vector of indices.
"""
function find_crossings(ts::AbstractVector, threshold::Real; direction=:near)
    crossings = Int[]
    for i in 2:length(ts)
        if direction == :from_below
            if ts[i-1] < threshold && ts[i] >= threshold
                push!(crossings, i)
            end
        elseif direction == :from_above
             if ts[i-1] > threshold && ts[i] <= threshold
                push!(crossings, i)
            end
        elseif direction == :near
            if abs(ts[i] - threshold) < 1e-2 # Adjust threshold for near crossings
                push!(crossings, i)
            end
        else
            error("Invalid direction specified. Use :from_below, :from_above, or :near.")
        end
    end
    return crossings
end


function main()
    # Parameters
    Nf = 600
    Nf_long = 20000 # --- NEW: Length for the auxiliary long time series
    segment_len = 20 # --- NEW: Length of each segment to plot for overlays
    
    # Systems to analyze
    systems = [
        ("GWN", nothing, 1, 0.5),
        ("AR (0.90)", 0.9, 1, 0.5), # Using 0.99 for stronger correlation
        ("Logistic Map", 4.0, 1, 3/4),
        ("Sine", 0.01, [1], 1),
    ]
    
    n_systems = length(systems)

    # Output directories
    figures_path = "figures/time_series_panel_$(today())/Nf$(Nf)/"
    mkpath(figures_path)
        
    letters_annotation = ["(a)"; "(b)"; "(c)"; "(d)"; "(e)"; "(f)"; "(g)"; "(h)"; "(i)"; "(j)"; "(k)"; "(l)"; "(m)"; "(n)"; "(o)"; "(p)"; "(q)"; "(r)"; "(s)"]
   
    # --- MODIFIED: Layout is now n_systems x 2 grid ---
    plt = plot(layout = grid(n_systems, 2),
                size = (2000, 220 * n_systems), # Adjusted size for 2 columns
                dpi = 300,
                widen=true,
                frame_style = :box)

    for (i, system_tuple) in enumerate(systems)
        system_name, params, component,  μ = system_tuple
        
        # --- Generate both a short and a long time series ---
        time_series_short = Array(generate_time_series(system_name, params, Nf, component))
        time_series_long = Array(generate_time_series(system_name, params, Nf_long, component))
        # --- Normalize the time series between 0 and 1 ---
        time_series_short = (time_series_short .- minimum(time_series_short)) ./ (maximum(time_series_short) - minimum(time_series_short))
        time_series_long = (time_series_long .- minimum(time_series_long)) ./ (maximum(time_series_long) - minimum(time_series_long))
        # --- Calculate the mean ---
       
        
        μ_long = μ #mean(time_series_long)

        # === PANEL 1: Original Time Series Plot ===
        plot_idx_1 = 2*i - 1 # Subplot index for the first panel
        
        plot!(plt, subplot=plot_idx_1,
            1:Nf,
            time_series_short,
            lc=:black,
            xlabel = "Time",
            ylabel = "Amplitude",
            title = letters_annotation[plot_idx_1] * "  " * system_name,
            titlelocation = :left, titlefontsize=12,
            legend = false,
            left_margin = 5*Plots.mm,
        )
        
        # --- NEW: Add horizontal line for the mean ---
        hline!(plt, subplot=plot_idx_1, [μ], ls=:dash, lc=:red, label="Mean")

        # === PANEL 2: Overlapping Mean-Crossing Segments ===
        plot_idx_2 = 2*i # Subplot index for the second panel
        
        # Find all crossing points in the long time series
        crossings = find_crossings(time_series_long[:,1], μ_long, direction=:from_below)
        
        # Plot the first segment to set up the plot
        if !isempty(crossings)
            first_crossing = crossings[1]
            if first_crossing > segment_len/2 && first_crossing < Nf_long - segment_len/2
                segment = time_series_long[Int(first_crossing - segment_len/2):Int(first_crossing + segment_len/2 - 1)]
                plot!(plt, subplot=plot_idx_2,
                    -segment_len/2 : segment_len/2 - 1, # Centered time axis
                    segment .- μ_long, # Center amplitude at 0
                    lc=:black, alpha=0.1, label=""
                )
            end
        end

        # Overlay all other segments
        for k in 2:length(crossings)
            crossing_idx = crossings[k]
            # Ensure the segment is fully within the time series bounds
            if crossing_idx > segment_len/2 && crossing_idx < Nf_long - segment_len/2
                segment = time_series_long[Int(crossing_idx - segment_len/2):Int(crossing_idx + segment_len/2 - 1)]
                plot!(plt, subplot=plot_idx_2,
                    -segment_len/2 : segment_len/2 - 1, # Centered time axis
                    segment .- μ_long, # Center amplitude at 0
                    lc=:black, alpha=0.1, label="" # High transparency
                )
            end
        end
        
        # --- NEW: Add horizontal line at zero for reference ---
        hline!(plt, subplot=plot_idx_2, [0], ls=:dash, lc=:red, label="")
        
        # --- Style the second panel ---
        plot!(plt, subplot=plot_idx_2,
            title=letters_annotation[plot_idx_2] * "  Mean-Crossing Segments",
            titlelocation=:left, titlefontsize=12,
            xlabel="Time relative to crossing",
            ylabel="Amplitude - Mean",
            left_margin=5Plots.mm,
            right_margin=5*Plots.mm,
        )
    end

    savefig(plt, joinpath(figures_path, "time_series_and_crossings_panel.png"))
    #savefig(plt, joinpath(figures_path, "time_series_and_crossings_panel.pdf"))

end


main()