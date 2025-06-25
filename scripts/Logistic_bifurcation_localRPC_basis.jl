# transition_time.jl

using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions
using Plots 
using LaTeXStrings
using DelimitedFiles
using JLD2

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs

gr() 
default(fontfamily="Computer Modern", linewidth=2, label=nothing, grid=false, framestyle=:box)
circ = Shape(Plots.partialcircle(0, 2π))


function main()
    # Parameters
    Nf = 5000
    
    eps = 0.01

    r_resol = 400
    rs = range(2.95, 4, length = r_resol) 
    
    di = 0

    dj_range=1:4  


    # Output directories
    data_path    = "data/Logistic_bifurcation_local_basis/Nf$(Nf)_r_resol$(r_resol)_eps$(eps)_dj$(dj_range)_di$(di)"
    figures_path = "figures/Logistic_bifurcation_local_basis_$(today())/Nf$(Nf)_r_resol$(r_resol)_eps$(eps)_dj$(dj_range)_di$(di)"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component



    
    # Calculate grid size for the plot layout
    n_rows = length(dj_range)

    num_panels = n_rows
    panel_labels = map(i -> "($(string('a' + i - 1)))", 1:min(num_panels, 26))


    RPC = zeros(Nf, r_resol, n_rows) 
    time_series  = zeros(Nf, r_resol) 

    # --- Data Caching (Good practice, kept from original) ---
    save_path = joinpath(data_path, "RPC_eps$(eps)_di$(di)_dj$(dj_range).jld2")
    if isfile(save_path)
        @load save_path rs time_series RPC # Load the results if they exist
    else
        
        Threads.@threads for i = 1:r_resol
            r = rs[i]
            time_series[:, i] = generate_time_series("Logistic", r, Nf, 1) # Generate time series

            RP = Matrix(RecurrenceMatrix(StateSpaceSet(time_series[:, i]), eps; metric = Euclidean(), parallel = true))
            
            for (row_idx, dj) in enumerate(dj_range) # Get both index and value
                rpc_values = RPC_local(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)
                # Replace `nothing` with zero
                RPC[:, i, row_idx] = map(x -> x === nothing ? 0.0 : x, rpc_values)
            end
        end
        
        @save save_path rs time_series RPC # Save the results
        
    end

    # Initialize the panel plot with the correct layout
    # The `layout` grid respects the order of plotting commands
    panel_plot = plot(layout = grid(n_rows, 1), 
                      size = (500, 120 * n_rows), # Start size, may need tuning
                      dpi = 300,
                      link = :both,
                      framestyle = :box,
                    )
    for (row_idx, dj) in enumerate(dj_range) # Get both index and value
        for i in 1:r_resol
            
            scatter!(panel_plot, subplot=row_idx, 
            fill(rs[i], length(time_series[10:end-10, i])), time_series[10:end-10, i], 
            marker_z=RPC[10:end-10, i, row_idx], 
            color=:vik, label="", ms=1.1, alpha=1.0, 
            strokewidth=0.0, markerstrokealpha=0, markershape=circ,
            )
        end
        # --- Applying subplot-specific attributes ---
        plot!(panel_plot, subplot = row_idx,
            xlabel = row_idx==n_rows ? L"r" : "",
            ylabel = L"x",
            #xticks = show_xticks,
            #yticks = show_yticks,
            colorbar_title=" \nlRPC",
            clims = (-1.0, 1.0),
            ylims=(-0.01,1.01),
            top_margin = row_idx==1 ? 0Plots.mm : -5Plots.mm, # Make panels touch
            right_margin = 4Plots.mm, # Make panels touch
            widen = false
        )
        

        annotate!(panel_plot, subplot=row_idx, 
                    3, 0.2, # position in top-left corner
                    text(panel_labels[row_idx] * L" \ w_{Δi, Δj} = δ_{Δi, %$di} δ_{Δj, %$dj}", :left, 10, :black))
        
    end

    savefig(joinpath(figures_path, "bifurcation_diagram_panels"*".png"))
    savefig(joinpath(figures_path, "bifurcation_diagram_panels"*".svg"))
    savefig(joinpath(figures_path, "bifurcation_diagram_panels"*".pdf"))
end


main()