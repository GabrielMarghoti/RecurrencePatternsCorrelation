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
    Nf =5000
    
    eps = 0.01

    r_resol = 200
    rs = range(2.95, 4, length = r_resol) 
    
    di_range = -0:3

    dj_range=-3:3


    # Output directories
    data_path    = "data/Logistic_bifurcation_local_basis_panels/Nf$(Nf)_r_resol$(r_resol)_eps$(eps)_dj$(dj_range)_di$(di_range)"
    figures_path = "figures/Logistic_bifurcation_local_basis_panels_$(today())/Nf$(Nf)_r_resol$(r_resol)_eps$(eps)_dj$(dj_range)_di$(di_range)"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component



    
    # Calculate grid size for the plot layout
    n_cols = length(di_range)
    n_rows = length(dj_range)

    num_panels = n_rows*n_cols
    panel_labels = map(i -> L"($(string('a' + i - 1)))", 1:(num_panels))


    lRPC = zeros(Nf, r_resol, n_cols, n_rows) 
    time_series  = zeros(Nf, r_resol) 

    # --- Data Caching (Good practice, kept from original) ---
    save_path = joinpath(data_path, "lRPC.jld2")
    if isfile(save_path)
        @load save_path rs time_series lRPC # Load the results if they exist
    else
        
        Threads.@threads for i = 1:r_resol
            r = rs[i]
            time_series[:, i] = generate_time_series("Logistic", r, Nf, 1) # Generate time series

            RP = Matrix(RecurrenceMatrix(StateSpaceSet(time_series[:, i]), eps; metric = Euclidean(), parallel = true))
            
            for (col_idx, di) in enumerate(di_range) # Get both index and value
                for (row_idx, dj) in enumerate(dj_range) # Get both index and value
                    if di == 0 && dj == 0
                        # Skip the case where both di and dj are zero
                        lRPC[:, i, col_idx, row_idx] = zeros(Nf) # Fill with zeros
                        continue
                    end
                    rpc_values = RPC_local(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)
                    # Replace `nothing` with zero
                    lRPC[:, i, col_idx, row_idx] = map(x -> x === nothing ? 0.0 : x, rpc_values)
                end
            end
        end
        
        @save save_path rs time_series lRPC # Save the results
        
    end

    # Initialize the panel plot with the correct layout
    # The `layout` grid respects the order of plotting commands
    panel_plot = plot(layout = grid(n_rows, n_cols), 
                      size = (20+ 350*n_cols, 20+ 150 * n_rows), # Start size, may need tuning
                      dpi = 300,
                      link = :both,
                      framestyle = :box,
                    )
    p_idx=1
    for (row_idx, dj) in enumerate(dj_range) # Get both index and value
        for (col_idx, di) in enumerate(di_range) # Get both index and value
            for i in 1:r_resol
                
                scatter!(panel_plot, subplot=p_idx, 
                fill(rs[i], length(time_series[10:end-10, i])), time_series[10:end-10, i], 
                marker_z=lRPC[10:end-10, i, col_idx, row_idx], 
                color=:vik, label="", ms=1.1, alpha=1.0, 
                colorbar = false,
                strokewidth=0.0, markerstrokealpha=0, markershape=circ,
                )
            end
            # --- Applying subplot-specific attributes ---
            plot!(panel_plot, subplot = p_idx,
                xlabel = row_idx==n_rows ? L"r" : "",
                ylabel = col_idx==1 ? L"x_i" : "",
                #xticks = show_xticks,
                #yticks = show_yticks,
                colorbar_title=" \nlRPC",
                clims = (-1.0, 1.0),
                ylims=(-0.01,1.01),
                top_margin = row_idx==1 ? 0Plots.mm : -5Plots.mm, # Make panels touch
                right_margin = 2Plots.mm, # Make panels touch
                widen = false
            )
            

            annotate!(panel_plot, subplot=p_idx, 
                        3, 0.2, # position in top-left corner
                        text(panel_labels[p_idx] * L" \ w_{Δi, Δj} = δ_{Δi, %$di} δ_{Δj, %$dj}", :left, 10, :black))
            
            p_idx += 1
        end
    end
    savefig(joinpath(figures_path, "bifurcation_diagram_panels"*".png"))
    #savefig(joinpath(figures_path, "bifurcation_diagram_panels"*".svg"))
    #savefig(joinpath(figures_path, "bifurcation_diagram_panels"*".pdf"))
end


main()