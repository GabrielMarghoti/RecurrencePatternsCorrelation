# transition_time.jl

using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions
using Plots 

using JLD2

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs


function main()
    # Parameters
    Nf = 1200
    
    eps = 0.02

    r_resol = 200
    rs = range(2.9, 4, length = r_resol) 
    
    # Output directories
    data_path    = "data/Logistic_bifurcation_local_basis_$(today())/Nf$(Nf)_r_resol$(r_resol)"
    figures_path = "figures/Logistic_bifurcation_local_basis_$(today())/Nf$(Nf)_r_resol$(r_resol)"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component

    time_series  = zeros(Nf, r_resol) 



    for di=-1:1
        for dj=-1:1
            if di == 0 && dj == 0
                continue # Skip the case where both di and dj are zero
            end
            RPC_name = "RPC_di$(di)_dj$(dj)"

            RPC = zeros(Nf, r_resol) 

            Threads.@threads for i = 1:r_resol
                r = rs[i]
                time_series[:, i] = generate_time_series("Logistic", r, Nf, 1) # Generate time series

                RP = Matrix(RecurrenceMatrix(StateSpaceSet(time_series[:, i]), eps; metric = Euclidean(), parallel = true))
            
                RPC[:, i] = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)
            end
            circ = Shape(Plots.partialcircle(0, 2π))
            # Plot bifurcation diagram with Moran's I (Diagonal) as colors
            for i in 1:Nf
                scatter!(repeat(rs, inner=1), time_series[i, :], 
                    marker_z=RPC[i, :], 
                    color=:viridis, label="", ms=1.8, alpha=0.8, 
                    strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ, 
                    frame_style=:box, grid=false, size=(800,600), dpi=200)
            end
            xlabel!("r")
            ylabel!("x")
            title!("Bifurcation Diagram with RPC w($(di), $(dj)) = 1")
            savefig(joinpath(figures_path, "bifurcation_diagram_diag_eps$(eps)"*RPC_name*".png"))



        end
    end
   

end


main()