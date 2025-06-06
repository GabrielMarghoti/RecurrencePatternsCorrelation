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
    Nf = 2000
    
    eps = 0.02

    r_resol = 300
    rs = range(2.9, 4, length = r_resol) 
    
    # Output directories
    data_path    = "data/Logistic_bifurcation_local_$(today())/Nf$(Nf)_r_resol$(r_resol)"
    figures_path = "figures/Logistic_bifurcation_local_$(today())/Nf$(Nf)_r_resol$(r_resol)"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component

    time_series  = zeros(Nf, r_resol) 

    Morans_I_diag  = zeros(Nf, r_resol) 

    Morans_I_anti_diag = zeros(Nf, r_resol) 

    Morans_I_vert_line = zeros(Nf, r_resol) 
    
    Morans_I_4sides = zeros(Nf, r_resol) 

    Morans_I_8sides = zeros(Nf, r_resol) 


    Threads.@threads for i = 1:r_resol
        r = rs[i]
        time_series[:, i] = generate_time_series("Logistic", r, Nf, 1) # Generate time series

        RP = Matrix(RecurrenceMatrix(StateSpaceSet(time_series[:, i]), eps; metric = Euclidean(), parallel = true))
    
        Morans_I_diag[:, i] = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == Δj ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)

        Morans_I_anti_diag[:, i] = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == -Δj ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)

        Morans_I_vert_line[:, i] = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δj == 0 ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)

        Morans_I_4sides[:, i] = local_morans_I(RP; weight_function = (Δi, Δj) -> (abs(Δi) + abs(Δj) == 1 ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)

        Morans_I_8sides[:, i] = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi in [-1, 0, 1] && Δj in [-1, 0, 1] ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)
        
    end
    circ = Shape(Plots.partialcircle(0, 2π))
    # Plot bifurcation diagram with Moran's I (Diagonal) as colors
    for i in 1:Nf
        scatter!(repeat(rs, inner=1), time_series[i, :], 
             marker_z=Morans_I_diag[i, :], 
             color=:viridis, label="", ms=1.5, alpha=0.8, 
             strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ, 
             frame_style=:box, grid=false, size=(800,600), dpi=200)
    end
    xlabel!("r")
    ylabel!("x")
    title!("Bifurcation Diagram with Moran's I (Diagonal)")
    savefig(joinpath(figures_path, "bifurcation_diagram_diag_eps$(eps).png"))

    # Plot bifurcation diagram with Moran's I (Anti-Diagonal) as colors
    for i in 1:Nf
        scatter!(repeat(rs, inner=1), time_series[i, :], 
                 marker_z=Morans_I_anti_diag[i, :], 
                 color=:viridis, label="", ms=1.5, alpha=0.8,
                 strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ, 
                 frame_style=:box, grid=false, size=(800,600), dpi=200)
    end
    xlabel!("r")
    ylabel!("x")
    title!("Bifurcation Diagram with Moran's I (Anti-Diagonal)")
    savefig(joinpath(figures_path, "bifurcation_diagram_anti_diag_eps$(eps).png"))

    # Plot bifurcation diagram with Moran's I (Vertical Line) as colors
    for i in 1:Nf
        scatter!(repeat(rs, inner=1), time_series[i, :], 
                 marker_z=Morans_I_vert_line[i, :], 
                 color=:viridis, label="", ms=1.5, alpha=0.8,
                 strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ, 
                 frame_style=:box, grid=false, size=(800,600), dpi=200)
    end
    xlabel!("r")
    ylabel!("x")
    title!("Bifurcation Diagram with Moran's I (Vertical Line)")
    savefig(joinpath(figures_path, "bifurcation_diagram_vert_line_eps$(eps).png"))

    # Plot bifurcation diagram with Moran's I (4 Sides) as colors
    for i in 1:Nf
        scatter!(repeat(rs, inner=1), time_series[i, :], 
                 marker_z=Morans_I_4sides[i, :], 
                 color=:viridis, label="", ms=1.5, alpha=0.8,
                 strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ, 
                 frame_style=:box, grid=false, size=(800,600), dpi=200)
    end
    xlabel!("r")
    ylabel!("x")
    title!("Bifurcation Diagram with Moran's I (4 Sides)")
    savefig(joinpath(figures_path, "bifurcation_diagram_4sides_eps$(eps).png"))

    # Plot bifurcation diagram with Moran's I (8 Sides) as colors
    for i in 1:Nf
        scatter!(repeat(rs, inner=1), time_series[i, :], 
                 marker_z=Morans_I_8sides[i, :], 
                 color=:viridis, label="", ms=1.5, alpha=0.8,
                 strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ, 
                 frame_style=:box, grid=false, size=(800,600), dpi=200)
    end
    xlabel!("r")
    ylabel!("x")
    title!("Bifurcation Diagram with Moran's I (8 Sides)")
    savefig(joinpath(figures_path, "bifurcation_diagram_8sides_eps$(eps).png"))
    
end


main()