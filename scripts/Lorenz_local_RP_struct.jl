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
    Nf = 10000
    
    rr = 0.02

    # Output directories
    data_path    = "data/Lorenz_struct_local_$(today())/Nf$(Nf)"
    figures_path = "figures/Lorenz_struct_local_$(today())/Nf$(Nf)"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component

    Morans_I_diag  = zeros(Nf) 

    Morans_I_anti_diag = zeros(Nf) 

    Morans_I_vert_line = zeros(Nf) 
    
    Morans_I_4sides = zeros(Nf) 

    Morans_I_8sides = zeros(Nf) 

    time_series= generate_time_series("Lorenz",  [[10.0, 28.0, 8 / 3], 0.02], Nf, [1,2,3]) # Generate time series

    RP = Matrix(RecurrenceMatrix(StateSpaceSet(time_series), GlobalRecurrenceRate(rr); metric = Euclidean(), parallel = true))
    
    Morans_I_diag = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == Δj ? 1 : 0), Δi_range = -5:5, Δj_range = -5:5)

    Morans_I_anti_diag = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == -Δj ? 1 : 0), Δi_range = -5:5, Δj_range = -5:5)

    Morans_I_vert_line = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δj == 0 ? 1 : 0), Δi_range = -5:5, Δj_range = -5:5)

    Morans_I_4sides = local_morans_I(RP; weight_function = (Δi, Δj) -> (abs(Δi) + abs(Δj) == 1 ? 1 : 0), Δi_range = -5:5, Δj_range = -5:5)

    Morans_I_8sides = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi in [-1, 0, 1] && Δj in [-1, 0, 1] ? 1 : 0), Δi_range = -5:5, Δj_range = -5:5)

    circ = Shape(Plots.partialcircle(0, 2π))



    # 3D Lorenz trajectory colored scatter
    scatter3d(time_series[:, 1], time_series[:, 2], time_series[:, 3], 
              marker_z=Morans_I_diag, color=:viridis, label="", ms=3, alpha=0.8,
              strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
              xlabel="x", ylabel="y", zlabel="z", title="3D Lorenz Trajectory with Moran's I (Diagonal)",
              size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_3d_diag.png"))

    # X-Y projection
    plot(time_series[:, 1], time_series[:, 2], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
    scatter!(time_series[:, 1], time_series[:, 2], 
             marker_z=Morans_I_diag, color=:viridis, label="", ms=3, alpha=0.8,
             strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
             xlabel="x", ylabel="y", title="Lorenz Trajectory X-Y Projection with Moran's I (Diagonal)",
             size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_xy_diag.png"))

    # X-Z projection
    plot(time_series[:, 1], time_series[:, 3], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
    scatter!(time_series[:, 1], time_series[:, 3], 
            marker_z=Morans_I_diag, color=:viridis, label="",ms=3, alpha=0.8,
            strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
            xlabel="x", ylabel="z", title="Lorenz Trajectory X-Z Projection with Moran's I (Diagonal)",
            size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_xz_diag.png"))


    # 3D Lorenz trajectory colored scatter for Anti-Diagonal
    scatter3d(time_series[:, 1], time_series[:, 2], time_series[:, 3], 
              marker_z=Morans_I_anti_diag, color=:viridis, label="",ms=3, alpha=0.8,
              strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
              xlabel="x", ylabel="y", zlabel="z", title="3D Lorenz Trajectory with Moran's I (Anti-Diagonal)",
              size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_3d_anti_diag.png"))

    # X-Y projection for Anti-Diagonal
    plot(time_series[:, 1], time_series[:, 2], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
    scatter!(time_series[:, 1], time_series[:, 2], 
            marker_z=Morans_I_anti_diag, color=:viridis, label="",ms=3, alpha=0.8,
            strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
            xlabel="x", ylabel="y", title="Lorenz Trajectory X-Y Projection with Moran's I (Anti-Diagonal)",
            size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_xy_anti_diag.png"))

    # X-Z projection for Anti-Diagonal
    plot(time_series[:, 1], time_series[:, 3], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
    scatter!(time_series[:, 1], time_series[:, 3], 
            marker_z=Morans_I_anti_diag, color=:viridis, label="",ms=3, alpha=0.8,
            strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
            xlabel="x", ylabel="z", title="Lorenz Trajectory X-Z Projection with Moran's I (Anti-Diagonal)",
            size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_xz_anti_diag.png"))

    # 3D Lorenz trajectory colored scatter for Vertical Line
    scatter3d(time_series[:, 1], time_series[:, 2], time_series[:, 3], 
              marker_z=Morans_I_vert_line, color=:viridis, label="",ms=3, alpha=0.8,
              strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
              xlabel="x", ylabel="y", zlabel="z", title="3D Lorenz Trajectory with Moran's I (Vertical Line)",
              size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_3d_vert_line.png"))

    # X-Y projection for Vertical Line
    plot(time_series[:, 1], time_series[:, 2], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
    scatter!(time_series[:, 1], time_series[:, 2], 
            marker_z=Morans_I_vert_line, color=:viridis, label="",ms=3, alpha=0.8,
            strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
            xlabel="x", ylabel="y", title="Lorenz Trajectory X-Y Projection with Moran's I (Vertical Line)",
            size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_xy_vert_line.png"))

    # X-Z projection for Vertical Line
    plot(time_series[:, 1], time_series[:, 3], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
    scatter!(time_series[:, 1], time_series[:, 3], 
            marker_z=Morans_I_vert_line, color=:viridis, label="",ms=3, alpha=0.8,
            strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
            xlabel="x", ylabel="z", title="Lorenz Trajectory X-Z Projection with Moran's I (Vertical Line)",
            size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_xz_vert_line.png"))

    # 3D Lorenz trajectory colored scatter for 4-Sides
    scatter3d(time_series[:, 1], time_series[:, 2], time_series[:, 3], 
              marker_z=Morans_I_4sides, color=:viridis, label="",ms=3, alpha=0.8,
              strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
              xlabel="x", ylabel="y", zlabel="z", title="3D Lorenz Trajectory with Moran's I (4-Sides)",
              size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_3d_4sides.png"))

    # X-Y projection for 4-Sides
    plot(time_series[:, 1], time_series[:, 2], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
    scatter!(time_series[:, 1], time_series[:, 2], 
            marker_z=Morans_I_4sides, color=:viridis, label="",ms=3, alpha=0.8,
            strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
            xlabel="x", ylabel="y", title="Lorenz Trajectory X-Y Projection with Moran's I (4-Sides)",
            size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_xy_4sides.png"))

    # X-Z projection for 4-Sides
    plot(time_series[:, 1], time_series[:, 3], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
    scatter!(time_series[:, 1], time_series[:, 3], 
            marker_z=Morans_I_4sides, color=:viridis, label="",ms=3, alpha=0.8,
            strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
            xlabel="x", ylabel="z", title="Lorenz Trajectory X-Z Projection with Moran's I (4-Sides)",
            size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_xz_4sides.png"))

    # 3D Lorenz trajectory colored scatter for 8-Sides
    scatter3d(time_series[:, 1], time_series[:, 2], time_series[:, 3], 
              marker_z=Morans_I_8sides, color=:viridis, label="",ms=3, alpha=0.8,
              strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
              xlabel="x", ylabel="y", zlabel="z", title="3D Lorenz Trajectory with Moran's I (8-Sides)",
              size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_3d_8sides.png"))

    # X-Y projection for 8-Sides
    plot(time_series[:, 1], time_series[:, 2], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
    scatter!(time_series[:, 1], time_series[:, 2], 
            marker_z=Morans_I_8sides, color=:viridis, label="",ms=3, alpha=0.8,
            strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
            xlabel="x", ylabel="y", title="Lorenz Trajectory X-Y Projection with Moran's I (8-Sides)",
            size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_xy_8sides.png"))

    # X-Z projection for 8-Sides
    plot(time_series[:, 1], time_series[:, 3], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
    scatter!(time_series[:, 1], time_series[:, 3], 
            marker_z=Morans_I_8sides, color=:viridis, label="",ms=3, alpha=0.8,
            strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
            xlabel="x", ylabel="z", title="Lorenz Trajectory X-Z Projection with Moran's I (8-Sides)",
            size=(800, 600), dpi=200, frame_style=:box, grid=false)
    savefig(joinpath(figures_path, "lorenz_trajectory_xz_8sides.png"))






end


main()