# transition_time.jl

using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions
using Plots 
using DelimitedFiles

using JLD2

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs


function main()
    # Parameters
    
    rr = 0.25

    Nf = 10000
    


    data_file = "orbit_manifold_1e-14_pi"# "orbit_random_0.2_pi" #


    # Output directories
    
    data_path    = "/home/gabrielm/projects/RPMotifs/data/lorenz_RPC_basis/"
    figures_path = "figures/lorenz_RPC_basis_$(today())/Nf$(Nf)_rr$(rr)/"

    mkpath(data_path)
    mkpath(figures_path)
    
    time_series= generate_time_series("Lorenz",  [[10.0, 28.0, 8 / 3], 0.02], Nf, [1,2,3]) # Generate time series

    RP = Matrix(RecurrenceMatrix(StateSpaceSet(time_series), GlobalRecurrenceRate(rr); metric = Euclidean(), parallel = true))

    plot_recurrence_matrix(RP[1:800,1:800], "Lorenz System", figures_path; filename="recurrence_plot.png")


    for di=-1:4
        for dj=-1:4
            if di == 0 && dj == 0
                continue # Skip the case where both di and dj are zero
            end
            RPC_name = "RPC_di$(di)_dj$(dj)"



            RPC = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)
                
            circ = Shape(Plots.partialcircle(0, 2π))   


            #x-y-z
            plot(time_series[:, 1], time_series[:, 2], time_series[:, 3], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
                scatter!(time_series[:, 1], time_series[:, 2], time_series[:, 3], 
                marker_z=RPC, color=:viridis, label="",ms=3, alpha=0.8,
                strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
                xlabel="x", ylabel="y", zlabel="z", title="RPC w($(di), $(dj)) = 1",
                size=(800, 600), dpi=200, frame_style=:box, grid=false)
            savefig(joinpath(figures_path, "lorenz_trajectory_xyz"*RPC_name*".png"))


            #x-y
            plot(time_series[:, 1], time_series[:, 2], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
                scatter!(time_series[:, 1], time_series[:, 2], 
                marker_z=RPC, color=:viridis, label="",ms=3, alpha=0.8,
                strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
                xlabel="x", ylabel="y", title="RPC w($(di), $(dj)) = 1",
                size=(800, 600), dpi=200, frame_style=:box, grid=false)
            savefig(joinpath(figures_path, "lorenz_trajectory_xy"*RPC_name*".png"))


            #x-z
            plot(time_series[:, 1], time_series[:, 3], color=:gray, lw=0.5, label="", alpha=0.5) # Smooth gray lines
                scatter!(time_series[:, 1], time_series[:, 3], 
                marker_z=RPC, color=:viridis, label="",ms=3, alpha=0.8,
                strokewidth=0,  markerstrokealpha = 0, markerstrokecolor = nothing, markershape = circ,
                xlabel="x", ylabel="z", title="RPC w($(di), $(dj)) = 1",
                size=(800, 600), dpi=200, frame_style=:box, grid=false)
            savefig(joinpath(figures_path, "lorenz_trajectory_xz"*RPC_name*".png"))

        end
    end
end


main()