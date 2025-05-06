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
    
    rr_resol = 20
    rrs = 10 .^ range(-4, -0.01, rr_resol)
    rr_resol = length(rrs)
    
    # 3D autoregressive model connection matrix
    A = [0.35  0.2  0.25;
         0.2  0.45  0.2;
         0.25  0.5  0.2]
    
    # Systems to analyze
    systems = [
        #("Tipping point sde", [[0.1, 0.004], 0.1], 1),
        ("Lorenz traj", [[10.0, 28.0, 8 / 3], 0.05], [1,2,3]),
        ("Rossler traj", [[0.2, 0.2, 5.7], 0.5], [1,2,3]),
        ("Logistic", 4.0, 1),
        ("AR 0.9", 0.9, 1),
        ("Logistic", 3.678, 1),
        ("Randn", nothing, 1),
        ("AR 0.1", 0.1, 1),
        ("AR 0.99", 0.99, 1),
        ("AR(2)", [[0.7, -0.2], 0.5], 1),  # AR(2) with noise variance
        ("Lorenz (x)", [[10.0, 28.0, 8 / 3], 0.05], 1),
        ("Lorenz (z)", [[10.0, 28.0, 8 / 3], 0.05], 3),
       # ("Logistic 3D", nothing, [3.711, 0.06], 1),
       # ("AR 0.3", nothing, 0.3, 1),
       # ("AR 0.8", nothing, 0.8, 1),
        ("3D AR", A, 1),
        ("Rossler (x)", [[0.2, 0.2, 5.7], 0.5], 1),
        ("Circle", 0.11347, [1,2]),
        #("GARCH", nothing, 
        #[0.01,
        #[0.1, 0.05],  # ARCH(2)
        #[0.7, 0.2, 0.05]]  # GARCH(3)
        #, 1)  # ω, α, β
    ]
    
    # Output directories
    data_path    = "data/Morans_I_global_$(today())/Nf$(Nf)"
    figures_path = "figures/Morans_I_global_$(today())/Nf$(Nf)"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component
    Morans_I_diag  = zeros(length(systems), rr_resol) 

    Morans_I_anti_diag = zeros(length(systems), rr_resol) 

    Morans_I_vert_line = zeros(length(systems), rr_resol) 
    
    Morans_I_4sides = zeros(length(systems), rr_resol) 

    Morans_I_8sides = zeros(length(systems), rr_resol) 


    for (i, system_tuple) in enumerate(systems)
        system_name, params, component = system_tuple
        
        println("Analyzing $system_name system...")

        system_path = figures_path*"/$system_name$(params)"
        mkpath(system_path)
        
        time_series = generate_time_series(system_name, params, Nf, component) # Generate time series

        for rr_idx in 1:rr_resol # Create recurrence plot
            RP = Matrix(RecurrenceMatrix(StateSpaceSet(time_series), GlobalRecurrenceRate(rrs[rr_idx]); metric = Euclidean(), parallel = true))
            plot_recurrence_matrix(RP, system_name, system_path, rrs[rr_idx]; filename="recurrence_plot.png")

            Morans_I_diag[i, rr_idx] = morans_I(RP; weight_function = (Δi, Δj) -> (Δi == Δj ? 1 : 0))

            Morans_I_anti_diag[i, rr_idx] = morans_I(RP; weight_function = (Δi, Δj) -> (Δi == -Δj ? 1 : 0))

            Morans_I_vert_line[i, rr_idx] = morans_I(RP; weight_function = (Δi, Δj) -> (Δj == 0 ? 1 : 0))

            Morans_I_4sides[i, rr_idx] = morans_I(RP; weight_function = (Δi, Δj) -> (abs(Δi) + abs(Δj) == 1 ? 1 : 0))

            Morans_I_8sides[i, rr_idx] = morans_I(RP; weight_function = (Δi, Δj) -> (Δi in [-1, 0, 1] && Δj in [-1, 0, 1] ? 1 : 0))

        end

        # save 
        file_path = data_path * "/morans_I_$(system_name).jld2"
        @save file_path Morans_I_diag Morans_I_anti_diag Morans_I_vert_line Morans_I_4sides Morans_I_8sides
    end

    # Plot histograms comparing systems for each recurrence rate
    for rr_idx in 1:rr_resol
        histogram_data = [
            Morans_I_diag[:, rr_idx],
            Morans_I_anti_diag[:, rr_idx],
            Morans_I_vert_line[:, rr_idx],
            Morans_I_4sides[:, rr_idx],
            Morans_I_8sides[:, rr_idx]
        ]
        labels = ["Diagonal Moran's I", "Anti-Diagonal Moran's I", "Vertical Line Moran's I", "4-Sides Moran's I", "8-Sides Moran's I"]

        # Plot histograms
        save_histograms(histogram_data, labels, systems, "Recurrence Structure Correlation for rr=$(rrs[rr_idx])", joinpath(figures_path, "rr$(rrs[rr_idx])"), "bar_plot.png")

    end

    # Plot Moran's I as a function of recurrence rate for each system
    for i in 1:length(systems)
        system_name, params, component = systems[i]

        plt = plot(rrs, [Morans_I_diag[i, :] Morans_I_anti_diag[i, :] Morans_I_vert_line[i, :] Morans_I_4sides[i, :] Morans_I_8sides[i, :]],
                    label = ["Diagonal" "Anti-Diagonal" "Vertical Line" "4-Sides" "8-Sides"],
                    xlabel = "Recurrence Rate (rr)", ylabel = "Moran's I", 
                    title = "$system_name Moran's I vs rr", 
                    legend = false, size = (800, 600), dpi = 200, grid = true, frame_style = :box)

        savefig(plt, joinpath(figures_path*"/$system_name$(params)", "Morans_I_vs_rr.png"))

    end
end


main()