# transition_time.jl

using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions
using Plots 
using DelayEmbeddings
using JLD2

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs

function main()
    # Parameters
    Nf = 2000
    
    # 3D autoregressive model connection matrix
    A = [0.35  0.2  0.25;
         0.2  0.45  0.2;
         0.25  0.5  0.2]
    
    # Systems to analyze
    systems = [
        ("Tipping point sde", [[0.1, 0.004], 0.1], 1),
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
    time_series_path = "data/time_series"
    data_path    = "data/RPC_d_ij_distributions_I_global/Nf$(Nf)/"
    figures_path = "figures/RPC_d_ij_distributions_I_global_$(today())/Nf$(Nf)/"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component
    Morans_I_diag  = zeros(length(systems)) 

    Morans_I_anti_diag = zeros(length(systems)) 

    Morans_I_vert_line = zeros(length(systems)) 
    
    Morans_I_4sides = zeros(length(systems)) 

    Morans_I_8sides = zeros(length(systems)) 

    det_results = zeros(length(systems)) 

    lam_results = zeros(length(systems)) 


    labels = ["Diagonal Moran's I", "Anti-Diagonal Moran's I", "Vertical Line Moran's I", "4-Sides Moran's I", "8-Sides Moran's I"]

    for (i, system_tuple) in enumerate(systems)
        system_name, params, component = system_tuple
        
        println("Analyzing $system_name system...")

        system_path = figures_path*"$(system_name)_$(params)"
        mkpath(system_path)
        
        # Try to load the time series from cache
        time_series_file = joinpath(time_series_path, "time_series_$(system_name)_$(params).jld2")
        if isfile(time_series_file)
            println("   Loading cached time series")
            @load time_series_file time_series
        else
            println("   Generating new time series")
            time_series = generate_time_series(system_name, params, Nf, component) # Generate time series
            @save time_series_file time_series
        end

        if size(time_series, 2) == 1
            #println("   Performing Pecuzal embedding for 1D time series")
            embedded_time_series = pecuzal_embedding(time_series; max_cycles=7)[1]
        else
            embedded_time_series = time_series
        end
        

        dij = distancematrix(StateSpaceSet(embedded_time_series))

        #plot_recurrence_matrix(dij, system_name, system_path; filename="distance_matrix.png")

        Morans_I_diag[i] = morans_I(dij; weight_function = (Δi, Δj) -> (Δi == Δj ? 1 : 0))

        Morans_I_anti_diag[i] = morans_I(dij; weight_function = (Δi, Δj) -> (Δi == -Δj ? 1 : 0))

        Morans_I_vert_line[i] = morans_I(dij; weight_function = (Δi, Δj) -> (Δj == 0 ? 1 : 0))

        Morans_I_4sides[i] = morans_I(dij; weight_function = (Δi, Δj) -> (abs(Δi) + abs(Δj) == 1 ? 1 : 0))

        Morans_I_8sides[i] = morans_I(dij; weight_function = (Δi, Δj) -> (Δi in [-1, 0, 1] && Δj in [-1, 0, 1] ? 1 : 0))
        for deltai in -1:1, deltaj in -1:1
            cond_values = _cond_recurrence_values(dij; Δi=deltai, Δj=deltaj)
            plot_quantifier_histogram(cond_values; xlabel="(d(i,j)-<d>)(d(i+Δi, j+ Δj)-<d>)", ylabel="Frequency", color=:blue, figures_path=system_path, filename="Δi$(deltai)_Δj$(deltaj).png")
        end

    end

    # Plot histograms comparing systems 

    histogram_data = [
        Morans_I_diag,
        Morans_I_anti_diag,
        Morans_I_vert_line,
        Morans_I_4sides,
        Morans_I_8sides
    ]

    # Plot histograms
    save_histograms(histogram_data, labels, systems, "Recurrence Pattern Correlation for distance matrix)", figures_path, "bar_plot_rr_dij.png")

end


main()