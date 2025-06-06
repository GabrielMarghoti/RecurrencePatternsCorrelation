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
    
    #rr_resol = 20
    rrs = [0.01; 0.1; 0.2] #10 .^ range(-4, -0.01, rr_resol)
    rr_resol = length(rrs)
    
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
    global_data_path = "data/Morans_I_global/Nf$(Nf)"
    time_series_path = "data/time_series"
    data_path    = "data/Morans_I_local/Nf$(Nf)"
    figures_path = "figures/Morans_I_local_$(today())/Nf$(Nf)"

    mkpath(data_path)
    mkpath(time_series_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component
    Morans_I_diag  = zeros(Nf, length(systems), rr_resol) 

    Morans_I_anti_diag = zeros(Nf, length(systems), rr_resol) 

    Morans_I_vert_line = zeros(Nf, length(systems), rr_resol) 
    
    Morans_I_4sides = zeros(Nf, length(systems), rr_resol) 

    Morans_I_8sides = zeros(Nf, length(systems), rr_resol) 


    for (i, system_tuple) in enumerate(systems)
        system_name, params, component = system_tuple
        
        println("Analyzing $system_name system...")

        system_path = figures_path*"/$system_name$(params)"
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

        for rr_idx in 1:rr_resol # Create recurrence plot            
            RP_file = joinpath(data_path, "RP_$(system_name)_$(params)_rr$(rrs[rr_idx]).jld2")
            if isfile(RP_file)
                @load RP_file RP _rqa_sys
            else
                RP = RecurrenceMatrix(StateSpaceSet(time_series), GlobalRecurrenceRate(rrs[rr_idx]); metric = Euclidean(), parallel = true)
                _rqa_sys = rqa(RP)
                @save RP_file RP _rqa_sys
            end

            plot_recurrence_matrix(RP, system_name, system_path, rrs[rr_idx]; filename="recurrence_plot.png")

            Morans_I_diag[:, i, rr_idx] = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == Δj ? 1 : 0), Δi_range = -5:5, Δj_range = -5:5)

            Morans_I_anti_diag[:, i, rr_idx] = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == -Δj ? 1 : 0), Δi_range = -5:5, Δj_range = -5:5)

            Morans_I_vert_line[:, i, rr_idx] = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δj == 0 ? 1 : 0), Δi_range = -5:5, Δj_range = -5:5)

            Morans_I_4sides[:, i, rr_idx] = local_morans_I(RP; weight_function = (Δi, Δj) -> (abs(Δi) + abs(Δj) == 1 ? 1 : 0), Δi_range = -5:5, Δj_range = -5:5)

            Morans_I_8sides[:, i, rr_idx] = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi in [-1, 0, 1] && Δj in [-1, 0, 1] ? 1 : 0), Δi_range = -5:5, Δj_range = -5:5)
            

            plot_colored_scatter(time_series, Morans_I_diag[:, i, rr_idx]; color_label="I Diagonals", 
                figures_path=system_path*"/rr$(rrs[rr_idx])", 
                filename="moran_I_diagonal.png")
            plot_colored_scatter(time_series, Morans_I_anti_diag[:, i, rr_idx]; color_label="I Anti Diagonals", 
                figures_path=system_path*"/rr$(rrs[rr_idx])", 
                filename="moran_I_anti_diagonal.png")
            plot_colored_scatter(time_series, Morans_I_vert_line[:, i, rr_idx]; color_label="I Vertical Lines", 
                figures_path=system_path*"/rr$(rrs[rr_idx])", 
                filename="moran_I_vertical_lines.png")

            plot_colored_scatter(time_series, Morans_I_4sides[:, i, rr_idx]; color_label="I 4 sides", 
                figures_path=system_path*"/rr$(rrs[rr_idx])", 
                filename="moran_I_4sides.png")

            plot_colored_scatter(time_series, Morans_I_8sides[:, i, rr_idx]; color_label="I 8 sides", 
                figures_path=system_path*"/rr$(rrs[rr_idx])", 
                filename="moran_I_8sides.png")

            if ndims(time_series) == 1 
                plot_shared_xaxis_scatter(time_series, Morans_I_diag[:, i, rr_idx]; quantifier_label="I Diagonals", 
                figures_path=system_path*"/rr$(rrs[rr_idx])", 
                filename="xaxis_share_moran_I_diagonal.png")
            end

            plot_quantifier_histogram(Morans_I_diag[:, i, rr_idx]; xlabel="Moran's I for diagonals", 
                ylabel="Frequency", 
                figures_path=system_path*"/rr$(rrs[rr_idx])", 
                filename="Diag_moran_I_histogram.png")
            plot_quantifier_histogram(Morans_I_anti_diag[:, i, rr_idx]; xlabel="Moran's I for anti-diagonals", 
                ylabel="Frequency", 
                figures_path=system_path*"/rr$(rrs[rr_idx])", 
                filename="Anti_diag_moran_I_histogram.png")
            plot_quantifier_histogram(Morans_I_vert_line[:, i, rr_idx]; xlabel="Moran's I for vertical lines", 
                ylabel="Frequency", 
                figures_path=system_path*"/rr$(rrs[rr_idx])", 
                filename="Vert_line_moran_I_histogram.png")
            plot_quantifier_histogram(Morans_I_4sides[:, i, rr_idx]; xlabel="Moran's I for 4 sides", 
                ylabel="Frequency", 
                figures_path=system_path*"/rr$(rrs[rr_idx])", 
                filename="4sides_moran_I_histogram.png")
            plot_quantifier_histogram(Morans_I_8sides[:, i, rr_idx]; xlabel="Moran's I for 8 sides", 
                ylabel="Frequency", 
                figures_path=system_path*"/rr$(rrs[rr_idx])", 
                filename="8sides_moran_I_histogram.png")
        end

        # save 
        file_path = data_path * "/morans_I_$(system_name).jld2"
        @save file_path Morans_I_diag Morans_I_anti_diag Morans_I_vert_line Morans_I_4sides Morans_I_8sides
    end

end


main()