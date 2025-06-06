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
    
    rr_resol = 10
    rrs = 10 .^ range(-3, -0.001, rr_resol)
    rr_resol = length(rrs)
    
    # 3D autoregressive model connection matrix
    A = [0.35  0.2  0.25;
         0.2  0.45  0.2;
         0.25  0.5  0.2]
    
    # Systems to analyze
    systems = [
        ("Lorenz traj", [[10.0, 28.0, 8 / 3], 0.05], [1,2,3]),
        ("Rossler traj", [[0.2, 0.2, 5.7], 0.5], [1,2,3]),
        ("Circle", 0.11347, [1,2]),
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
        ("Tipping point sde", [[0.1, 0.004], 0.1], 1),
        #("GARCH", nothing, 
        #[0.01,
        #[0.1, 0.05],  # ARCH(2)
        #[0.7, 0.2, 0.05]]  # GARCH(3)
        #, 1)  # ω, α, β
    ]
    
    # Output directories
    time_series_path = "data/time_series"
    data_path    = "data/RPC_d_ij_epsilondiff_distributions/Nf$(Nf)/"
    figures_path = "figures/RPC_d_ij_epsilondiff_distributions$(today())/Nf$(Nf)/"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component
    Morans_I_diag  = zeros(length(systems), rr_resol) 

    Morans_I_anti_diag = zeros(length(systems), rr_resol) 

    Morans_I_vert_line = zeros(length(systems), rr_resol) 
    
    Morans_I_4sides = zeros(length(systems), rr_resol) 

    Morans_I_8sides = zeros(length(systems), rr_resol) 

    det_results = zeros(length(systems), rr_resol) 

    lam_results = zeros(length(systems), rr_resol) 


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
        
        file_path = data_path * "rr_resol$(rr_resol)_morans_I_$(system_name).jld2"
        #if isfile(file_path)
        #    @load  file_path rrs Morans_I_diag Morans_I_anti_diag Morans_I_vert_line Morans_I_4sides Morans_I_8sides  det_results lam_results
        Threads.@threads for rr_idx in 1:rr_resol

            rr_fig_path = joinpath(system_path, "rr$(rrs[rr_idx])/")
            rr_data_path = joinpath(data_path, "rr$(rrs[rr_idx])/")
            mkpath(rr_data_path)
            mkpath(rr_fig_path)
            
            dij = distancematrix(StateSpaceSet(embedded_time_series))
            epsilon = rrs[rr_idx]*maximum(dij)
            

            RP = RecurrenceMatrix(StateSpaceSet(embedded_time_series), epsilon; metric = Euclidean())
            _rqa_sys = rqa(RP)
            
            det_results[i, rr_idx] = _rqa_sys[:DET]
            lam_results[i, rr_idx] = _rqa_sys[:LAM]

            plot_recurrence_matrix(dij.-epsilon, system_name, rr_fig_path; filename="dij-eps.png")

            Morans_I_diag[i, rr_idx] = RPC((dij), (epsilon); weight_function = (Δi, Δj) -> (Δi == Δj ? 1 : 0))

            Morans_I_anti_diag[i, rr_idx] = RPC((dij), (epsilon); weight_function = (Δi, Δj) -> (Δi == -Δj ? 1 : 0))

            Morans_I_vert_line[i, rr_idx] = RPC((dij), (epsilon); weight_function = (Δi, Δj) -> (Δj == 0 ? 1 : 0))

            Morans_I_4sides[i, rr_idx] = RPC((dij), (epsilon); weight_function = (Δi, Δj) -> (abs(Δi) + abs(Δj) == 1 ? 1 : 0))

            Morans_I_8sides[i, rr_idx] = RPC((dij), (epsilon); weight_function = (Δi, Δj) -> (Δi in [-1, 0, 1] && Δj in [-1, 0, 1] ? 1 : 0))
            for deltai in -1:1, deltaj in -1:1
                cond_values = _cond_recurrence_values((dij); Δi=deltai, Δj=deltaj)
                plot_quantifier_histogram(cond_values; xlabel="(R(i,j)-rr)(R(i+Δi, j+ Δj)-rr)", ylabel="Frequency", color=:blue, figures_path=rr_fig_path, filename="Δi$(deltai)_Δj$(deltaj).png")
            end
        end
                    
        # Plot in linear scale
        plt_linear = plot(rrs, [Morans_I_diag[i, :] Morans_I_anti_diag[i, :] Morans_I_vert_line[i, :] Morans_I_4sides[i, :] Morans_I_8sides[i, :]],
                label = ["Diagonal" "Anti-Diagonal" "Vertical Line" "4-Sides" "8-Sides"],
                xlabel = "Recurrence Rate (rr)", ylabel = "Moran's I", 
                title = "$system_name Moran's I vs rr (Linear Scale)", 
                legend = true, size = (600, 400), dpi = 200, grid = false, frame_style = :box)
        savefig(plt_linear, joinpath(figures_path*"$(system_name)_$(params)", "Morans_I_vs_rr_linear.png"))
        # 
        plt_diag_vs_DET = plot(rrs, [Morans_I_diag[i, :] det_results[i, :]],
                label = ["RPC Diagonal" "DET"],
                xlabel = "Recurrence Rate (rr)", ylabel = "Quantifier", 
                title = "$system_name", 
                legend = true, size = (600, 400), dpi = 200, grid = false, frame_style = :box)
        savefig(plt_diag_vs_DET, joinpath(figures_path*"$(system_name)_$(params)", "RPC_diag_DET.png"))

        # Plot in log-log scale
        plt_xlog = plot(rrs, [Morans_I_diag[i, :] Morans_I_anti_diag[i, :] Morans_I_vert_line[i, :] Morans_I_4sides[i, :] Morans_I_8sides[i, :]],
                label = ["Diagonal" "Anti-Diagonal" "Vertical Line" "4-Sides" "8-Sides"],
                xlabel = "Recurrence Rate (rr)", ylabel = "Moran's I", 
                title = "$system_name Moran's I vs rr (Log-Log Scale)", 
                legend = true, xscale = :log10, size = (600, 400), dpi = 200, grid = false, frame_style = :box)
        savefig(plt_xlog, joinpath(figures_path*"$(system_name)_$(params)", "Morans_I_vs_rr_xlog.png"))

        # Scatter plot for :DET
        plt_det = scatter(det_results[i, :], [Morans_I_diag[i, :] Morans_I_anti_diag[i, :] Morans_I_vert_line[i, :] Morans_I_4sides[i, :] Morans_I_8sides[i, :]],
            label = ["Diagonal" "Anti-Diagonal" "Vertical Line" "4-Sides" "8-Sides"],
            xlabel = ":DET", ylabel = "Moran's I", 
            title = "$system_name Moran's I vs :DET", 
            legend = :outerright, size = (600, 400), dpi = 200, grid = true, frame_style = :box)
        savefig(plt_det, joinpath(figures_path*"$(system_name)_$(params)", "Morans_I_vs_DET.png"))

        # Scatter plot for :LAM
        plt_lam = scatter(lam_results[i, :], [Morans_I_diag[i, :] Morans_I_anti_diag[i, :] Morans_I_vert_line[i, :] Morans_I_4sides[i, :] Morans_I_8sides[i, :]],
            label = ["Diagonal" "Anti-Diagonal" "Vertical Line" "4-Sides" "8-Sides"],
            xlabel = ":LAM", ylabel = "Moran's I", 
            title = "$system_name Moran's I vs :LAM", 
            legend = :outerright, size = (600, 400), dpi = 200, grid = true, frame_style = :box)
        savefig(plt_lam, joinpath(figures_path*"$(system_name)_$(params)", "Morans_I_vs_LAM.png"))
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
  
        # Plot histograms
        save_histograms(histogram_data, labels, systems, "Recurrence Structure Correlation for rr=$(rrs[rr_idx])", figures_path, "bar_plot_rr$(rrs[rr_idx]).png")
    end


    # Scatter plot for :DET
    plt_det = scatter(det_results', Morans_I_diag',
                    label = reshape([dat[1] for dat in systems], 1, length(systems)),
                    xlabel = "DET", ylabel = "C_diag", 
                    title = "Moran's I Diag vs DET", 
                    legend = :outerright, size = (600, 400), dpi = 200, grid = true, frame_style = :box)
    savefig(plt_det, joinpath(figures_path, "all_systems_Morans_I_vs_DET.png"))

    # Scatter plot for :LAM
    plt_lam = scatter(lam_results', Morans_I_vert_line',
                    label = reshape([dat[1] for dat in systems], 1, length(systems)),
                    xlabel = "LAM", ylabel = "C_vert_line", 
                    title = "Moran's I vert. line vs LAM", 
                    legend = :outerright, size = (600, 400), dpi = 200, grid = true, frame_style = :box)
    savefig(plt_lam, joinpath(figures_path, "all_systems_Morans_I_vs_LAM.png"))
end


main()