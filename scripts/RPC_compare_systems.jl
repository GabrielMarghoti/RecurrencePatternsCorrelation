# transition_time.jl

using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions
using Plots, Colors
using DelayEmbeddings
using JLD2
using LaTeXStrings

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs

function main()
    # Parameters
    Nf = 1000
    
    rr_resol = 10
    rrs = range(0.0001, 0.999, rr_resol) #10 .^ range(-3, -0.001, rr_resol)
    rr_resol = length(rrs)
    
    # 3D autoregressive model connection matrix
    A = [0.35  0.2  0.25;
         0.2  0.45  0.2;
         0.25  0.5  0.2]
    
    # Systems to analyze
    systems = [
        #("Tipping point sde", [[0.1, 0.004], 0.1], 1),
        ("UWN", nothing, 1),
       # ("GWN", nothing, 1),
        ("AR (0.6)", 0.6, 1),
        ("AR (0.99)", 0.99, 1),
       # ("NAR (0, 0.99)", [0.0; 0.99], 1),
        ("Logistic", 4.0, 1),
       # ("Logistic", 3.678, 1),
        ("Lorenz", [[10.0, 28.0, 8 / 3], 0.05], [1,2,3]),
        ("Rossler", [[0.2, 0.2, 5.7], 0.5], [1,2,3]),
       # ("AR(2)", [[0.7, -0.2], 0.5], 1),  # AR(2) with noise variance
       # ("Lorenz (x)", [[10.0, 28.0, 8 / 3], 0.05], 1),
       # ("Lorenz (z)", [[10.0, 28.0, 8 / 3], 0.05], 3),
       # ("Logistic 3D", nothing, [3.711, 0.06], 1),
       # ("AR 0.3", nothing, 0.3, 1),
       # ("AR 0.8", nothing, 0.8, 1),
       # ("3D AR", A, 1),
       # ("Rossler (x)", [[0.2, 0.2, 5.7], 0.5], 1),
        ("Sine", 0.11347, [1,2]),
        #("GARCH", nothing, 
        #[0.01,
        #[0.1, 0.05],  # ARCH(2)
        #[0.7, 0.2, 0.05]]  # GARCH(3)
        #, 1)  # ω, α, β
    ]
    
    n_systems = length(systems)

    # Output directories
    time_series_path = "data/time_series"
    data_path    = "data/RPC_compare_sys_global/Nf$(Nf)_rr_reso$(rr_resol)/"
    figures_path = "figures/RPC_compare_sys_global_$(today())/Nf$(Nf)_rr_reso$(rr_resol)/"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component

    det_results = zeros(length(systems), rr_resol) 

    lam_results = zeros(length(systems), rr_resol) 

    di_dj_tuples =[(1, -1), (1, 0), (1, 1)] 

    len_RPC = length(di_dj_tuples)
    didj_labels = [L"\Delta i=%$(shift[1]), \Delta j=%$(shift[2])" for shift in di_dj_tuples]
    
    RPC = zeros(length(systems), rr_resol, len_RPC) # Initialize RPC array

    for (i, system_tuple) in enumerate(systems)
        system_name, params, component = system_tuple
        
        println("Analyzing $system_name system...")

        system_path = figures_path*"$(system_name)_$(params)"
        mkpath(system_path)
        
        time_series = generate_time_series(system_name, params, Nf, component) # Generate time series

        if size(time_series, 2) == 1
            #println("   Performing Pecuzal embedding for 1D time series")
            embedded_time_series = pecuzal_embedding(time_series; max_cycles=7)[1]
        else
            embedded_time_series = time_series
        end
        
        file_path = data_path * "rr_resol$(rr_resol)_morans_I_$(system_name).jld2"
        #if isfile(file_path)
        #    @load  file_path rrs RPC_diag RPC_anti_diag RPC_vert_line RPC_4sides RPC_8sides  det_results lam_results
        for rr_idx in 1:rr_resol

            rr_fig_path = joinpath(system_path, "rr$(rrs[rr_idx])/")
            rr_data_path = joinpath(data_path, "rr$(rrs[rr_idx])/")
            mkpath(rr_data_path)

            RP = RecurrenceMatrix(StateSpaceSet(embedded_time_series), GlobalRecurrenceRate(rrs[rr_idx]); metric = Euclidean(), parallel = true)
            _rqa_sys = rqa(RP)

            det_results[i, rr_idx] = _rqa_sys[:DET]
            lam_results[i, rr_idx] = _rqa_sys[:LAM]
            for (RPC_idx, di_dj) in enumerate(di_dj_tuples)
                di, dj = di_dj
                RPC_name = "RPC_di$(di)_dj$(dj)"

                plot_recurrence_matrix(RP, system_name, rr_fig_path; filename="recurrence_plot.png")
                
                RPC[i, rr_idx, RPC_idx] = morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)      

            end
        end

    end



    n_systems = length(systems)
    colors = cgrad(:jet1, n_systems, categorical=true)

    
    # Plot histograms comparing systems for each recurrence rate
    for rr_idx in 1:rr_resol
        histogram_data = [
            RPC[:, rr_idx, i] for i in 1:len_RPC
        ]
  
        # Plot histograms
        save_histograms(histogram_data, didj_labels, 
            systems, 
            "",#"Recurrence Patterns Correlation for rr=$(round(rrs[rr_idx]; digits=2))", 
            figures_path, 
            "bar_plot_rr$(round(rrs[rr_idx]; digits=2)).png"
        )

    end

    
    n_panels = size(RPC, 3)
    labels = [sys_info[1] for sys_info in systems]

    plt = plot(layout = (n_panels, 1), size = (500, 180 * n_panels), dpi = 300)

    letters_annotation = [L"(%c)" % ('a' + i - 1) for i in 1:n_panels]
   
    for i in 1:n_panels
        for k in 1:n_systems
            plot!(plt[i], rrs, RPC[k, :, i],
                label = (i == 1 ? labels[k] : ""),  # Show legend only on first panel
                color = colors[k])
        end
        xlabel = i == n_panels ? L"Recurrence \ Rate \ (rr)" : ""
        ylabel = L"RPC"
        annotation_text = L"%$(letters_annotation[i]) \ w_{%$(di_dj_tuples[i][1]),%$(di_dj_tuples[i][2])}=1"
        plot!(plt[i],
            xlabel = xlabel,
            ylabel = ylabel,
            annotation = (0.0, 0.95, annotation_text, :left, 10, :black),
            legend = (i == 1 ? :topright : false),
            grid = false,
            frame_style = :box)
    end

    mkpath(figures_path)
    savefig(plt, joinpath(figures_path, "all_systems_RPC_vs_rr_panels.png"))
    # Plot all systems in the same plot, each system in a panel (subplot) in a different row
   
    plt = plot(layout = (n_panels, 1), size = (600, 180 * n_panels), dpi = 300)
    for i in 1:n_panels
        for k in 1:n_systems
            plot!(plt[i], rrs, RPC[k, :, i],
                label = (i == 1 ? labels[k] : ""),  # Show legend only on first panel
                color = colors[k])
        end
        xlabel = i == n_panels ? L"Recurrence \ Rate \ (rr)" : ""
        ylabel = L"RPC"
        annotation_text = L"%$(letters_annotation[i]) \ w_{%$(di_dj_tuples[i][1]),%$(di_dj_tuples[i][2])}=1"
        plot!(plt[i],
            xlabel = xlabel,
            ylabel = ylabel,
            annotation = (0.0, 0.95, annotation_text, :left, 10, :black),
            legend = (i == 1 ? :topright : false),
            grid = false,
            xscale=:log10,
            frame_style = :box)
    end

    savefig(plt, joinpath(figures_path, "all_systems_RPC_vs_rr_panels_logx.png"))

    #= Scatter plot for :DET
    plt_det = scatter(det_results', RPC_diag',
                    label = reshape([dat[1] for dat in systems], 1, length(systems)),
                    xlabel = "DET", ylabel = "C_diag", 
                    title = "Moran's I Diag vs DET", 
                    legend = :outerright, size = (600, 400), dpi = 200, grid = true, frame_style = :box)
    savefig(plt_det, joinpath(figures_path, "all_systems_RPC_vs_DET.png"))

    # Scatter plot for :LAM
    plt_lam = scatter(lam_results', RPC_vert_line',
                    label = reshape([dat[1] for dat in systems], 1, length(systems)),
                    xlabel = "LAM", ylabel = "C_vert_line", 
                    title = "Moran's I vert. line vs LAM", 
                    legend = :outerright, size = (600, 400), dpi = 200, grid = true, frame_style = :box)
    savefig(plt_lam, joinpath(figures_path, "all_systems_RPC_vs_LAM.png"))
    =#
end


main()