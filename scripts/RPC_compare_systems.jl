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

function diag_marker(ratio=1.0)
    x = ratio*[-0.5, 0.5]
    y = ratio*[-0.5, 0.5]
    return Shape(x, y)
end


function anti_diag_marker(ratio=1.0)
    x = ratio*[-0.5, 0.5]
    y = ratio*[0.5, -0.5]
    return Shape(x, y)
end


function vert_marker(ratio=1.0)
    x = ratio*[0.0, 0.0]
    y = ratio*[-0.1, 1.0]
    return Shape(x, y)
end

function main()
    # Parameters
    Nf = 5000
    
    rr_resol = 100
    rrs = range(0.001, 0.999, rr_resol) #10 .^ range(-3, -0.001, rr_resol)
    rr_resol = length(rrs)
    
    # 3D autoregressive model connection matrix
    A = [0.35  0.2  0.25;
         0.2  0.45  0.2;
         0.25  0.5  0.2]
    
    # Systems to analyze
    systems = [
        #("Tipping point sde", [[0.1, 0.004], 0.1], 1),
        #("UWN", nothing, 1),
        ("GWN", nothing, 1),
        ("AR (0.8)", 0.8, 1),
        ("AR (0.99)", 0.99, 1),
       # ("NAR (0, 0.99)", [0.0; 0.99], 1),
        ("Logistic", 4.0, 1),
       # ("Logistic", 3.678, 1),
        ("Lorenz", [[10.0, 28.0, 8 / 3], 0.05], [1,2,3]),
       # ("Rossler", [[0.2, 0.2, 5.7], 0.5], [1,2,3]),
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

    di_dj_tuples =[(0, 1), (1, -1), (1, 1), (0, 2), (2, -2), (2, 2), (0, 4), (4, -4), (4, 4)] 

    len_RPC = length(di_dj_tuples)
    didj_labels = [L" w_{\Delta i, \Delta j}=\delta _{\Delta i,%$(shift[1])} \delta_{\Delta j, %$(shift[2])}" for shift in di_dj_tuples]
    
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

    @save joinpath(data_path, "rr_resol$(rr_resol)_morans_I.jld2") rrs RPC systems di_dj_tuples didj_labels



    n_systems = length(systems)

    colors = [:blue, :blue, :blue,  :green, :green, :green, :red, :red, :red] #cgrad(:jet1, len_RPC, categorical=true)
    lws = [2.5, 2.5, 2.5,1.7,1.7,1.7,1,1,1] # Line widths for each RPC type
    markers = [vert_marker(1.0), anti_diag_marker(1.0), diag_marker(1.0), vert_marker(1.5), anti_diag_marker(1.5), diag_marker(1.5), vert_marker(2.0), anti_diag_marker(2.0), diag_marker(2.0)]

    
    # Plot histograms comparing systems for each recurrence rate
    for rr_idx in 1:rr_resol
        histogram_data = [
            RPC[:, rr_idx, i] for i in 1:3
        ]
  
        # Plot histograms
        save_histograms(histogram_data, didj_labels[1:3], 
            systems, 
            "",#"Recurrence Patterns Correlation for rr=$(round(rrs[rr_idx]; digits=2))", 
            figures_path, 
            "bar_plot_rr$(round(rrs[rr_idx]; digits=2)).png"
        )
        # Plot histograms
        save_histograms(histogram_data, didj_labels[1:3], 
            systems, 
            "",#"Recurrence Patterns Correlation for rr=$(round(rrs[rr_idx]; digits=2))", 
            figures_path, 
            "bar_plot_rr$(round(rrs[rr_idx]; digits=2)).pdf"
        )
        # Plot histograms
        save_histograms(histogram_data, didj_labels[1:3], 
            systems, 
            "",#"Recurrence Patterns Correlation for rr=$(round(rrs[rr_idx]; digits=2))", 
            figures_path, 
            "bar_plot_rr$(round(rrs[rr_idx]; digits=2)).svg"
        )
    end

    
    n_panels = n_systems
    
    #labels = [sys_info[1] for sys_info in systems]

    plt = plot(layout =  grid(n_panels, 1, heights=[0.25 ,0.15, 0.15, 0.15, 0.15, 0.15]), size = (550, 150 * n_panels), dpi = 300)

    letters_annotation = ["(a)"; "(b)"; "(c)"; "(d)"; "(e)"; "(f)"; "(g)"; "(h)"]
   
    for i in 1:n_panels
        for k in 1:len_RPC
            plot!(plt[i], rrs, RPC[i, :, k],
                label = (i == 1 ? didj_labels[k] : ""),  # Show legend only on first panel
                color = colors[k], marker= markers[k], markercolor = colors[k],lw=lws[k])
        end
        xlabel = i == n_panels ? L"Recurrence \ Rate \ (rr)" : ""
        ylabel = L"RPC"
        annotation_text = letters_annotation[i]# \ w_{\Delta i, \Delta j}=\delta _{\Delta i,%$(di_dj_tuples[i][1])} \delta_{\Delta j, %$(di_dj_tuples[i][2])}"
        plot!(plt[i],
            xlabel = xlabel,
            ylabel = ylabel,
            annotation = (-0.135, 1.15, annotation_text),
            legend = (i == 1 ? :outertop : false),
            legendcolumns=3,
            top_margin = (i == 1 ? 0*Plots.mm : -4*Plots.mm),
            left_margin = 6*Plots.mm,
            grid = false,
            ylims= (-0.4, 1.1),
            frame_style = :box)
    end

    mkpath(figures_path)
    savefig(plt, joinpath(figures_path, "all_systems_RPC_vs_rr_panels.png"))
    savefig(plt, joinpath(figures_path, "all_systems_RPC_vs_rr_panels.pdf"))
    savefig(plt, joinpath(figures_path, "all_systems_RPC_vs_rr_panels.svg"))


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