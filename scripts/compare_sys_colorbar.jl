# transition_time.jl

using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions
using Plots, Colors
using DelayEmbeddings
using JLD2
using LaTeXStrings


gr() 
default(fontfamily="Computer Modern", linewidth=2, label=nothing, grid=false, framestyle=:box)

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs

function diag_marker(ratio=1.0)
    x = ratio*[-0.1, 1.0]
    y = ratio*[-0.1, 1.0]
    return Shape(x, y)
end


function anti_diag_marker(ratio=1.0)
    x = ratio*[-0.1, 1.0]
    y = ratio*[0.1, -1.0]
    return Shape(x, y)
end


function vert_marker(ratio=1.0)
    x = ratio*[0.0, 0.0]
    y = ratio*[-0.1, 1.0]
    return Shape(x, y)
end

function main()
    # Parameters
    Nf = 10000
    
    load_cache = true

    rr = 0.01
    
    # Systems to analyze
    systems = [
        ("GWN", nothing, 1),
        ("AR(1)[0.8]", 0.8, 1),
        ("AR(1)[0.99]", 0.99, 1),
        ("Logistic Map", 4.0, 1),
        ("Lorenz'63", [[10.0, 28.0, 8 / 3], 0.05], [1,2,3]),
        ("Sine function", 0.01, [1,2]),
    ]
    
    n_systems = length(systems)

    # Output directories
    time_series_path = "data/time_series"
    data_path    = "data/RPC_compare_sys_global_colorbar/Nf$(Nf)_rr_$(rr)/"
    figures_path = "figures/RPC_compare_sys_global_colorbar_$(today())/Nf$(Nf)_rr_$(rr)/"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component
    det_results = zeros(length(systems)) 

    lam_results = zeros(length(systems)) 

    RPC = zeros(length(systems), 3) # Initialize RPC array

    cache_file = joinpath(data_path, "data.jld2")


    if isfile(cache_file) && load_cache
        @load cache_file RPC systems det_results lam_results
    else
        for (i, system_tuple) in enumerate(systems)
            system_name, params, component = system_tuple
            
            println("Analyzing $system_name system...")

            system_path = figures_path*"$(system_name)_$(params)"
            mkpath(system_path)
            
            time_series = generate_time_series(system_name, params, Nf, component) # Generate time series

            if size(time_series, 2) == 1
                #println("   Performing Pecuzal embedding for 1D time series")
                embedded_time_series = pecuzal_embedding(StateSpaceSet(time_series); max_cycles=10)[1]
            else
                embedded_time_series = StateSpaceSet(time_series)
            end
            
         
            RP = RecurrenceMatrix(embedded_time_series, GlobalRecurrenceRate(rr); metric = Euclidean(), parallel = true)
            _rqa_sys = rqa(RP)

            det_results[i] = _rqa_sys[:DET]
            lam_results[i] = _rqa_sys[:LAM]

             
            RPC[i, 1] = RPMotifs.RPC(RP; weight_function = (Δi, Δj) -> ((abs(Δi) == 1 && Δj == 0) || abs(Δj) == 1 && Δi == 0 ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)      

            RPC[i, 2] = RPMotifs.RPC(RP; weight_function = (Δi, Δj) -> (Δi == Δj && Δi != 0 ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)      

            RPC[i, 3] = RPMotifs.RPC(RP; weight_function = (Δi, Δj) -> (Δi == -Δj && Δi != 0 ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)   

        end

        @save cache_file RPC systems det_results lam_results
    end

    # Plot histograms comparing systems for each recurrence rate
    histogram_data = [
        RPC[:, i] for i in 1:3
    ]

    # Plot histograms
    save_histograms(histogram_data, 
        systems, 
        "",#"Recurrence Patterns Correlation for rr=$(round(rrs[rr_idx]; digits=2))", 
        figures_path, 
        "bar_plot.png"
    )

    # Plot histograms
    save_histograms(histogram_data, 
        systems, 
        "",#"Recurrence Patterns Correlation for rr=$(round(rrs[rr_idx]; digits=2))", 
        figures_path, 
        "bar_plot.svg"
    )
    # Plot histograms
    save_histograms(histogram_data, 
        systems, 
        "",#"Recurrence Patterns Correlation for rr=$(round(rrs[rr_idx]; digits=2))", 
        figures_path, 
        "bar_plot.pdf"
    )

    resolution_inplot = 40
    N_trials = 20
    RPC_trials = zeros(resolution_inplot, length(systems), N_trials) # Initialize RPC array

    # generate sample sizes (actual Nf values)
    Nf_values = Int.(round.(range(10, Nf, length=resolution_inplot), digits=0))

    cache2_file = joinpath(data_path, "data_rpc_vs_Nfresol$(resolution_inplot)_ntrial$(N_trials).jld2")


    if isfile(cache2_file) && load_cache
        @load cache2_file RPC_trials Nf_values systems
    else

        for (nf_idx, Nf) in enumerate(Nf_values)
            
            
            println("Analyzing Nf=$Nf ...")
            for trial_idx in 1:N_trials
                for (i, system_tuple) in enumerate(systems)
                    system_name, params, component = system_tuple
                    
                    println("Analyzing $system_name system...")

                    system_path = figures_path*"$(system_name)_$(params)"
                    mkpath(system_path)
                    
                    time_series = generate_time_series(system_name, params, Nf, component) # Generate time series

                    embedded_time_series = StateSpaceSet(time_series)
                    #if size(time_series, 2) == 1
                        #println("   Performing Pecuzal embedding for 1D time series")
                    #    embedded_time_series = pecuzal_embedding(StateSpaceSet(time_series); max_cycles=10)[1]
                    #else
                    #    embedded_time_series = StateSpaceSet(time_series)
                    #end
                    
                
                    RP = RecurrenceMatrix(embedded_time_series, GlobalRecurrenceRate(rr); metric = Euclidean(), parallel = true)

                    
                    RPC_trials[nf_idx, i, trial_idx] = RPMotifs.RPC(RP; weight_function = (Δi, Δj) -> ((abs(Δi) == 1 && Δj == 0) || abs(Δj) == 1 && Δi == 0 ? 1 : 0), Δi_range = -1:1, Δj_range = -1:1)      
                end
            end
            
            RPC_means = mean(RPC_trials, dims=3)
            RPC_stds  = std(RPC_trials, dims=3)

            rpc_vs_serie_size(Nf_values, RPC_means, RPC_stds,
                systems, 
                "",
                figures_path, 
                "rpc_function_serie_size.png"
            )
            @save cache2_file  RPC_trials Nf_values systems
        end
    end

    RPC_means = mean(RPC_trials, dims=3)
    RPC_stds  = std(RPC_trials, dims=3)

    rpc_vs_serie_size(Nf_values, RPC_means, RPC_stds,
        systems, 
        "",
        figures_path, 
        "rpc_function_serie_size.png"
    )
    #=
    rpc_vs_serie_size(Nfs, RPC_means, RPC_stds,
        systems, 
        "",
        figures_path, 
        "rpc_function_serie_size.svg"
    )


    rpc_vs_serie_size(Nfs, RPC_means, RPC_stds,
        systems, 
        "",
        figures_path, 
        "rpc_function_serie_size.pdf"
    )
=#

end


main()