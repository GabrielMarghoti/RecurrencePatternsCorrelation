# transition_time.jl

using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions

using JLD2

include("../DynamicalSystemsToolkit.jl")
using ..DynamicalSystemsToolkit

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../Quantifiers.jl")
using ..Quantifiers

function main()
    # Parameters
    Nf = 1701
    LMAX = (10,10)
    #resolution = 32
    rrs = [0.01; 0.1; 0.2] # 10 .^ range(-4, -0.01, resolution)
    resolution = length(rrs)
    
    # 3D autoregressive model connection matrix
    A = [0.35  0.2  0.25;
         0.2  0.45  0.2;
         0.25  0.5  0.2]
    
    # Systems to analyze
    systems = [
        ("Tipping point sde", tipping_sde, [[0.1, 0.004], 0.1], 1),
        ("Lorenz traj", lorenz!, [[10.0, 28.0, 8 / 3], 0.05], [1,2,3]),
        ("Rossler traj", rossler!, [[0.2, 0.2, 5.7], 0.5], [1,2,3]),
        ("Logistic", nothing, 4.0, 1),
        ("AR 0.9", nothing, 0.9, 1),
        ("Logistic", nothing, 3.678, 1),
        ("Randn", nothing, nothing, 1),
        ("AR 0.1", nothing, 0.1, 1),
        ("AR 0.99", nothing, 0.99, 1),
        ("AR(2)", nothing, [[0.7, -0.2], 0.5], 1),  # AR(2) with noise variance
        ("Lorenz (x)", lorenz!, [[10.0, 28.0, 8 / 3], 0.05], 1),
        ("Lorenz (z)", lorenz!, [[10.0, 28.0, 8 / 3], 0.05], 3),
       # ("Logistic 3D", nothing, [3.711, 0.06], 1),
       # ("AR 0.3", nothing, 0.3, 1),
       # ("AR 0.8", nothing, 0.8, 1),
        ("3D AR", nothing, A, 1),
        ("Rossler (x)", rossler!, [[0.2, 0.2, 5.7], 0.5], 1),
        ("Circle", nothing, 0.11347, [1,2]),
        #("GARCH", nothing, 
        #[0.01,
        #[0.1, 0.05],  # ARCH(2)
        #[0.7, 0.2, 0.05]]  # GARCH(3)
        #, 1)  # ω, α, β
    ]
    
    # Output directories
    data_path    = "data/Morans_I_$(today())/Nf$(Nf)_LMAX$(LMAX)"
    figures_path = "figures/Morans_I_$(today())/Nf$(Nf)_LMAX$(LMAX)"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component
    probabilities = zeros(length(systems), resolution, 2*LMAX[1]+1, 2*LMAX[2]+1, Nf, 2*2)  # Extra dimension for components
    
    for (i, system_tuple) in enumerate(systems)
        system_name, system, params, component = system_tuple
        
        println("Analyzing $system_name system...")

        system_path = figures_path*"/$system_name$(params)"
        mkpath(system_path)
        
        # Generate trajectory
        if system_name == "Randn"
            time_series = randn(Nf)
            time_series2 = randn(Nf)
        
        elseif startswith(system_name, "AR 0.")
            time_series = generate_ar_trajectory(params, Nf)
            time_series2 = generate_ar_trajectory(params, Nf)
        
        elseif occursin("3D AR", system_name)
            trajectory = generate_3d_ar_trajectory(params, Nf) # Generate system trajectory
            time_series = trajectory[:, component]  # Extract component
            trajectory2 = generate_3d_ar_trajectory(params, Nf) # Generate system trajectory
            time_series2 = trajectory2[:, component]  # Extract component
        
        elseif occursin("Logistic 3D", system_name)
            trajectory = generate_3d_hyperchaotic_logistic_map(params, Nf) # Generate system trajectory
            time_series = trajectory[:, component]  # Extract component
            trajectory2 = generate_3d_hyperchaotic_logistic_map(params, Nf) # Generate system trajectory
            time_series2 = trajectory2[:, component]  # Extract component
        
        elseif occursin("Logistic", system_name)
            trajectory = generate_logistic_map(params, Nf) # Generate system trajectory
            time_series = trajectory
            trajectory2 = generate_logistic_map(params, Nf) # Generate system trajectory
            time_series2 = trajectory2
        
        elseif occursin("Circle", system_name)
            trajectory  = [sin.(2π/3.3 * params * range(1, Nf)) cos.(2π/3.3 * params * range(1, Nf))]
            time_series = trajectory[:, component]
            trajectory2  = [sin.(2π/3.3 * params * range(1, Nf)) cos.(2π/3.3 * params * range(1, Nf))]
            time_series2 = trajectory2[:, component]
        
        elseif system_name == "GARCH"
            time_series = generate_garch(Nf, params...)
            time_series2 = generate_garch(Nf, params...)

        elseif system_name == "AR(2)"
            time_series = generate_ar(Nf, params[1], params[2])
            time_series2 = generate_ar(Nf, params[1], params[2])

        elseif occursin("sde", system_name)

            time_series = analyze_sde_system(system, params, Nf)  # Generate system trajectory
            time_series2 = analyze_sde_system(system, [[0.1, 0.0], 0.1], Nf)  # Generate system trajectory

        else
            trajectory = analyze_system(system, params, Nf)  # Generate system trajectory
            time_series = trajectory[:, component]  # Extract component
            trajectory2 = analyze_system(system, params, Nf)  # Generate system trajectory
            time_series2 = trajectory2[:, component]  # Extract component
        end

        # Plot and save the time series
  
        is = -LMAX[1]:LMAX[1]
        js = -LMAX[2]:LMAX[2]
        # Compute probabilities for each recurrence rate
        for idx in 1:length(rrs) # Create recurrence plot
            RP = RecurrenceMatrix(StateSpaceSet(time_series), GlobalRecurrenceRate(rrs[idx]); metric = Euclidean(), parallel = true)
            plot_recurrence_matrix(RP, system_name, system_path, rrs[idx]; filename="recurrence_plot.png")
   
            Threads.@threads for i_idx in eachindex(is)
                iprime = is[i_idx]
                for (j_idx, jprime) in enumerate(js)
                    L = (iprime, jprime)
                    probabilities[i, idx, i_idx, j_idx, :, :] = motifs_probabilities(RP, L; shape=:timepair, sampling=:columnwise, sampling_region=:lower)
                end
            end

            for t in 1:200:Nf
                plot_motifs_transition_joint_prob(probabilities[i, idx, :, :, t, :], rrs[idx], LMAX, log_scale=false, figures_path=system_path*"/rr$(rrs[idx])/t$(t)")
                plot_motifs_transition_joint_prob(probabilities[i, idx, :, :, t, :], rrs[idx], LMAX, log_scale=true, figures_path=system_path*"/rr$(rrs[idx])/t$(t)")
            end
            
            I_diag = zeros(length(time_series[:,1]))
            I_diag_pos = zeros(length(time_series[:,1]))
            I_diag_neg = zeros(length(time_series[:,1]))

            for t in 1:Nf
                I_diag[t] = morans_I(probabilities[i, idx, :, :, t, end])
                I_diag_pos[t] = morans_I(probabilities[i, idx, :, :, t, end]; weight_function = (Δi, Δj) -> (Δi in [0,1] && Δj in [0,1] ? 1 : 0))
                I_diag_neg[t] = morans_I(probabilities[i, idx, :, :, t, end]; weight_function = (Δi, Δj) -> (Δi in [0,-1] && Δj in [0,-1] ? 1 : 0))
            end

            plot_colored_scatter(time_series, I_diag; color_label="I Diagonals", 
                                 figures_path=system_path*"/rr$(rrs[idx])", 
                                 filename="moran_I_diagonal.png")
            plot_colored_scatter(time_series, I_diag_pos; color_label="I Positive Diagonals", 
                                 figures_path=system_path*"/rr$(rrs[idx])", 
                                 filename="moran_I_diagonal_pos.png")
            plot_colored_scatter(time_series, I_diag_neg; color_label="I Negative Diagonals", 
                                 figures_path=system_path*"/rr$(rrs[idx])", 
                                 filename="moran_I_diagonal_neg.png")
           
            plot_quantifier_histogram(I_diag; xlabel="Moran's I for diagonals", 
                                       ylabel="Frequency", 
                                       figures_path=system_path*"/rr$(rrs[idx])", 
                                       filename="Diag_moran_I_histogram.png")
            plot_quantifier_histogram(I_diag_pos; xlabel="Moran's I for positive diagonals", 
                                       ylabel="Frequency", 
                                       figures_path=system_path*"/rr$(rrs[idx])", 
                                       filename="Diag_pos_moran_I_histogram.png")
            plot_quantifier_histogram(I_diag_neg; xlabel="Moran's I for negative diagonals", 
                                       ylabel="Frequency", 
                                       figures_path=system_path*"/rr$(rrs[idx])", 
                                       filename="Diag_neg_moran_I_histogram.png")
        end

        # save 
        file_path = data_path * "/probabilities_$(system_name).jld2"
        @save file_path probabilities
    end
end

main()