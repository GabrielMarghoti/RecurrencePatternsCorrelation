# transition_time.jl

using DynamicalSystems
using DifferentialEquations
using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions


include("DynamicalSystemsToolkit.jl")
using .DynamicalSystemsToolkit

include("PlotUtils.jl")
using .PlotUtils

# Function to compute mutual information
function mutual_information(series, delay)
    n = length(series)
    shifted_series = circshift(series, -delay)
    joint_prob = fit(Histogram, (series, shifted_series), closed=:left)
    marginal_prob_x = fit(Histogram, series, closed=:left)
    marginal_prob_y = fit(Histogram, shifted_series, closed=:left)
    
    joint_prob = joint_prob.weights / n
    marginal_prob_x = marginal_prob_x.weights / n
    marginal_prob_y = marginal_prob_y.weights / n
    
    mi = 0.0
    for i in 1:length(marginal_prob_x)
        for j in 1:length(marginal_prob_y)
            if joint_prob[i, j] > 0
                mi += joint_prob[i, j] * log(joint_prob[i, j] / (marginal_prob_x[i] * marginal_prob_y[j]))
            end
        end
    end
    return mi
end


       
function main()
    # Parameters
    Nf = 2001
    LMAX = (20,20)
    #resolution = 32
    rrs = [0.01; 0.1; 0.2] # 10 .^ range(-4, -0.01, resolution)
    resolution = length(rrs)
    
    # 3D autoregressive model connection matrix
    A = [0.35  0.2  0.25;
         0.2  0.45  0.2;
         0.25  0.5  0.2]
    
    # Systems to analyze
    systems = [
        ("Lorenz traj", lorenz!, [[10.0, 28.0, 8 / 3], 0.1], [1,2,3]),
        ("Logistic", nothing, 4.0, 1),
        ("AR 0.9", nothing, 0.9, 1),
        ("Logistic", nothing, 3.678, 1),
        ("Randn", nothing, nothing, 1),
        ("AR 0.1", nothing, 0.1, 1),
        ("AR 0.99", nothing, 0.99, 1),
        ("AR(2)", nothing, [[0.7, -0.2], 0.5], 1),  # AR(2) with noise variance
        ("Lorenz (x)", lorenz!, [[10.0, 28.0, 8 / 3], 0.1], 1),
        ("Lorenz (z)", lorenz!, [[10.0, 28.0, 8 / 3], 0.1], 3),
       # ("Logistic 3D", nothing, [3.711, 0.06], 1),
       # ("AR 0.3", nothing, 0.3, 1),
       # ("AR 0.8", nothing, 0.8, 1),
        ("3D AR", nothing, A, 1),
        ("Rossler traj", rossler!, [[0.2, 0.2, 5.7], 1], [1,2,3]),
        ("Rossler (x)", rossler!, [[0.2, 0.2, 5.7], 1], 1),
        ("Circle (sine)", nothing, 0.11347, 1),
        #("GARCH", nothing, 
        #[0.01,
        #[0.1, 0.05],  # ARCH(2)
        #[0.7, 0.2, 0.05]]  # GARCH(3)
        #, 1)  # ω, α, β
    ]
    
    # Output directories
    data_path    = "data/fixed_state_cond_recur_$(today())/Nf$(Nf)_LMAX$(LMAX)"
    figures_path = "figures/fixed_state_cond_recur_$(today())/Nf$(Nf)_LMAX$(LMAX)"

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
            trajectory  = [sin.(2π * params * range(1, Nf)) cos.(2π * params * range(1, Nf))]
            time_series = trajectory[:, component]
            trajectory2  = [sin.(2π * params * range(1, Nf)) cos.(2π * params * range(1, Nf))]
            time_series2 = trajectory2[:, component]
        
        elseif system_name == "GARCH"
            time_series = generate_garch(Nf, params...)
            time_series2 = generate_garch(Nf, params...)

        elseif system_name == "AR(2)"
            time_series = generate_ar(Nf, params[1], params[2])
            time_series2 = generate_ar(Nf, params[1], params[2])

        else
            trajectory = analyze_system(system, params, Nf)  # Generate system trajectory
            time_series = trajectory[:, component]  # Extract component
            trajectory2 = analyze_system(system, params, Nf)  # Generate system trajectory
            time_series2 = trajectory2[:, component]  # Extract component
        end

        # Plot and save the time series
        plot(time_series, title = "$system_name Time Series", xlabel = "Time", ylabel = "Value", size=(2000,800))
        savefig(system_path*"/time_series.png")

        plot_mutual_information(time_series[:, 1], system_name; max_delay=LMAX[1])
        savefig(system_path*"/mutual_information.png")

        is = -LMAX[1]:LMAX[1]
        js = -LMAX[2]:LMAX[2]
        # Compute probabilities for each recurrence rate
        for idx in 1:length(rrs) # Create recurrence plot
            RP = CrossRecurrenceMatrix(StateSpaceSet(time_series), StateSpaceSet(time_series2), GlobalRecurrenceRate(rrs[idx]); metric = Euclidean(), parallel = true)

            for (i_idx,iprime) in enumerate(ProgressBar(is))
                for (j_idx,jprime) in enumerate(js)
                    L = (iprime, jprime)
                    probabilities[i, idx, i_idx, j_idx, :, :] = motifs_probabilities(RP, L; shape=:timepair, sampling=:columnwise)
                end
            end

            for t in 1:50:Nf
                plot_motifs_transition_joint_prob(probabilities[i, idx, :, :, t, :], rrs[idx], LMAX, log_scale=false, figures_path=system_path*"/rr$(rrs[idx])/t$(t)")
                plot_motifs_transition_joint_prob(probabilities[i, idx, :, :, t, :], rrs[idx], LMAX, log_scale=true, figures_path=system_path*"/rr$(rrs[idx])/t$(t)")
            end
            #plot_motifs_transition_joint_prob_GIF(probabilities[i, idx, :, :, :, :], rrs[idx], LMAX, log_scale=true, figures_path=system_path*"/rr$(rrs[idx])")
            
            # Plot and save the recurrence plot
            heatmap(RP, title = "$system_name Recurrence Plot", xlabel="Time", ylabel="Time",
            c=:grays, colorbar_title="Recurrence", size=(800, 600), dpi=200,
            colorbar=false, frame_style=:box, aspect_ratio=1, widen=false)
            savefig(system_path*"/rr$(rrs[idx])/recurrence_plot.png")
            #plot_motifs_transition_cond_prob(probabilities[i, idx, :, :, :], LMAX, log_scale=true, figures_path=figures_path*"/$system_name/rr$(rrs[idx])")
        end
        for t in 1:100:Nf
            plot_cond_prob_level_curves(probabilities[i, :, :, :, t, :], rrs, is, js, log_scale=false, figures_path=system_path*"/level_curves/t$(t)")
        end
    end
end

main()