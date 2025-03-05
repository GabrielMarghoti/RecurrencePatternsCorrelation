# transition_time.jl

using DynamicalSystems
using DifferentialEquations
using RecurrenceAnalysis
using Plots
using Measures
using ProgressBars
using Dates
using Random, Distributions, ARCHModels

function generate_garch(N, ω, α, β)
    p = length(β)  # GARCH memory
    q = length(α)  # ARCH memory
    max_lag = max(p, q)
    
    ε = zeros(N)
    σ² = zeros(N)
    
    # Initialize with small noise
    for t in 1:max_lag
        ε[t] = randn()
        σ²[t] = ω / (1 - sum(β))  # Stationary variance assumption
    end
    
    # GARCH recursion
    for t in (max_lag + 1):N
        σ²[t] = ω + sum(α .* ε[t .- (1:q)].^2) + sum(β .* σ²[t .- (1:p)])
        ε[t] = sqrt(σ²[t]) * randn()
    end
    
    return ε
end

# General AR(p) Model
function generate_ar(n::Int, coeffs::Vector{Float64}, sigma::Float64; seed=42)
    Random.seed!(seed)
    p = length(coeffs)  # Order of AR process
    x = zeros(n)
    x[1:p] = rand(Normal(0, sigma), p)  # Initialize with random values
    
    for t in (p + 1):n
        x[t] = sum(coeffs .* x[t-p:t-1]) + rand(Normal(0, sigma))
    end
    
    return x
end
# Lorenz system
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Rossler system
function rossler!(du, u, p, t)
    a, b, c = p
    x, y, z = u
    du[1] = -y - z
    du[2] = x + a * y
    du[3] = b + z * (x - c)
end

# Autoregressive (AR) model
function generate_ar_trajectory(a, Nf)
    trajectory = zeros(Nf)
    trajectory[1] = 0.01*randn()  # Initial condition
    for t in 2:Nf
        trajectory[t] = a * trajectory[t-1] + 0.01 * randn()
    end
    return trajectory
end

# 3D Autoregressive (AR) Model
function generate_3d_ar_trajectory(A, Nf)
    trajectory = zeros(Nf, 3)
    trajectory[1, :] = randn(3)  # Initial condition
    
    for t in 2:Nf
        trajectory[t, :] = A * trajectory[t-1, :] .+ randn(3)
    end
    
    return trajectory
end

# Noisy Lorenz system
function noisy_lorenz!(du, u, p, t)
    σ, ρ, β, noise_std = p
    du[1] = σ * (u[2] - u[1]) + noise_std * randn()
    du[2] = u[1] * (ρ - u[3]) - u[2] + noise_std * randn()
    du[3] = u[1] * u[2] - β * u[3] + noise_std * randn()
end

# 3D Hyperchaotic Logistic Map
function generate_3d_hyperchaotic_logistic_map(p, Nf)
    r, s = p
    # Initialize trajectory array
    trajectory = zeros(Nf, 3)
    
    # Random initial conditions in the range [0, 1]
    trajectory[1, :] = rand(3)
    
    # Iterate the map
    for t in 2:Nf
        x, y, z = trajectory[t-1, :]
        
        # 3D Hyperchaotic Logistic Map equations
        x_next = r * x * (1 - x) + s * y
        y_next = r * y * (1 - y) + s * z
        z_next = r * z * (1 - z) + s * x
        
        # Update trajectory
        trajectory[t, :] = [x_next, y_next, z_next]
    end
    
    return trajectory
end

#Logistic Map
function generate_logistic_map(r, Nf)
    # Initialize trajectory array
    trajectory = zeros(Nf)
    
    # Random initial conditions in the range [0, 1]
    trajectory[1] = rand()
    
    # Iterate the map
    for t in 2:Nf
        x = trajectory[t-1]
        
        # 3D Hyperchaotic Logistic Map equations
        x_next = r * x * (1 - x)
        
        # Update trajectory
        trajectory[t] = x_next
    end
    
    return trajectory
end


# Analyze a dynamical system
function analyze_system(system, params, Nf)
    Δt =  params[2]
    problem = ODEProblem(system, 2 * rand(3), (0.0, round(Int, 2 * Nf * params[2])), params[1])
    trajectory = Matrix(solve(problem, Tsit5(), dt = Δt, saveat = Δt, reltol = 1e-9, abstol = 1e-9, maxiters = 1e7)[:, end - Nf:end]')
    return trajectory
end

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

function plot_motifs_transition_joint_prob(probabilities, rr, LMAX; log_scale=true, figures_path=".")
    mkpath(figures_path)

    Xticks_nume = [-LMAX[1]:2:-2; 0 ; 2:2:LMAX[1]]
    Xticks_name = ["i".*string.(-LMAX[1]:2:-2); "i" ;"i+".*string.(2:2:LMAX[1])]
    Yticks_nume = [-LMAX[2]:2:-2; 0 ; 2:2:LMAX[2]]
    Yticks_name = ["j".*string.(-LMAX[2]:2:-2); "j" ;"j+".*string.(2:2:LMAX[2])]

    # Plot transition times matrix
    for motif_idx in 1:2^2
        Rij, R00 = divrem(motif_idx - 1, 2)
        
        probabilities_motif = R00==1 ? probabilities[:, :, motif_idx]/rr : probabilities[:, :, motif_idx]/(1-rr)
        if log_scale
            probabilities_motif = log.(probabilities_motif .+ 1e-10)
        end
    
        trans_matrix_plot = heatmap(-LMAX[1]:LMAX[1], -LMAX[2]:LMAX[2], 
                                    probabilities_motif, 
                                    aspect_ratio = 1, c = :viridis, xlabel = "i+i'", ylabel = "j+j'", 
                                    title = "P[R(i+i',j+j')=$(Rij) | R(i,j)= $(R00)]", colorbar_title = "Probability", 
                                    xticks = (Xticks_nume, Xticks_name), yticks = (Yticks_nume, Yticks_name),
                                    zlims = log_scale == true ? (-8,0) : (0,1),
                                    size = (800, 800), dpi=300, grid = false, transpose = true, frame_style=:box, widen=false)
        savefig(trans_matrix_plot, "$figures_path/joint_P_log_$(log_scale)_$(string(motif_idx-1, base=2)).png")
    end
end

function plot_motifs_transition_cond_prob(joint_probabilities, isolated_probabilities, LMAX; log_scale=true, figures_path=".")
    mkpath(figures_path)

    Xticks_nume = [-LMAX[1]:2:-2; 0 ; 2:2:LMAX[1]]
    Xticks_name = ["i".*string.(-LMAX[1]:2:-2); "i" ;"i+".*string.(2:2:LMAX[1])]
    Yticks_nume = [-LMAX[2]:2:-2; 0 ; 2:2:LMAX[2]]
    Yticks_name = ["j".*string.(-LMAX[2]:2:-2); "j" ;"j+".*string.(2:2:LMAX[2])]

    # Plot transition times matrix
    for motif_idx in 1:2^2
        Rij, R00 = divrem(motif_idx - 1, 2)

        if log_scale
            probabilities = log.(probabilities .+ 1e-10)
        end
    
        trans_matrix_plot = heatmap(-LMAX[1]:LMAX[1], -LMAX[2]:LMAX[2], 
                                    probabilities[:, :, motif_idx], 
                                    aspect_ratio = 1, c = :viridis, xlabel = "i+i'", ylabel = "j+j'", 
                                    title = "P[R(i+i',j+j')=$(Rij) | R(i,j)= $(R00)]", colorbar_title = "Probability", 
                                    xticks = (Xticks_nume, Xticks_name), yticks = (Yticks_nume, Yticks_name),
                                    size = (800, 800), dpi=300, grid = false, transpose = true, frame_style=:box, widen=false)
        savefig(trans_matrix_plot, "$figures_path/conditional_P_log_$(log_scale)_$(string(motif_idx-1, base=2)).png")
    end
end

# Function to plot mutual information
function plot_mutual_information(series, label; max_delay=50)
    delays = -max_delay:max_delay
    mi_values = [mutual_information(series, delay) for delay in delays]
    
    plot(delays, mi_values, label=label, xlabel="Time Delay", ylabel="Mutual Information", legend=:topright)
end

function main()
    # Parameters
    Nf = 2000
    LMAX = (42,42)
    resolution = 32
    rrs = [0.01; 0.05; 0.1; 0.2] # 10 .^ range(-4, -0.01, resolution)
    
    # 3D autoregressive model connection matrix
    A = [0.15  0.2  0.05;
         0.1  0.2  0.1;
         0.05  0.15  0.2]
    
    # Systems to analyze
    systems = [
        ("GARCH", nothing, 
        [0.01,
        [0.1, 0.05],  # ARCH(2)
        [0.7, 0.2, 0.05]]  # GARCH(3)
        , 1),  # ω, α, β
        ("AR(2)", nothing, [[0.7, -0.2], 0.5], 1),  # AR(2) with noise variance
        ("Lorenz traj", lorenz!, [[10.0, 28.0, 8 / 3], 0.2], [1,2,3]),
        ("Lorenz (x)", lorenz!, [[10.0, 28.0, 8 / 3], 0.2], 1),
        ("Lorenz (z)", lorenz!, [[10.0, 28.0, 8 / 3], 0.2], 3),
        ("Logistic 1D", nothing, 4.0, 1),
       # ("Logistic 3D", nothing, [3.711, 0.06], 1),
        ("Randn", nothing, nothing, 1),
        ("AR 0.1", nothing, 0.1, 1),
       # ("AR 0.3", nothing, 0.3, 1),
       # ("AR 0.8", nothing, 0.8, 1),
        ("AR 0.9", nothing, 0.9, 1),
       # ("3D AR 0.4", nothing, A, 1),
        ("Rossler traj", rossler!, [[0.2, 0.2, 5.7], 1], [1,2,3]),
        ("Rossler (x)", rossler!, [[0.2, 0.2, 5.7], 1], 1),
        ("Circle (sine)", nothing, 0.11347, 1)
    ]
    
    # Output directories
    data_path    = "data/MI_motifs_global_recur_$(today())/Nf$(Nf)_LMAX$(LMAX)"
    figures_path = "figures/MI_motifs_global_recur_$(today())/Nf$(Nf)_LMAX$(LMAX)"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component
    probabilities = zeros(length(systems), resolution, 2*LMAX[1]+1, 2*LMAX[2]+1, 2*2)  # Extra dimension for components
    
    for (i, system_tuple) in enumerate(systems)
        system_name, system, params, component = system_tuple
        
        println("Analyzing $system_name system...")

        system_path = figures_path*"/$system_name$(params)"
        mkpath(system_path)
        
        # Generate trajectory
        if system_name == "Randn"
            time_series = randn(Nf)
        
        elseif startswith(system_name, "AR 0.")
            time_series = generate_ar_trajectory(params, Nf)
        
        elseif occursin("3D AR", system_name)
            trajectory = generate_3d_ar_trajectory(params, Nf) # Generate system trajectory
            time_series = trajectory[:, component]  # Extract component
        
        elseif occursin("Logistic 3D", system_name)
            trajectory = generate_3d_hyperchaotic_logistic_map(params, Nf) # Generate system trajectory
            time_series = trajectory[:, component]  # Extract component
        
        elseif occursin("Logistic 1D", system_name)
            trajectory = generate_logistic_map(params, Nf) # Generate system trajectory
            time_series = trajectory
        
        elseif occursin("Circle", system_name)
            trajectory  = [sin.(2π * params * range(1, Nf)) cos.(2π * params * range(1, Nf))]
            time_series = trajectory[:, component]
        
        elseif system_name == "GARCH"
            time_series = generate_garch(Nf, params...)

        elseif system_name == "AR(2)"
            time_series = generate_ar(Nf, params[1], params[2])

        else
            trajectory = analyze_system(system, params, Nf)  # Generate system trajectory
            time_series = trajectory[:, component]  # Extract component
        end

        # Plot and save the time series
        plot(time_series, title = "$system_name Time Series", xlabel = "Time", ylabel = "Value", size=(1200,800))
        savefig(system_path*"/time_series.png")

        plot_mutual_information(time_series[:, 1], system_name; max_delay=LMAX[1])
        savefig(system_path*"/mutual_information.png")

        # Compute probabilities for each recurrence rate
        for idx in 1:length(rrs) # Create recurrence plot
            RP = RecurrenceMatrix(StateSpaceSet(time_series), GlobalRecurrenceRate(rrs[idx]); metric = Euclidean(), parallel = true)

            for (i_idx,iprime) in enumerate(ProgressBar(-LMAX[1]:LMAX[1]))
                Threads.@threads for j_idx in 1:length(-LMAX[2]:LMAX[2])
                    jprime = (-LMAX[2]:LMAX[2])[j_idx]
                    L = (iprime, jprime)
                    probabilities[i, idx, i_idx, j_idx, :] = motifs_probabilities(RP, L; shape=:timepair, sampling=:random, sampling_region=:upper, num_samples=0.05)
                end
            end
            plot_motifs_transition_joint_prob(probabilities[i, idx, :, :, :], rrs[idx], LMAX, log_scale=false, figures_path=system_path*"/rr$(rrs[idx])")
            plot_motifs_transition_joint_prob(probabilities[i, idx, :, :, :], rrs[idx], LMAX, log_scale=true, figures_path=system_path*"/rr$(rrs[idx])")
           
            # Plot and save the recurrence plot
            heatmap(RP, title = "$system_name Recurrence Plot", xlabel="Time", ylabel="Time",
            c=:grays, colorbar_title="Recurrence", size=(800, 600), dpi=200,
            colorbar=false, frame_style=:box, aspect_ratio=1, widen=false)
            savefig(system_path*"/rr$(rrs[idx])/recurrence_plot.png")
            #plot_motifs_transition_cond_prob(probabilities[i, idx, :, :, :], LMAX, log_scale=true, figures_path=figures_path*"/$system_name/rr$(rrs[idx])")
        end
    end
end

main()