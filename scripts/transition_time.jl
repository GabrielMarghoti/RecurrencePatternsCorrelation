# transition_time.jl

using DynamicalSystems
using RecurrenceAnalysis
using Plots
using Measures
using ProgressBars
using Dates


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
    trajectory[1] = randn()  # Initial condition
    for t in 2:Nf
        trajectory[t] = a * trajectory[t-1] + randn()
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
function analyze_system(system, params, Nf, Δt)
    problem = ODEProblem(system, 2 * rand(3), (0.0, round(Int, 2 * Nf * Δt)), params)
    trajectory = Matrix(solve(problem, Tsit5(), dt = Δt, saveat = Δt, reltol = 1e-9, abstol = 1e-9, maxiters = 1e7)[:, end - Nf:end]')
    return trajectory
end

function plot_motifs_transition_times(probabilities, LMAX, log_scale=True, figures_path)
        
        mkpath(figures_path)

        # Plot transition times matrix
        for motif_idx=1:2^2
        trans_matrix_plot = heatmap(1:LMAX[1], 1:LMAX[2], 
                            probabilities[:, :, motif_idx]', 
                            aspect_ratio = 1, c = :viridis, xlabel = "i'", ylabel = "j'", 
                            title = "Transition Times Matrix", colorbar_title = "Probability", 
                            xticks = (1:5:LMAX[1], 'i'.*string.(1:5:LMAX[1])), yticks = (1:5:LMAX[2], 'j'.*string.(1:5:LMAX[2])),
                            size = (800, 800), dpi=300, grid = false,
                            yscale = log_scale ? :log10 : :linear , minorgrid = false, ylims = (1e-8, 1.01))
        savefig(trans_matrix_plot, "$figures_path/log_$(log_scale)-all_systems_motif_$(string(motif_idx-1, base=2)).png")
 
        end
end


function main()
    # Parameters
    Nf = 1000
    Δt = 0.05
    LMAX = (100,100)
    resolution = 32
    rrs = [0.01; 0.1; 0.2; 0.5] # 10 .^ range(-4, -0.01, resolution)
    
    # 3D autoregressive model connection matrix
    A = [0.0  0.0  0.0;
         0.0  0.0  0.0;
         0.0  0.0  0.0]
    
    # Systems to analyze
    systems = [
        ("Logistic 1D", nothing, 3.6, 1),
        ("Logistic 3D", nothing, [3.7, 0.1], 1),
        ("Randn", nothing, nothing, 1),
        ("AR 0.1", nothing, 0.1, 1),
        ("3D AR 0.1", nothing, A, 1),
        ("Lorenz (x)", lorenz!, [10.0, 28.0, 8 / 3], 1),
        ("Lorenz (z)", lorenz!, [10.0, 28.0, 8 / 3], 3),
        ("Rossler (x)", rossler!, [0.2, 0.2, 5.7], 1),
        ("Circle (sine)", nothing, 0.01, 1)
    ]
    
    # Output directories
    data_path    = "data/motifs_transition_times_$(today())/Nf$(Nf)_LMAX$(LMAX)"
    figures_path = "figures/motifs_transition_times_$(today())/Nf$(Nf)_LMAX$(LMAX)"

    mkpath(data_path)
    mkpath(figures_path)
    
    # Adjust array to store results for each system and component
    probabilities = zeros(length(systems), resolution, LMAX[1], LMAX[2], 2*2)  # Extra dimension for components
    
    for (i, system_tuple) in enumerate(systems)
        system_name, system, params, component = system_tuple
        
        println("Analyzing $system_name system...")
        
        # Generate trajectory
        if system_name == "Randn"
            time_series = randn(Nf)
        
        elseif startswith(system_name, "AR")
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
        
        else
            trajectory = analyze_system(system, params, Nf, Δt)  # Generate system trajectory
            time_series = trajectory[:, component]  # Extract component
        end
        
        # Compute probabilities for each recurrence rate
        for idx in 1:length(rrs) # Create recurrence plot
            RP = RecurrenceMatrix(StateSpaceSet(time_series), GlobalRecurrenceRate(rr); metric = Euclidean(), parallel = true)
            for iprime in 1:LMAX[1]
                for jprime in 1:LMAX[2]
                    L = (iprime, jprime)
                    rr = rrs[idx]
                    probabilities[i, idx, iprime, jprime, :] = motifs_probabilities(RP, L; shape=:timepair, sampling=:full, num_samples=1.0)
                end
            end
            plot_motifs_transition_times(probabilities[i, idx, :, :, :], systems, LMAX, figures_path*"/$system_name/rr$(rrs[idx])")
        end
    end
end

main()