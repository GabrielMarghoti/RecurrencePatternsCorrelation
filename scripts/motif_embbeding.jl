using RecurrenceAnalysis
using DynamicalSystems
using DifferentialEquations
using Plots
using Statistics
using ProgressBars
using LinearAlgebra
using LaTeXStrings
using DelayEmbeddings
using InformationMeasures

gr()

# Define constants
const SMALL_THRESHOLD = 1.0e-8

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

# Calculate microstate probabilities
function calculate_microstate_probabilities(data, L, rr)
    N = size(data, 1)
    num_microstates = 2^(L * L)
    histogram = fill(1.0e-30, num_microstates)

    # Ensure data is in the correct format for RecurrenceMatrix
    data_set = StateSpaceSet(data)
    RP = RecurrenceMatrix(data_set, GlobalRecurrenceRate(rr); metric = Euclidean(), parallel = true)

    for x in 1:(N - L + 1)
        for y in 1:(N - L + 1)
            microstate_id = 0
            for ly in 1:L, lx in 1:L
                if RP[x + lx - 1, y + ly - 1] == 1
                    microstate_id += 2^((ly - 1) * L + (lx - 1))
                end
            end
            histogram[Int(1 + microstate_id)] += 1
        end
    end

    probabilities = histogram / sum(histogram)
    probabilities[probabilities .< SMALL_THRESHOLD] .= 0
    return probabilities
end

# Analyze a dynamical system
function analyze_system(system, params, Nf, Δt)
    problem = ODEProblem(system, 2 * rand(3), (0.0, round(Int, 2 * Nf * Δt)), params)
    trajectory = solve(problem, Tsit5(), dt = Δt, saveat = Δt, reltol = 1e-9, abstol = 1e-9, maxiters = 1e7)
    return trajectory[1, end - Nf + 1:end]  # Extract the first dimension (x)
end

# Plot motif probabilities as a function of recurrence rate
function plot_motif_probabilities(rrs, probabilities, m_max, system_name, motif_idx, save_path, log__scale)
    if log__scale
        plot(
            rrs, probabilities[:, motif_idx, :],
            xlabel = "Recurrence Rate", ylabel = "Probability",
            label = string.((1:m_max)'), title="$(system_name) motif : $(string(motif_idx-1, base=2))",
            la = 0.9, size = (600, 400), dpi = 200,
            legend = :outerright, frame_style = :box,
            xscale = :log10, yscale = :log10, minorgrid = false, ylims = (0.00000001, 1.01), xlims = (0.00001, 1.01)
        )
        savefig("$save_path/log_log_motif_$(string(motif_idx-1, base=2)).png")
    else
        plot(
            rrs, probabilities[:, motif_idx, :],
            xlabel = "Recurrence Rate", ylabel = "Probability",
            label = string.((1:m_max)'), title="$(system_name) motif : $(string(motif_idx-1, base=2))",
            la = 0.9, size = (600, 400), dpi = 200,
            legend = :outerright, frame_style = :box,
            yscale = :log10, minorgrid = false, ylims = (0.00000001, 1.01)
        )
        savefig("$save_path/lin_log_motif_$(string(motif_idx-1, base=2)).png")
    end
end

# Main function
function main()
    # Parameters
    Nf = 3000
    Δt = 0.05
    L = 3
    m_max = 6
    resolution = 50
    rrs = 10 .^ range(-4, -0.01, resolution)
    max_delay = 100

    # Systems to analyze
    systems = [
        ("Random", nothing, nothing),
        ("AR", nothing, 0.1),  # AR model with memory parameter a = 0.1
        ("AR", nothing, 0.9),  # AR model with memory parameter a = 0.9
        ("Lorenz", lorenz!, [10.0, 28.0, 8 / 3]),
        ("Rossler", rossler!, [0.2, 0.2, 5.7])
    ]

    # Output directory
    base_output_dir = "/home/gabrielm/figuras/pik/motif_embedding_analysis/resol$(resolution)_Nf$(Nf)_L$(L)_mMax$(m_max)"
    mkpath(base_output_dir)

    # Analyze each system
    for (system_name, system, params) in ProgressBar(systems)
        println("Analyzing $system_name system...")

        # Create output directory for the system
        system_output_dir = "$base_output_dir/$(system_name)_$(params)"
        mkpath(system_output_dir)

        # Generate trajectory
        if system_name == "Random"
            trajectory = randn(Nf)
        elseif system_name == "AR"
            trajectory = generate_ar_trajectory(params, Nf)  # Generate AR trajectory
        else
            trajectory = analyze_system(system, params, Nf, Δt)
        end

        τ = estimate_delay(trajectory, "mi_min")  # Use mutual information to estimate delay
        println("Optimal tau: ", τ)

        probabilities = zeros(resolution, 2^(L^2), m_max)

        # Analyze for different embedding dimensions
        for m in 1:m_max
            println("  Embedding dimension m = $m")

            # Embed the trajectory if m > 1
            if m == 1
                embedded_trajectory = trajectory
            else
                embedded_trajectory = reconstruct(trajectory, m, τ)
            end

            # Calculate microstate probabilities for each recurrence rate
            for (idx, rr) in enumerate(rrs)
                probabilities[idx, :, m] = calculate_microstate_probabilities(embedded_trajectory, L, rr)
            end
            
            # Plot motif probabilities for each motif and system, different m's
            for motif_idx in 1:2^(L^2)
                plot_motif_probabilities(rrs, probabilities[:, motif_idx, :], m_max, system_name, motif_idx, system_output_dir, true)
                plot_motif_probabilities(rrs, probabilities[:, motif_idx, :], m_max, system_name, motif_idx, system_output_dir, false)
            end
        end
    end
end

# Run the main function
main()