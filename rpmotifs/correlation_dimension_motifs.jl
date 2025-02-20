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

# Define constants
const SMALL_THRESHOLD = 1.0e-8

function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

function rossler!(du, u, p, t)
    a, b, c = p
    x, y, z = u
    du[1] = -y - z
    du[2] = x + a * y
    du[3] = b + z * (x - c)
end

function calculate_microstate_probabilities(data, L, threshold)
    """Calculates microstate probabilities based on the recurrence matrix."""
    N = size(data, 1)
    num_microstates = 2^(L * L)
    histogram = fill(1.0e-30, num_microstates)

    RP = RecurrenceMatrix(StateSpaceSet(data), threshold; metric = Euclidean(), parallel = true)

    for x in 1:(N - L - 2)
        for y in 1:(x - L - 1)
            microstate_id = 0
            for ly in 1:L, lx in 1:L
                if RP[x + lx - 1, y + ly - 1] == 1
                    microstate_id += 2^((ly - 1) * L + lx - 1)
                end
            end
            histogram[Int(1 + microstate_id)] += 1
        end
    end

    probabilities = histogram / sum(histogram)
    probabilities[probabilities .< SMALL_THRESHOLD] .= 0
    return probabilities
end

function analyze_system(system, params, Nf, Δt)
    """Analyzes a dynamical system by calculating motif probabilities."""
    problem = ODEProblem(system, 2 * rand(3), (0.0, round(Int, 2 * Nf * Δt)), params)
    trajectory = solve(problem, Tsit5(), dt = Δt, saveat = Δt, reltol = 1e-9, abstol = 1e-9, maxiters = 1e7)[1, end - Nf:end]
    
    return (trajectory .- minimum(trajectory))/(maximum(trajectory)-minimum(trajectory))
end

function plot_motif_probabilities(eps, probabilities1, probabilitiesm, dim, expoen, save_path, log__scale)
    if log__scale==true
        """Plots and saves motif probabilities as log-log plots."""
        plot(
            eps, probabilitiesm,
            xlabel = "Threshold", ylabel = "Probability",
            label = ["Lorenz C($(dim))" "Rossler C($(dim))" "Stochas C($(dim))"],
            lc = [:blue :green :black],
            la = 0.9, size = (600, 400), dpi = 200, legend = :outerright, frame_style = :box,
            xscale = :log10, yscale = :log10, minorgrid = false, ylims = (0.00000001, 1.01)
        )
        plot!(
            eps, probabilities1 .^expoen,
            xlabel = "Threshold", ylabel = "Probability",
            label = ["Lorenz C(1)^$(expoen)" "Rossler C(1)^$(expoen)" "Stochas C(1)^$(expoen)"],
            lc = [:blue :green :black],
            ls = :dot,
            la = 1.0, size = (600, 400), dpi = 200, legend = :outerright, frame_style = :box,
            xscale = :log10, yscale = :log10, minorgrid = false,
        )
        savefig(save_path*"log_log.png")
    else
        """Plots and saves motif probabilities as NOT log plot."""
        plot(
            eps, probabilitiesm,
            xlabel = "Threshold", ylabel = "Probability",
            label = ["Lorenz C($(dim))" "Rossler C($(dim))" "Stochas C($(dim))"],
            lc = [:blue :green :black],
            la = 0.9, size = (600, 400), dpi = 200, legend = :outerright, frame_style = :box,
            yscale = :log10, minorgrid = false, ylims = (0.00000001, 1.01)
        )
        plot!(
            eps, probabilities1 .^expoen,
            xlabel = "Threshold", ylabel = "Probability",
            label = ["Lorenz C(1)^$(expoen)" "Rossler C(1)^$(expoen)" "Stochas C(1)^$(expoen)"],
            lc = [:blue :green :black],
            ls = :dot,
            la = 1.0, size = (600, 400), dpi = 200, legend = :outerright, frame_style = :box,
            yscale = :log10, minorgrid = false,
        )
        savefig(save_path*"lin_log.png")
    end
end

function mutual_information(data, max_delay)
    """Calculates the mutual information for different delays."""
    mi_values = zeros(max_delay)
    for τ in 1:max_delay
        shifted_data = circshift(data, -τ)
        mi_values[τ] = get_mutual_information(data, shifted_data)  # Use mutualinfo from InformationMeasures
    end
    return mi_values
end

function find_optimal_delay(data, max_delay)
    """Finds the optimal delay using the mutual information method."""
    mi_values = mutual_information(data, max_delay)
    # The optimal delay is the first local minimum of the mutual information
    optimal_delay = findfirst(diff(mi_values) .> 0)
    return optimal_delay
end

function main()
    # Parameters
    Nf = 3000
    Δt = 0.05
    L = 2
    dim = 3
    expoen = 1
    resolution = 50
    eps = 10 .^ range(-4, 0, resolution)
    max_delay = 100

    # Output directory
    output_dir = "/home/gabrielm/figuras/pik/motif_correlation_dimension_L$(L)_dim$(dim)_exp$(expoen)"
    mkpath(output_dir)

    # Allocate storage for probabilities
    num_microstates = 2^(L^2)
    probabilities_m1 = zeros(resolution, num_microstates, 3)
    probabilities_m3 = zeros(resolution, num_microstates, 3)

    lorenz_traj  = analyze_system(lorenz!, [10.0, 28.0, 8 / 3], Nf, Δt)
    rossler_traj = analyze_system(rossler!, [0.2, 0.2, 5.7], Nf, Δt)

    random_data = randn(Nf)
    random_data = (random_data .- minimum(random_data))/(maximum(random_data)-minimum(random_data))

    # Find optimal delays
    τ_lorenz = find_optimal_delay(lorenz_traj, max_delay)
    τ_rossler = find_optimal_delay(rossler_traj, max_delay)
    τ_random = find_optimal_delay(random_data, max_delay)

    println("Optimal delay for Lorenz system: ", τ_lorenz)
    println("Optimal delay for Rossler system: ", τ_rossler)
    println("Optimal delay for random data: ", τ_random)

    # Compute probabilities
    for idx in ProgressBar(1:resolution)
        threshold = eps[idx]
        probabilities_m1[idx, :, 1] = calculate_microstate_probabilities(lorenz_traj,  L, threshold)
        probabilities_m1[idx, :, 2] = calculate_microstate_probabilities(rossler_traj, L, threshold)
        probabilities_m1[idx, :, 3] = calculate_microstate_probabilities(random_data,  L, threshold)

        threshold = eps[idx]
        probabilities_m3[idx, :, 1] = calculate_microstate_probabilities(embed(lorenz_traj,  dim, τ_lorenz),  L, threshold)
        probabilities_m3[idx, :, 2] = calculate_microstate_probabilities(embed(rossler_traj,  dim, τ_rossler), L, threshold)
        probabilities_m3[idx, :, 3] = calculate_microstate_probabilities(embed(random_data,  dim, τ_random),  L, threshold)
    end

    for motif_idx=1:num_microstates
        plot_motif_probabilities(eps, probabilities_m1[:, motif_idx, :], probabilities_m3[:, motif_idx, :], dim, expoen, "$output_dir/motif_idx$(string(motif_idx-1, base=2))", true)
        plot_motif_probabilities(eps, probabilities_m1[:, motif_idx, :], probabilities_m3[:, motif_idx, :], dim, expoen, "$output_dir/motif_idx$(string(motif_idx-1, base=2))", false)
    end

    return
end

main()