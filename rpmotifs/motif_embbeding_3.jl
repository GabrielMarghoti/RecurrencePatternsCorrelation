using RecurrenceAnalysis
using DynamicalSystems
using DifferentialEquations
using Plots
using StatsPlots
using Measures
using Statistics
using ProgressBars
using LinearAlgebra
using LaTeXStrings
using DelayEmbeddings
using InformationMeasures
using Distances 
using StatsBase  


gr()

# Function to compute Hellinger distance
function hellinger_distance(p, q)
    return sqrt(0.5 * sum((sqrt.(p) .- sqrt.(q)).^2))
end

# Define constants
const SMALL_THRESHOLD = 1.0e-8
function find_optimal_delay(trajectory, max_delay; method="mi_min", binning=10)
    """
    Find the optimal delay for embedding a time series.

    Parameters:
    - trajectory: The time series (1D array).
    - max_delay: The maximum delay to consider.
    - method: The method to use for finding the optimal delay. Options:
        - "mi_min": First minimum of mutual information.
        - "mi_zero": First zero crossing of mutual information.
    - binning: Number of bins for mutual information calculation.

    Returns:
    - τ: The optimal delay.
    """
    delays = 1:max_delay
    mi_values = zeros(length(delays))

    # Calculate mutual information for each delay
    for (i, τ) in enumerate(delays)
        shifted_trajectory = trajectory[1+τ:end]
        original_trajectory = trajectory[1:end-τ]
        mi_values[i] = get_mutual_information(original_trajectory, shifted_trajectory; base=2)
    end

    # Find the optimal delay based on the chosen method
    if method == "mi_min"
        # Find the first minimum of mutual information
        τ = findfirst(diff(mi_values) .> 0)
        if τ === nothing
            τ = max_delay  # Default to max_delay if no minimum is found
        end
    elseif method == "mi_zero"
        # Find the first zero crossing of mutual information
        τ = findfirst(mi_values .< 0)
        if τ === nothing
            τ = max_delay  # Default to max_delay if no zero crossing is found
        end
    else
        error("Unknown method: $method. Use 'mi_min' or 'mi_zero'.")
    end

    return τ
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

    RP = RecurrenceMatrix(StateSpaceSet(data), GlobalRecurrenceRate(rr); metric = Euclidean(), parallel = true)

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

# Analyze a dynamical system
function analyze_system(system, params, Nf, Δt)
    problem = ODEProblem(system, 2 * rand(3), (0.0, round(Int, 2 * Nf * Δt)), params)
    trajectory = Matrix(solve(problem, Tsit5(), dt = Δt, saveat = Δt, reltol = 1e-9, abstol = 1e-9, maxiters = 1e7)[:, end - Nf:end]')
    return trajectory
end

# Plot motif probabilities as a function of recurrence rate
function plot_motif_probabilities(rrs, probabilities, L, m_max, systems, save_path, log_scale)
    # Create directories if they don't exist
    if !isdir(save_path)
        mkdir(save_path)
    end

    # Define layout with one row and multiple columns for each system
    num_systems = length(systems)
    layout = (num_systems)  # 1 row, multiple columns

    for motif_idx in 1:2^(L^2)
        plt_all_systems = plot(layout=layout, size=(1200, 800), dpi=200)        

        for (i, (system_name, system, params)) in enumerate(systems)
            if log_scale
                plot!(plt_all_systems[i], rrs, probabilities[i,:, motif_idx, :],
                    xlabel = "Recurrence Rate", ylabel = (i == 1) ? "Probability" : "",
                    label = ["Orig" "m=".*string.((1:m_max)')], title="$(system_name) - Motif : $(string(motif_idx-1, base=2))",
                    la = 0.9, legend = (i == num_systems) ? :outerright : false, frame_style = :box,
                    xscale = :log10, yscale = :log10, minorgrid = false, ylims = (1e-8, 1.01), xlims = (1e-5, 1.01)
                )
            else
                plot!(plt_all_systems[i], rrs, probabilities[i,:, motif_idx, :],
                    xlabel = "Recurrence Rate", ylabel = (i == 1) ? "Probability" : "",
                    label = ["Orig" "m=".*string.((1:m_max)')], title="$(system_name) - Motif : $(string(motif_idx-1, base=2))",
                    la = 0.9, legend = (i == num_systems) ? :outerright : false, frame_style = :box,
                    yscale = :log10, minorgrid = false, ylims = (1e-8, 1.01)
                )
            end

            if i == num_systems
                # Convert motif index to binary (with padding to length L^2)
                binary_string = digits(motif_idx - 1, base=2, pad=L^2)

                # Group into chunks of L elements
                grouped_binary = [join(binary_string[i:min(i+L-1, end)]) for i in 1:L:length(binary_string)]

                # Join groups with "\n" to enforce line breaks
                motif_label = join(grouped_binary, "\n")

                annotate!(plt_all_systems[i], 1.1, 0.4, text(motif_label, 10, :left))
            end
        end

        savefig(plt_all_systems, "$save_path/log_$(log_scale)-all_systems_motif_$(string(motif_idx-1, base=2)).png")
    end


    # Plot each system separately
        for (i, (system_name, system, params)) in enumerate(systems)
        
        println("Plotting $system_name system...")

        if !isdir(save_path*"/$(system_name)_$(params)/")
            mkdir(save_path*"/$(system_name)_$(params)/")
        end
        for motif_idx in 1:2^(L^2)
            if log_scale
                plot_sys_fix_motif = plot(
                    rrs, probabilities[i,:, motif_idx, :],
                    xlabel = "Recurrence Rate", ylabel = "Probability",
                    label = ["Orig" "m=".*string.((1:m_max)')], title="$(system_name) motif : $(string(motif_idx-1, base=2))",
                    la = 0.9, size = (600, 400), dpi = 200,
                    legend = :outerright, frame_style = :box,
                    xscale = :log10, yscale = :log10, minorgrid = false, ylims = (0.00000001, 1.01), xlims = (0.00001, 1.01)
                )
                savefig(plot_sys_fix_motif, "$save_path/$(system_name)_$(params)/log_log_$(system_name)_motif_$(string(motif_idx-1, base=2)).png")
            else
                plot_sys_fix_motif = plot(
                    rrs, probabilities[i,:, motif_idx, :],
                    xlabel = "Recurrence Rate", ylabel = "Probability",
                    label = ["Orig" "m=".*string.((1:m_max)')], title="$(system_name) motif : $(string(motif_idx-1, base=2))",
                    la = 0.9, size = (600, 400), dpi = 200,
                    legend = :outerright, frame_style = :box,
                    yscale = :log10, minorgrid = false, ylims = (0.00000001, 1.01)
                )
                savefig(plot_sys_fix_motif, "$save_path/$(system_name)_$(params)/lin_log_$(system_name)_motif_$(string(motif_idx-1, base=2)).png")
            end
        end
    end

    # Plot panels for fixed m and different motifs as curves
        for (i, (system_name, system, params)) in enumerate(systems)
            
        for m_idx in 1:(m_max+1)
            plt_fixed_m = plot()
            for motif_idx in 1:2^(L^2)
                if log_scale
                    plot!(plt_fixed_m, rrs, probabilities[i,:, motif_idx, m_idx],
                        xlabel = "Recurrence Rate", ylabel = "Probability",
                        label = "Motif $(string(motif_idx-1, base=2))", title = (m_idx == 1) ? "Orig" : "$(m_idx-1)",
                        la = 0.9, size = (600, 400), dpi = 200,
                        legend = :outerright, frame_style = :box,
                        xscale = :log10, yscale = :log10, minorgrid = false, ylims = (0.00000001, 1.01), xlims = (0.00001, 1.01)
                    )
                else
                    plot!(plt_fixed_m, rrs, probabilities[i,:, motif_idx, m_idx],
                        xlabel = "Recurrence Rate", ylabel = "Probability",
                        label = "Motif $(string(motif_idx-1, base=2))", title="m=$(m_idx-1)",
                        la = 0.9, size = (600, 400), dpi = 200,
                        legend = :outerright, frame_style = :box,
                        yscale = :log10, minorgrid = false, ylims = (0.00000001, 1.01)
                    )
                end
            end
            savefig(plt_fixed_m, "$save_path/$(system_name)_$(params)/fixed_m_$(m_idx-1).png")
        end
    end
end


# Plot Hellinger distance between probability distributions
function plot_hellinger_distance(rrs, probabilities, L, m_max, systems, save_path)
    if !isdir(save_path)
        mkdir(save_path)
    end

    num_systems = length(systems)
    system_names = [system[1] for system in systems]
    colors = distinguishable_colors(num_systems)  # Assign distinct colors to each system

    hellinger_data = zeros(m_max*num_systems)
    m_values = zeros(m_max*num_systems)
    color_bars_plot = Vector{RGB}(undef, m_max*num_systems)  # Corrected initialization

    system_offsets = ((0:num_systems-1) .- (num_systems-1)/2)  * 0.1
    group_systems_label = Vector{String}(undef, m_max*num_systems)

    for (i, (system_name, system, params)) in enumerate(systems)
        for m in 1:m_max
            hellinger_distances = []

            for rr_idx in 1:length(rrs)
                p_ref = probabilities[i, rr_idx, :, 1]  # Reference distribution (m=1)
                p_m = probabilities[i, rr_idx, :, m+1]  # Distribution for m
                push!(hellinger_distances, hellinger_distance(p_ref, p_m))
            end

            hellinger_data[(m-1)*num_systems+i] = mean(hellinger_distances)  # Average over motifs
            color_bars_plot[(m-1)*num_systems+i] = colors[i]  # Assign color for each system

            m_values[(m-1)*num_systems+i] = m + system_offsets[i]
            group_systems_label[(m-1)*num_systems+i] =system_name
        end
    end

    plt = bar(
        m_values, hellinger_data,
        xlabel="m", ylabel="Avg. Hellinger Distance",
        group=group_systems_label,  # Correct grouping
        title="Distance between embedded and original data's recurrence motif probability distributions",
        bar_width=0.08, legend=:topright, size=(1200, 600), dpi=300,
        color=color_bars_plot, xticks=1:m_max,
        frame_style=:box, grid=false,
        bottom_margin = 5mm,
        left_margin = 5mm,
    )

    savefig(plt, "$save_path/hellinger_distance_all_systems.png")
end

# Plot Hellinger distance between probability distributions, grouped by system
function plot_hellinger_distance_grouped_by_system(rrs, probabilities, L, m_max, systems, save_path)
    if !isdir(save_path)
        mkdir(save_path)
    end

    num_systems = length(systems)
    system_names = [system[1] for system in systems]
    colors = distinguishable_colors(num_systems)  # Assign distinct colors to each system

    hellinger_data = zeros(m_max*num_systems)
    x_axis_values = zeros(m_max*num_systems)
    color_bars_plot = Vector{RGB}(undef, m_max*num_systems)  # Corrected initialization

    m_offsets = ((0:m_max-1) .- (m_max-1)/2)  * 0.1
    group_m_label = Vector{String}(undef, m_max*num_systems)

    for (i, (system_name, system, params)) in enumerate(systems)
        for m in 1:m_max
            hellinger_distances = []

            for rr_idx in 1:length(rrs)
                p_ref = probabilities[i, rr_idx, :, 1]  # Reference distribution (m=1)
                p_m = probabilities[i, rr_idx, :, m+1]  # Distribution for m
                push!(hellinger_distances, hellinger_distance(p_ref, p_m))
            end

            hellinger_data[(i-1)*m_max+m] = mean(hellinger_distances)  # Average over motifs
            color_bars_plot[(i-1)*m_max+m] = colors[m]  # Assign color for each system

            x_axis_values[(i-1)*m_max+m] = i + m_offsets[m]
            group_m_label[(i-1)*m_max+m] = string(m)
        end
    end

    plt = bar(
        x_axis_values, hellinger_data,
        xlabel="System", ylabel="Avg. Hellinger Distance",
        group=group_m_label,  # Correct grouping
        title="Distance between embedded and original data's recurrence motif probability distributions",
        bar_width=0.08, legend=:topright, size=(1200, 600), dpi=300,
        color=color_bars_plot, xticks=(1:num_systems, system_names),
        frame_style=:box, grid=false,
        bottom_margin = 5mm,
        left_margin = 5mm,
    )

    savefig(plt, "$save_path/hellinger_distance_grouped_by_system.png")
end


# Main function
function main()
    # Parameters
    Nf = 1000
    Δt = 0.05
    L = 3
    m_max = 4
    resolution = 40
    rrs = 10 .^ range(-4, -0.01, resolution)
    max_delay = 100

    # Systems to analyze
    systems = [
        ("Randn", nothing, nothing, 1),
        ("AR 0.1", nothing, 0.1, 1),
        ("AR 0.9", nothing, 0.9, 1),
        ("Lorenz (x)", lorenz!, [10.0, 28.0, 8 / 3], 1),
        ("Lorenz (y)", lorenz!, [10.0, 28.0, 8 / 3], 2),
        ("Lorenz (z)", lorenz!, [10.0, 28.0, 8 / 3], 3),
        ("Rossler (x)", rossler!, [0.2, 0.2, 5.7], 1),
        ("Circle (sine)", nothing, 0.01, 1)
    ]

    # Output directory
    base_output_dir = "motif_embedding_analysis_Feb_10/resol$(resolution)_Nf$(Nf)_L$(L)_mMax$(m_max)"
    mkpath(base_output_dir)

    # Adjust array to store results for each system and component
    probabilities = zeros(length(systems), resolution, 2^(L^2), m_max+1)  # Extra dimension for components

    for (i, system_tuple) in enumerate(systems)
        system_name, system, params, component = system_tuple

        println("Analyzing $system_name system...")

        # Generate trajectory
        if system_name == "Randn"
            time_series = randn(Nf)

            # Compute probabilities for each recurrence rate
            Threads.@threads for idx in 1:length(rrs)
                probabilities[i, idx, :, 1] = calculate_microstate_probabilities(time_series, L, rrs[idx])
            end

        elseif occursin("AR", system_name)
            time_series = generate_ar_trajectory(params, Nf)

            # Compute probabilities for each recurrence rate
            Threads.@threads for idx in 1:length(rrs)
                probabilities[i, idx, :, 1] = calculate_microstate_probabilities(time_series, L, rrs[idx])
            end

        elseif occursin("Circle", system_name)
            trajectory  = [sin.(2π * params[1] * range(1, Nf)) cos.(2π * params[1] * range(1, Nf))]
            time_series = trajectory[:, component]


            # Compute probabilities for each recurrence rate
            Threads.@threads for idx in 1:length(rrs)
                probabilities[i, idx, :, 1] = calculate_microstate_probabilities(trajectory, L, rrs[idx])
            end

        else
            trajectory = analyze_system(system, params, Nf, Δt)  # Generate system trajectory
            time_series = trajectory[:, component]  # Extract component

            # Compute probabilities for each recurrence rate
            Threads.@threads for idx in 1:length(rrs)
                probabilities[i, idx, :, 1] = calculate_microstate_probabilities(trajectory, L, rrs[idx])
            end
        end

        τ = find_optimal_delay(time_series, max_delay)
        println("  Optimal τ for component $component: ", τ)

        # Analyze for different embedding dimensions
        for m in 1:m_max
            println("  Embedding dimension m = $m")

            # Embed trajectory if m > 1
            embedded_ts = (m == 1) ? time_series : embed(time_series, m, τ)

            # Compute probabilities
            Threads.@threads for idx in 1:length(rrs)
                probabilities[i, idx, :, m+1] = calculate_microstate_probabilities(embedded_ts, L, rrs[idx])
            end
        end
    end

    # Plot motif probabilities for each motif and system, 
    plot_hellinger_distance(rrs, probabilities, L, m_max, systems, base_output_dir)
    plot_hellinger_distance_grouped_by_system(rrs, probabilities, L, m_max, systems, base_output_dir)


    plot_motif_probabilities(rrs, probabilities, L, m_max, systems, base_output_dir, true)
    plot_motif_probabilities(rrs, probabilities, L, m_max, systems, base_output_dir, false)
end


# Run the main function
main()