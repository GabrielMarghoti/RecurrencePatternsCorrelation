using DifferentialEquations, Plots, Random, FileIO, Statistics
using RecurrenceAnalysis  # Ensure this package is installed for recurrence analysis

# Define the SDE function with slow forcing α(t)
function tipping_sde(dx, x, p, t)
    σ, ε = p  # Parameters: noise level, forcing rate
    α = ε * t  # Slow drift forcing
    dx[1] = (x[1] - x[1]^3 + α)  # Modified drift term
end

function noise(dx, x, p, t)
    σ, ε = p
    dx[1] = σ  # Diffusion term (stochastic noise)
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
                size = (800, 800), dpi=300, grid = false, transpose = true, frame_style=:box, widen=false,
                xrotation = 50)
        savefig(trans_matrix_plot, "$figures_path/joint_P_log_$(log_scale)_$(string(motif_idx-1, base=2)).png")
    end
end

# Initial condition
x0 = [-0.9]

# Time span (extended to a longer duration)
tspan = (0.0, 600.0)  # Longer time to see tipping event

# Parameters: noise level (σ), slow forcing rate (ε)
σ = 0.1  # Climate variability (adjust for sensitivity)
ε = 0.002 # Rate of external forcing (higher = faster climate change)
p = [σ, ε]

# Split the time series into segments of 2000 points
segment_size = 2000

# Recurrence analysis parameters
rrs = [0.05, 0.1, 0.2]  # Example recurrence rates
LMAX = (32, 32)  # Example limits


# Solve the SDE (using more stable SOSRI() solver)
prob = SDEProblem(tipping_sde, noise, x0, tspan, p)
sol = solve(prob, SOSRI(), dt=0.01)  # Higher-order solver for accuracy

# Extract time and state values
t = sol.t
x = [u[1] for u in sol.u]

# Compute the drifting parameter α(t)
α = ε .* t

# Plot the solution and drifting parameter
plot_time_series = plot(t, x, 
    label="Climate State X(t)", xlabel="Time", ylabel="State", lw=2, 
    legend=:topright, grid=true, 
    frame_style=:box, size=(900,450), dpi=300, color=:blue)

# Add α(t) to the plot with a secondary y-axis
plot!(twinx(), t, α, 
    label="Drifting Parameter α(t)", ylabel="α(t)", lw=2, 
    color=:red, linestyle=:dash, legend=:bottomright)

title!("Climate Tipping Point Simulation with Drifting Parameter")

# Define a portable save path
save_dir = joinpath(pwd(), "figures", "tipping_point","p$(p)_tspan$(tspan)_segsize$(segment_size)")
mkpath(save_dir)  # Ensure directory exists

# Save the figure
savefig(plot_time_series, joinpath(save_dir, "timeseries_with_alpha.png"))
println("Figure saved to: ", joinpath(save_dir, "timeseries_with_alpha.png"))

num_segments = length(x) ÷ segment_size
segments = [x[(i-1)*segment_size+1:i*segment_size] for i in 1:num_segments]
segment_times = [t[(i-1)*segment_size+1:i*segment_size] for i in 1:num_segments]
# Plot the full time series with segments highlighted
plot_full_time_series = plot(t, x, 
    label="Climate State X(t)", xlabel="Time", ylabel="State", lw=2, 
    legend=:topright, grid=true, 
    frame_style=:box, size=(900,450), dpi=300, color=:blue)

# Highlight each segment and add labels in the middle of the highlighted area
for (segment_idx, (segment_t, segment_x)) in enumerate(zip(segment_times, segments))
    # Highlight the segment
    plot!(segment_t, segment_x, 
        fillrange=minimum(x), fillalpha=0.3, linealpha=0, 
        label=false, color=segment_idx)
    
    # Calculate the middle of the segment for the label
    middle_time = (segment_t[1] + segment_t[end]) / 2  # Middle time of the segment
    middle_value = (mean(segment_x) < 0) ? mean(segment_x) + 0.4 : mean(segment_x) - 0.4
    
    # Add the label in the middle of the highlighted area
    annotate!(middle_time, middle_value, text("Seg.$segment_idx", :center, 8, :black))
end

title!("Full Time Series with Segments Highlighted")
savefig(plot_full_time_series, joinpath(save_dir, "full_timeseries_with_segments.png"))
println("Figure saved to: ", joinpath(save_dir, "full_timeseries_with_segments.png"))

# Perform analysis on each segment
for (segment_idx, segment) in enumerate(segments)
    println("Analyzing segment $segment_idx of $num_segments")

    # Create a subdirectory for this segment
    segment_dir = joinpath(save_dir, "segment_$segment_idx")
    mkpath(segment_dir)

    # Initialize probabilities for this segment
    probabilities = zeros(length(rrs), length(-LMAX[1]:LMAX[1]), length(-LMAX[2]:LMAX[2]), 4)

    for idx in 1:length(rrs)  # Create recurrence plot
        RP = RecurrenceMatrix(StateSpaceSet(segment), GlobalRecurrenceRate(rrs[idx]); metric = Euclidean(), parallel = true)

        for (i_idx, iprime) in enumerate(-LMAX[1]:LMAX[1])
            Threads.@threads for j_idx in 1:length(-LMAX[2]:LMAX[2])
                jprime = (-LMAX[2]:LMAX[2])[j_idx]
                L = (iprime, jprime)
                probabilities[idx, j_idx, i_idx, :] = motifs_probabilities(RP, L; shape=:timepair, sampling=:random, sampling_region=:upper, num_samples=0.1)
            end
        end
        segment_dir_rr = joinpath(segment_dir,"rr$(rrs[idx])_LMAX$(LMAX)")
        mkpath(segment_dir_rr)

        # Plot and save motif transition probabilities
        plot_motifs_transition_joint_prob(probabilities[idx, :, :, :], rrs[idx], LMAX, log_scale=false, figures_path=segment_dir_rr)
        plot_motifs_transition_joint_prob(probabilities[idx, :, :, :], rrs[idx], LMAX, log_scale=true, figures_path=segment_dir_rr)

        # Plot and save the recurrence plot
        heatmap(RP, title = "Recurrence Plot (Segment $segment_idx)", xlabel="Time", ylabel="Time",
        c=:grays, colorbar_title="Recurrence", size=(800, 600), dpi=200,
        colorbar=false, frame_style=:box, aspect_ratio=1, widen=false)
        savefig(joinpath(segment_dir_rr, "recurrence_plot.png"))
    end
end