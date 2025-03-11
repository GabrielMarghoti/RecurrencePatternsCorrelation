using DifferentialEquations, Plots, Random, FileIO

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

# Initial condition
x0 = [-0.8]

# Time span (extended to a longer duration)
tspan = (0.0, 500.0)  # Longer time to see tipping event

# Parameters: noise level (σ), slow forcing rate (ε)
σ = 0.2  # Climate variability (adjust for sensitivity)
ε = 0.001 # Rate of external forcing (higher = faster climate change)
p = [σ, ε]

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
save_dir = joinpath(pwd(), "figures", "tipping_point")
mkpath(save_dir)  # Ensure directory exists

# Save the figure
savefig(plot_time_series, joinpath(save_dir, "timeseries_with_alpha.png"))
println("Figure saved to: ", joinpath(save_dir, "timeseries_with_alpha.png"))