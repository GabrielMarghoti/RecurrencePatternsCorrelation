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
x0 = [0.1]

# Time span
tspan = (0.0, 100.0)  # Longer time to see tipping event

# Parameters: noise level (σ), slow forcing rate (ε)
σ = 0.2  # Climate variability (adjust for sensitivity)
ε = 0.01 # Rate of external forcing (higher = faster climate change)
p = [σ, ε]

# Solve the SDE (using more stable SOSRI() solver)
prob = SDEProblem(tipping_sde, noise, x0, tspan, p)
sol = solve(prob, SOSRI(), dt=0.01)  # Higher-order solver for accuracy

# Plot the solution with improved aesthetics
plot_time_series = plot(sol,  # Access the first row of sol.u for plotting
    label="Climate State X(t)", xlabel="Time", ylabel="State", lw=2, 
    legend=:topright, grid=true, 
    frame_style=:box, size=(700,450), dpi=300)
title!("Climate Tipping Point Simulation")

# Define a portable save path
save_dir = joinpath(pwd(), "figures", "tipping_point")
mkpath(save_dir)  # Ensure directory exists

# Save the figure
savefig(plot_time_series, joinpath(save_dir, "timeseries.png"))
println("Figure saved to: ", joinpath(save_dir, "timeseries.png"))
