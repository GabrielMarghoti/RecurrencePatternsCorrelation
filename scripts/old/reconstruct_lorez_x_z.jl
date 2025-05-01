using DifferentialEquations
using DynamicalSystems
using RecurrenceAnalysis
using Plots

# Define the Lorenz system
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Parameters for the Lorenz system
σ = 10.0
ρ = 28.0
β = 8.0 / 3.0
u0 = [1.0, 0.0, 0.0]  # Initial conditions
p = (σ, ρ, β)
tspan = (0.0, 110.0)

# Solve the Lorenz system
prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=0.1)

# Extract components of the solution
x = sol[1, 100:end]
y = sol[2, 100:end]
z = sol[3, 100:end]

# Reconstruct the attractor using time-delay embedding
τ = 2  # Time delay
dim = 7 # Embedding dimension
reconstructed_x = embed(x, τ, dim)
reconstructed_z = embed(z, τ, dim)

# Create a recurrence plot
ε = 3.0  # Recurrence threshold
rp_x = RecurrenceMatrix(reconstructed_x, ε)
rp_z = RecurrenceMatrix(reconstructed_z, ε)

# Plot the recurrence plot
heatmap(rp_x, 
title="Recurrence Plot: x", xlabel="Time Index", ylabel="Time Index", color=:grays, 
colorbar=false, aspect_ratio=1, size=(500,450), dpi=200, widen=false)
savefig("/home/gabrielm/figuras/recurrence_plot_x.png")

# Plot the recurrence plot
heatmap(rp_z, 
title="Recurrence Plot: z", xlabel="Time Index", ylabel="Time Index", 
color=:grays, colorbar=false, aspect_ratio=1, size=(500,450), dpi=200, widen=false)
savefig("/home/gabrielm/figuras/recurrence_plot_z.png")
