module DynamicalSystemsToolkit

using Random
using Distributions
using DifferentialEquations

export generate_garch,
       generate_ar,
       generate_ar_trajectory,
       generate_3d_ar_trajectory,
       generate_logistic_map,
       generate_3d_hyperchaotic_logistic_map,
       lorenz!,
       noisy_lorenz!,
       rossler!,
       tipping_sde,
       analyze_system,
       analyze_sde_system


# GARCH model generator
function generate_garch(N, ω, α, β)
    p = length(β)
    q = length(α)
    max_lag = max(p, q)

    ε = zeros(N)
    σ² = zeros(N)

    for t in 1:max_lag
        ε[t] = randn()
        σ²[t] = ω / (1 - sum(β))
    end

    for t in (max_lag + 1):N
        σ²[t] = ω + sum(α .* ε[t .- (1:q)].^2) + sum(β .* σ²[t .- (1:p)])
        ε[t] = sqrt(σ²[t]) * randn()
    end

    return ε
end

# General AR(p) model
function generate_ar(n::Int, coeffs::Vector{Float64}, sigma::Float64)
    p = length(coeffs)
    x = zeros(n)
    x[1:p] = rand(Normal(0, sigma), p)

    for t in (p + 1):n
        x[t] = sum(coeffs .* x[t-p:t-1]) + rand(Normal(0, sigma))
    end

    return x
end

# 1D AR trajectory
function generate_ar_trajectory(a, Nf)
    trajectory = zeros(Nf)
    trajectory[1] = 0.01 * randn()
    for t in 2:Nf
        trajectory[t] = a * trajectory[t-1] + 0.01 * randn()
    end
    return trajectory
end

# 3D AR trajectory
function generate_3d_ar_trajectory(A, Nf)
    trajectory = zeros(Nf, 3)
    trajectory[1, :] = randn(3)
    for t in 2:Nf
        trajectory[t, :] = A * trajectory[t-1, :] .+ randn(3)
    end
    return trajectory
end

# Lorenz system
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Noisy Lorenz
function noisy_lorenz!(du, u, p, t)
    σ, ρ, β, noise_std = p
    du[1] = σ * (u[2] - u[1]) + noise_std * randn()
    du[2] = u[1] * (ρ - u[3]) - u[2] + noise_std * randn()
    du[3] = u[1] * u[2] - β * u[3] + noise_std * randn()
end

# Rossler system
function rossler!(du, u, p, t)
    a, b, c = p
    x, y, z = u
    du[1] = -y - z
    du[2] = x + a * y
    du[3] = b + z * (x - c)
end

# Logistic Map
function generate_logistic_map(r, Nf)
    trajectory = zeros(Nf)
    trajectory[1] = rand()
    for t in 1:Nf
        trajectory[1] = r * trajectory[1] * (1 - trajectory[1])
    end
    for t in 2:Nf
        trajectory[t] = r * trajectory[t-1] * (1 - trajectory[t-1])
    end
    return trajectory
end

# 3D Hyperchaotic Logistic Map
function generate_3d_hyperchaotic_logistic_map(p, Nf)
    r, s = p
    trajectory = zeros(Nf, 3)
    trajectory[1, :] = rand(3)
    for t in 2:Nf
        x, y, z = trajectory[t-1, :]
        trajectory[t, :] = [
            r * x * (1 - x) + s * y,
            r * y * (1 - y) + s * z,
            r * z * (1 - z) + s * x
        ]
    end
    return trajectory
end

# Analyze ODE-based system
function analyze_system(system, params, Nf)

    Δt = params[2]
    tspan = (0.0, round(Int, 2 * Nf * Δt))
    u0 = 2 * rand(3)
    prob = ODEProblem(system, u0, tspan, params[1])
    sol = solve(prob, Tsit5(), dt = Δt, saveat = Δt, reltol = 1e-9, abstol = 1e-9, maxiters = 1e7)
    return Matrix(sol[:, end - Nf + 1:end]')
end


# Define the SDE function with slow forcing α(t)
function tipping_sde(dx, x, p, t)
    σ, ε = p  # Parameters: noise level, forcing rate
    α = ε * t  # Slow drift forcing
    dx[1] = (x[1] - x[1]^3 + α)  # Modified drift term
end

# Analyze ODE-based system
function analyze_sde_system(system, params, Nf)

    function noise_sde(dx, x, p, t)
        σ, ε = p
        dx[1] = σ  # Diffusion term (stochastic noise)
    end
    
    Δt = params[2]
    tspan = (0.0, Nf * Δt)
    
    # Solve the SDE (using more stable SOSRI() solver)
    prob = SDEProblem(system, noise_sde, [-0.9], tspan, params[1])
    sol = solve(prob, SOSRI(), dt=0.01, saveat=Δt)  # Ensure saving at Δt intervals

    # Extract time and state values
    x = [u[1] for u in sol.u[1:Nf]]
    
    return x
end

end # module
