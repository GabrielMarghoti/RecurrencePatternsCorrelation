# BouncingParticles.jl

using Plots
using LinearAlgebra

gr() 
default(fontfamily="Computer Modern", linewidth=2, label=nothing, grid=false, framestyle=:box)

# --- Define the properties of a particle ---
mutable struct Particle
    position::Vector{Float64}
    velocity::Vector{Float64}
    radius::Float64
    mass::Float64
    color::Symbol
end

# --- Simulation Parameters ---
const N_PARTICLES = 100
const BOX_WIDTH = 100.0
const BOX_HEIGHT = 100.0
const RADIUS = 1.5
const MASS = 1.0
const MAX_INITIAL_VEL = 50.0

const NUM_STEPS = 5000
const DT = 0.02 # Time step

"""
Initializes N particles inside the box with random positions and velocities.
Ensures no initial overlap.
"""
function initialize_particles(N::Int)
    particles = Particle[]
    for i in 1:N
        # Make sure particles don't start overlapping with walls or each other
        while true
            pos = [
                RADIUS + rand() * (BOX_WIDTH - 2 * RADIUS),
                RADIUS + rand() * (BOX_HEIGHT - 2 * RADIUS)
            ]
            
            # Check for overlap with existing particles
            has_overlap = false
            for p_existing in particles
                if norm(p_existing.position - pos) < 2 * RADIUS
                    has_overlap = true
                    break
                end
            end

            if !has_overlap
                vel = (rand(2) .- 0.5) .* 2 .* MAX_INITIAL_VEL
                color = (i == 1) ? :red : :blue # Make the first particle red
                push!(particles, Particle(pos, vel, RADIUS, MASS, color))
                break
            end
        end
    end
    return particles
end

"""
The main simulation function.
"""
function simulate()
    particles = initialize_particles(N_PARTICLES)
    
    # We will track the position of the first particle
    particle_history = NTuple{3, Float64}[]
    
    # Setup for animation
    anim = @animate for step in 1:NUM_STEPS
        # 1. Handle Wall Collisions
        for p in particles
            # Left/Right walls
            if p.position[1] <= p.radius && p.velocity[1] < 0
                p.velocity[1] *= -1
            elseif p.position[1] >= BOX_WIDTH - p.radius && p.velocity[1] > 0
                p.velocity[1] *= -1
            end
            # Top/Bottom walls
            if p.position[2] <= p.radius && p.velocity[2] < 0
                p.velocity[2] *= -1
            elseif p.position[2] >= BOX_HEIGHT - p.radius && p.velocity[2] > 0
                p.velocity[2] *= -1
            end
        end

        # 2. Handle Particle-Particle Collisions (Elastic Collision)
        for i in 1:N_PARTICLES
            for j in (i+1):N_PARTICLES
                p1 = particles[i]
                p2 = particles[j]

                dist_vec = p1.position - p2.position
                dist = norm(dist_vec)

                if dist < p1.radius + p2.radius
                    # --- 2D Elastic Collision Physics ---
                    # Normal vector
                    n = dist_vec / dist
                    # Tangent vector
                    t = [-n[2], n[1]]

                    # Project velocities onto normal and tangent vectors
                    v1n = dot(p1.velocity, n)
                    v1t = dot(p1.velocity, t)
                    v2n = dot(p2.velocity, n)
                    v2t = dot(p2.velocity, t)
                    
                    # In 1D, for equal masses, the normal velocities simply swap
                    v1n_new = v2n
                    v2n_new = v1n
                    
                    # Convert scalar projections back to vectors and add them up
                    p1.velocity = v1n_new * n + v1t * t
                    p2.velocity = v2n_new * n + v2t * t

                    # --- Prevent sticking by separating overlapping particles ---
                    overlap = (p1.radius + p2.radius - dist) / 2
                    p1.position .+= overlap * n
                    p2.position .-= overlap * n
                end
            end
        end

        # 3. Update positions
        for p in particles
            p.position .+= p.velocity .* DT
        end
        
        # 4. Record the history of the first particle
        tracked_particle = particles[1]
        push!(particle_history, (step * DT, tracked_particle.position[1], tracked_particle.position[2]))

        # --- Draw the current frame for the animation ---
        scatter(
            [p.position[1] for p in particles], 
            [p.position[2] for p in particles],
            markercolor = [p.color for p in particles],
            markersize = 2 * RADIUS, # Scale marker size correctly
            xlims = (0, BOX_WIDTH),
            ylims = (0, BOX_HEIGHT),
            aspect_ratio = :equal,
            label = "",
            title = "Particle Simulation (Time: $(round(step*DT, digits=2))s)",
            framestyle = :box,
            legend = false
        )
    end
    
    # Save the animation
    gif(anim, "figures/particle_simulation.gif", fps = 30)
    println("Animation saved to particle_simulation.gif")
    
    return particle_history
end

"""
Plots the time series of the tracked particle's position.
"""
function plot_time_series(history)
    times = [h[1] for h in history]
    x_coords = [h[2] for h in history]
    y_coords = [h[3] for h in history]

    # Create the x-position plot
    p1 = plot(
        times, x_coords, 
        label = "x-position", 
        xlabel = "Time (s)", 
        ylabel = "x-coordinate",
        title = "Position of Tracked Particle (Red)",
        legend = :topright,
        color = :seagreen
    )

    # Create the y-position plot
    p2 = plot(
        times, y_coords, 
        label = "y-position", 
        xlabel = "Time (s)", 
        ylabel = "y-coordinate",
        legend = :topright,
        color = :purple
    )

    # Combine them into a single figure with two subplots
    final_plot = plot(p1, p2, layout = (2, 1), size=(800, 600), framestyle = :box, grid=false)
    
    # Save the plot
    savefig(final_plot, "figures/particle_time_series.png")
    println("Time series plot saved to particle_time_series.png")
end

# --- Run the full program ---
function main()
    println("Starting particle simulation...")
    history = simulate()
    println("Simulation finished. Plotting time series...")
    plot_time_series(history)
    println("Done.")
end

# Execute the main function
main()