# nonlinear_noise_rpc.jl

using RecurrenceAnalysis
using Distributions
using Plots
using StatsBase # For Pearson correlation
using LaTeXStrings

# Use a professional-looking plotting theme
gr()
default(fontfamily="Computer Modern",
        linewidth=1.5,
        label=nothing,
        grid=false,
        framestyle=:box,
        titlefontsize=12,
        guidefontsize=11,
        tickfontsize=10)

# --- Define the RPC/Moran's I function ---
# Copied from your likely implementation for self-containment.
# This assumes `local_morans_I` is what you called `lRPC` or `IRPC` in the paper.
# It calculates the local Moran's I for each time point i.
function local_morans_I(RP::AbstractMatrix{Bool}; weight_function, Δi_range, Δj_range)
    N = size(RP, 1)
    lRPC = zeros(Float64, N)
    
    # Pre-calculate non-zero weights to speed up
    weights = []
    for Δi in Δi_range, Δj in Δj_range
        if weight_function(Δi, Δj) != 0
            push!(weights, (Δi, Δj, weight_function(Δi, Δj)))
        end
    end
    
    if isempty(weights)
        return lRPC
    end

    for i in 1:N
        # Local recurrence rate for point i
        rri = sum(RP[i, :]) / N
        
        # Avoid division by zero for points with no recurrences or full recurrences
        if rri == 0.0 || rri == 1.0
            lRPC[i] = 0.0
            continue
        end
        
        numerator = 0.0
        weight_sum = 0.0
        
        # Iterate over pre-calculated non-zero weights
        for (Δi, Δj, w) in weights
            i_prime = i + Δi
            
            # Check bounds for i_prime
            if 1 <= i_prime <= N
                for j in 1:N
                    j_prime = j + Δj
                    
                    # Check bounds for j_prime and exclude self-comparisons
                    if 1 <= j_prime <= N && i != j && i_prime != j_prime
                        # Moran's I numerator term
                        term = w * (RP[i, j] - rri) * (RP[i_prime, j_prime] - rri)
                        numerator += term
                        weight_sum += w
                    end
                end
            end
        end
        
        if weight_sum > 0
            # Denominator is local variance * sum of weights
            denominator = rri * (1 - rri) * weight_sum
            lRPC[i] = numerator / denominator
        end
    end
    return lRPC
end

# --- Data Generation ---
"""
Generates a 2D time series with nonlinearly correlated noise.
x(t) = e_x(t)
y(t) = a * x(t)^2 + b * e_y(t)
"""
function simulate_coupled_maps(N::Int; r::Float64=4.0, c::Float64=0.1, transient::Int=500)
    total_len = N + transient
    x = zeros(total_len)
    y = zeros(total_len)
    
    # Start with different initial conditions
    x[1] = rand()
    y[1] = rand()
    
    for i in 1:(total_len - 1)
        x_next = r * x[i] * (1 - x[i]) + c * (y[i] - x[i])
        y_next = r * y[i] * (1 - y[i]) + c * (x[i] - y[i])
        
        # Ensure values stay within [0, 1] for stability at high coupling
        x[i+1] = mod(x_next, 1.0)
        y[i+1] = mod(y_next, 1.0)
    end
    
    # Return the time series after the transient has passed
    ts_x = x[(transient+1):end]
    ts_y = y[(transient+1):end]
    return hcat(ts_x, ts_y)
end

"""
Generates a 2D time series with uncorrelated noise.
"""
function generate_uncorrelated_noise(N::Int)
    dist = Normal(0, 1)
    x = rand(dist, N)
    y = rand(dist, N)
    
    # Normalize
    x = (x .- mean(x)) ./ std(x)
    y = (y .- mean(y)) ./ std(y)
    
    return hcat(x, y)
end


# --- Main Analysis Function ---
function main()
    # Parameters
    N = 10000          # Length of the time series
    rr = 0.05         # Target recurrence rate to determine epsilon
    
    # --- 1. Generate Data ---
    println("Generating time series...")
    ts_nonlinear = simulate_coupled_maps(N)

    # --- 3. Construct Recurrence Plots ---
    println("\nConstructing recurrence plots...")
    
    
    RP_nonlinear =  RecurrenceMatrix(StateSpaceSet(ts_nonlinear) , GlobalRecurrenceRate(rr); metric = Euclidean())

    # --- 4. Calculate RPC for a Diagonal Motif ---
    println("\nCalculating RPC for diagonal motif (Δi=1, Δj=1)...")
    motif(Δi, Δj) = (Δi == 1 && Δj == 1) ? 1.0 : 0.0
    Δi_rng, Δj_rng = 1:1, 1:1

    # We are interested in the global average of the local RPC values
    rpc_nonlinear_local = local_morans_I(RP_nonlinear, weight_function=motif, Δi_range=Δi_rng, Δj_range=Δj_rng)

    # --- 5. Visualize Results ---
    println("\nGenerating plot...")
    
    circ = Shape(Plots.partialcircle(0, 2π))
    plote =  scatter(ts_nonlinear[5:end-5, 1], ts_nonlinear[5:end-5, 2], 
                marker_z=rpc_nonlinear_local[5:end-5], color=:vik, label="", ms=1.5, alpha=0.9,
                strokewidth=0, markerstrokealpha=0, markerstrokecolor=nothing, markershape=circ,
                xlabel=L"x", ylabel=L"y", widen=false,
                colorbar_title="lRPC",
                clims=(-1.0, 1.0),
                size=(500, 400), dpi=200, frame_style=:box, grid=false)
                   
    
    savefig(plote, "nonlinear_noise_local_rpc.pdf")
    savefig(plote, "nonlinear_noise_local_rpc.png")
    
    println("\nAnalysis complete. Plot saved as 'nonlinear_noise_rpc_comparison.pdf'.")
end

# Run the main function
main()