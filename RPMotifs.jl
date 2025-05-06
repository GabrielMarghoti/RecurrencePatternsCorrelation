module RPMotifs 

using Statistics

include("./DynamicalSystemsToolkit.jl")
using .DynamicalSystemsToolkit

export morans_I, local_morans_I,
       generate_time_series

    # Global Moran's I 
    function morans_I(
        x::AbstractMatrix;
        weight_function = (Δi, Δj) -> (Δi == Δj ? 1 : 0),
        Δi_range = -1:1,
        Δj_range = -1:1
    )
        N_i, N_j = size(x)
        mean_x = mean(x)
    
        numerator = 0.0
        W = 0.0
    
        for i in 1:N_i 
            for j in 1:N_j
                if j==i
                    continue
                end
                xi = x[j, i] - mean_x
        
                for Δi in Δi_range, Δj in Δj_range
                    if Δi == 0 && Δj == 0
                        continue
                    end
        
                    w = weight_function(Δi, Δj)
                    if w == 0
                        continue
                    end
        
                    i′ = i - Δi
                    j′ = j - Δj
        
                    if 1 ≤ i′ ≤ N_i && 1 ≤ j′ ≤ N_j
                        xj = x[j′, i′] - mean_x
                        numerator += w * xi * xj
                        W += w
                    end
                end
            end
        end
    
        denominator = sum((x .- mean_x).^2)
    
        return ((N_i * N_j) / W) * (numerator / denominator)
    end
    

    # Local Moran's I, column-wise
    function local_morans_I(
        x::AbstractMatrix;
        weight_function = (Δi, Δj) -> (Δi == Δj ? 1 : 0),
        Δi_range = -1:1,
        Δj_range = -1:1
    )
        N_j, N_i = size(x)
        results = fill(NaN, N_i)
    
        for i in 1:N_i
            mean_x = mean(x[:, i])
            numerator = 0.0
            W = 0.0
    
            for j in 1:N_j
                if j==i
                    continue
                end
                xi = x[j, i] - mean_x
                for Δi in Δi_range, Δj in Δj_range
                    if Δi == 0 && Δj == 0
                        continue
                    end
        
                    w = weight_function(Δi, Δj)
                    if w == 0
                        continue
                    end
        
                    i′ = i - Δi
                    j′ = j - Δj
        
                    if 1 ≤ i′ ≤ N_i && 1 ≤ j′ ≤ N_j
                        xj = x[j′, i′] - mean_x
                        numerator += w * xi * xj
                        W += w
                    end
                end
            end
    
            if W > 0
                denominator = sum((x[:, i] .- mean_x).^2)
                results[i] = (N_j / W) * (numerator / denominator)
            end
        end
    
        return results
    end
    
    function generate_time_series(system_name, params, Nf, component)

        # Generate trajectory
        if system_name == "Randn"
            time_series = randn(Nf)
        
        elseif startswith(system_name, "AR 0.")
            time_series = generate_ar_trajectory(params, Nf)
        
        elseif occursin("3D AR", system_name)
            trajectory = generate_3d_ar_trajectory(params, Nf) # Generate system trajectory
            time_series = trajectory[:, component]  # Extract component
        
        elseif occursin("Logistic 3D", system_name)
            trajectory = generate_3d_hyperchaotic_logistic_map(params, Nf) # Generate system trajectory
            time_series = trajectory[:, component]  # Extract component
        
        elseif occursin("Logistic", system_name)
            trajectory = generate_logistic_map(params, Nf) # Generate system trajectory
            time_series = trajectory
        elseif occursin("Circle", system_name)
            trajectory  = [sin.(2π/3.3 * params * range(1, Nf)) cos.(2π/3.3 * params * range(1, Nf))]
            time_series = trajectory[:, component]
        
        elseif system_name == "GARCH"
            time_series = generate_garch(Nf, params...)

        elseif system_name == "AR(2)"
            time_series = generate_ar(Nf, params[1], params[2])

        elseif occursin("Tipping point sde", system_name)
            time_series = analyze_sde_system(tipping_sde, params, Nf)  

        elseif occursin("Rossler", system_name)
            trajectory = analyze_system(rossler!, params, Nf)  
            time_series = trajectory[:, component]

        elseif occursin("Lorenz", system_name)
            trajectory = analyze_system(lorenz!, params, Nf)   
            time_series = trajectory[:, component]
        end

        return time_series

    end






end # module