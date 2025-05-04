module Quantifiers

using StatsBase
using Distributions

export mutual_information, 
       morans_I


function morans_I(x::AbstractMatrix; weight_function = (Δi, Δj) -> (Δi == Δj ? 1 : 0))
    N_i, N_j = size(x)
    mean_x = 0#mean(x)

    numerator = 0.0
    W = 0.0

    for i in 1:N_i, j in 1:N_j
        for i′ in 1:N_i, j′ in 1:N_j
            if i == i′ && j == j′
                continue  # skip self-pairs
            end

            Δi = i - i′
            Δj = j - j′
            w = weight_function(Δi, Δj)

            xi = x[j, i] - mean_x
            xj = x[j′, i′] - mean_x

            numerator += w * xi * xj
            W += w
        end
    end

    denominator = sum((x .- mean_x).^2)

    return ((N_i * N_j) / W) * (numerator / denominator)
end
    


# Function to compute mutual information
function mutual_information(series, delay)
    n = length(series)
    shifted_series = circshift(series, -delay)
    joint_prob = fit(Histogram, (series, shifted_series), closed=:left)
    marginal_prob_x = fit(Histogram, series, closed=:left)
    marginal_prob_y = fit(Histogram, shifted_series, closed=:left)
    
    joint_prob = joint_prob.weights / n
    marginal_prob_x = marginal_prob_x.weights / n
    marginal_prob_y = marginal_prob_y.weights / n
    
    mi = 0.0
    for i in 1:length(marginal_prob_x)
        for j in 1:length(marginal_prob_y)
            if joint_prob[i, j] > 0
                mi += joint_prob[i, j] * log(joint_prob[i, j] / (marginal_prob_x[i] * marginal_prob_y[j]))
            end
        end
    end
    return mi
end


end # module