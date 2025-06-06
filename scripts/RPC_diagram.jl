
using RecurrenceAnalysis
using ProgressBars
using Dates
using Random, Distributions
using Plots, Measures
using DelayEmbeddings
using JLD2

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs

# Calculate correlation quantifier from recurrence plots
function correlation_quantifier(RP1, RP2)
    # Simple correlation measure between recurrence plots
    return cor(vec(RP1), vec(RP2))
end

# Time-delay embedding (Takens embedding)
function time_delay_embedding(s, d, τ)
    N = length(s)
    M = N - (d-1)*τ
    embedded = zeros(M, d)
    for i in 1:M
        for j in 1:d
            embedded[i,j] = s[i + (j-1)*τ]
        end
    end
    return embedded
end

# Main function
function main()
    # Parameters
    d = 2  # embedding dimension
    τ = 100  # time delay (in samples)
    ε = 0.2  # recurrence threshold
    t = 0:0.01:10π  # time vector
    
    data_path    = "data/diagram"
    figures_path = "figures/diagram_$(today())/ε$(ε)_d$(d)_τ$(τ)"

    mkpath(data_path)
    mkpath(figures_path)
    


    # Generate time series
    noise_level=0.2
    sx = sin.(t) .* (1 .+noise_level .*randn(length(t)))  # sin(t) with noise
    # Create time-delay embeddings using DynamicalSystems.jl

    embedded_s1 = embed(sx, d, τ)

    # Create cross-recurrence plot
    rp = RecurrenceMatrix(embedded_s1, ε; metric = Euclidean())
    
    ty, tx = size(rp)

    i = t[100]
    di = 45*(t[2]-t[1])

    j = t[200]
    dj = -22*(t[2]-t[1])

    p1 = plot(t[end-tx:end], sx[end-tx:end], ylabel="sin(t) + η", xlabel="Time",
                lc=:black,
                frame_style=:box, legend=false, grid=false)
                vline!([i], lc = :blue, linestyle=:solid, label="", lw=2)
                vline!([i+di], lc = :blue, linestyle=:dash, label="", lw=2)

                vline!([j], lc = :red, linestyle=:solid, label="", lw=2)
                vline!([j+dj], lc = :red, linestyle=:dash, label="", lw=2)
    p2 = heatmap(t[end-tx:end], t[end-ty:end], rp, title="Recurrence Plot", xlabel="Time", ylabel="Time", 
                color=:Greys, aspect_ratio=1, widen=false,
                xlims=(t[end-tx], t[end]), ylims=(t[end-ty], t[end]),
                frame_style=:box, colorbar=false,
    )
                vline!([i], lc = :blue, linestyle=:solid, label="", lw=3,la=8)
                vline!([i+di], lc = :blue, linestyle=:dash, label="", lw=3,la=8)

                hline!([j], lc = :red, linestyle=:solid, label="", lw=3,la=8)
                hline!([j+dj], lc = :red, linestyle=:dash, label="", lw=3,la=8)
    
    # Layout with time series on top, CRP at bottom
    l = @layout [
        a 
        b{0.7h}
    ]
    
   
    diagram_plot =  plot(p1, p2, 
        layout = l, 
        dpi = 300,
        margin = 5mm,
        link = :x,
        size = (500, 900))  # Adjusted for single column width
    
    savefig(diagram_plot, joinpath(figures_path, "diagram.png"))
    savefig(diagram_plot, joinpath(figures_path, "diagram.svg"))
end

# Run the analysis
main()