using Plots
using RecurrenceAnalysis
using DynamicalSystemsBase # For StateSpaceSet
using DelayEmbeddings      # For embedding the AR series
using Dates
using LaTeXStrings

include("../RPMotifs.jl")
using ..RPMotifs
# Assuming you have a way to generate the Lorenz data.
# And here is a helper for the AR series.
function simple_ar_series(N, alpha=0.99)
    ts = zeros(N)
    ts[1] = randn()
    for i in 2:N
        ts[i] = alpha * ts[i-1] + randn()
    end
    return ts
end

function main()
    # Parameters
    rr = 0.05
    Nf = 501
    dt = 0.1
    
    figures_path = "figures/Comparative_RPs_With_TS_$(today())/"
    mkpath(figures_path)
    
    # --- Generate Data for Both Systems ---
    
    # 1. Lorenz System
    println("Generating Lorenz system data...")
    lorenz_traj =  generate_time_series("Lorenz",  [[10.0, 28.0, 8 / 3], dt], Nf, [1,2,3]) # Generate time series
    lorenz_ts_z = lorenz_traj[:, 1] # Use z-component for the time series plot
    println("Calculating Lorenz recurrence matrix...")
    lorenz_rp = Matrix(RecurrenceMatrix(StateSpaceSet(lorenz_traj), GlobalRecurrenceRate(rr)))


    # Define the layout: 2 rows, 2 columns.
    # Make the top row (time series) shorter than the bottom row (RPs).
    l = @layout [
        a{0.15h}  
        b             
    ]
    
    comparative_plot = plot(
        layout = l,
        size = (600, 800),
        dpi = 300,
        fontfamily = "Computer Modern",
        link = :x, # Link x-axes vertically between top and bottom plots
        framestyle = :box
    )
    
    # --- Populate the panels ---
    
    # Panel 1: Lorenz Time Series (Top-Left)
    plot!(comparative_plot, subplot = 1,
        1:Nf, lorenz_ts_z,
        title = "(a) Time series",
        ylabel = L"x_i",
        lw=2,
        
        xformatter =Returns(""), # Hide x-axis labels
        legend = false,
        titleposition = :left,
        color = :black,
        xlims = (1, Nf),
        grid = false
    )

    # Panel 3: Lorenz RP (Bottom-Left)
    heatmap!(comparative_plot, subplot = 2,
        lorenz_rp,
        xlabel = "Time i",
        ylabel = "Time j",
        aspect_ratio = 1,
        color = :binary,
        colorbar=false,
        legend = false,
        title = "(b) Recurrence Plot",
        titleposition = :left,
        widen=false,
        ylims = (1, Nf),
        xlims = (1, Nf),
        grid=false
    )
    annotate!(comparative_plot, subplot=2, 
        5, -50, # position in center
        text("(c) Weight matrix:", :left, 14, "Computer Modern", :black))
    
    output_filename = "diagram_rp"
    
    println("Saving figure to $(figures_path)...")
    savefig(comparative_plot, joinpath(figures_path, output_filename * ".png"))
    savefig(comparative_plot, joinpath(figures_path, output_filename * ".pdf"))

    savefig(comparative_plot, joinpath(figures_path, output_filename * ".svg"))
    println("Done.")
end


 main()