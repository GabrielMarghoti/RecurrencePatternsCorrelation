
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Julia Script for Analyzing Coherence Resonance (CR)              #
#  Based on the Pikovsky & Kurths 1997 PRL paper                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# -------------------------------------
# 1. SETUP: Load necessary packages
# -------------------------------------
using DifferentialEquations
using Plots
using DSP           # For Power Spectral Density (periodogram)
using Statistics    # For mean, std
using RecurrenceAnalysis # for Recurrence Plots
using DelayEmbeddings
using Dates

include("../PlotRPMotifs.jl")
using ..PlotRPMotifs

include("../RPMotifs.jl")
using ..RPMotifs

# Ensure high-quality plots
gr() 
default(fontfamily="Computer Modern", linewidth=2, label=nothing, grid=false, framestyle=:box)


# Output directories

data_path    = "/home/gabrielm/projects/RPMotifs/data/coherence_resonance/"
figures_path = "figures/coherence_resonance$(today())/"

mkpath(data_path)
mkpath(figures_path)

# ---------------------------------------------------------------------
# 2. MODEL DEFINITION: FitzHugh-Nagumo from Pikovsky & Kurths (1997)
# ---------------------------------------------------------------------
# The model is taken directly from the paper:
#
#   dx/dt = x - x^3/3 - y
#   dy/dt = ε * (x + a) + Dξ(t)
#
# Here, 'x' is the fast variable (voltage) and 'y' is the slow
# recovery variable.
#
# **Crucially, the noise Dξ(t) is applied to the SLOW variable 'y'.**

# --- Deterministic Part (Drift) ---
# du is the output [dx/dt, dy/dt]
# u is the state vector [x, y]
# p is the parameter vector (ε, a, D)
# t is time
function fhn_drift_paper!(du, u, p, t)
    x, y = u
    ε, a, _ = p # Unpack parameters, ignore noise amplitude D here
    
    du[1] = (x - (x^3 / 3) - y)/ε
    du[2] = (x + a)
end

# --- Stochastic Part (Diffusion) ---
# This function defines the noise term. It receives the same parameter
# object 'p' as the drift function.
function fhn_diffusion_paper!(du, u, p, t)
    _, _, D = p # Unpack only the noise amplitude D
    
    du[1] = 0.0 # No noise on x
    du[2] = D   # Noise with amplitude D is on y
end

# ---------------------------------------------------------------------
# 3. ANALYSIS FUNCTION: Quantifying Coherence (Q-Factor)
# ---------------------------------------------------------------------
# We quantify the regularity of the output spikes using the Q-factor 
# from the Power Spectral Density (PSD) of the voltage time series (x).
# A higher Q-factor means more regular, periodic firing.
# Note: The paper uses correlation time (τc) and relative pulse
# duration fluctuations (Rp), which are time-domain measures. The Q-factor
# is a frequency-domain measure of the same underlying phenomenon.

function calculate_q_factor(timeseries, dt)
    fs = 1 / dt
    pgram = welch_pgram(timeseries, fs=fs)
    freqs, power = pgram.freq, pgram.power

    if length(freqs) <= 2; return 0.0; end
    
    # Ignore DC component
    search_power = power[2:end]
    P_peak, peak_idx_local = findmax(search_power)
    
    if P_peak < 1e-3; return 0.0; end
    
    peak_idx_global = peak_idx_local + 1
    f_peak = freqs[peak_idx_global]
    
    half_max = P_peak / 2.0
    
    # Find Full-Width at Half-Maximum (FWHM)
    idx_right_local = findfirst(p -> p < half_max, @view search_power[peak_idx_local:end])
    idx_left_local_rev = findfirst(p -> p < half_max, @view search_power[peak_idx_local:-1:1])

    if isnothing(idx_right_local) || isnothing(idx_left_local_rev); return 0.0; end
    
    f_right = freqs[peak_idx_local + idx_right_local]
    f_left = freqs[peak_idx_local - idx_left_local_rev + 2]
    
    fwhm = f_right - f_left
    return fwhm > 0 ? (f_peak / fwhm) : 0.0
end

"""
Calculates the correlation time (τc) of a time series.
Resonance is indicated by a PEAK in τc.
"""
function calculate_correlation_time(timeseries, dt)
    # The ACF is calculated on the fluctuating part of the signal
    signal = timeseries .- mean(timeseries)
    if std(signal) < 1e-4; return 0.0; end

    # Calculate normalized autocorrelation function (ACF)
    unnormalized_acf = autocor(signal, 0:length(signal)-1)
    acf = unnormalized_acf ./ unnormalized_acf[1] # Normalize by variance

    # Integrate the square of the ACF (using simple summation)
    # We integrate until the first zero-crossing for stability
    
    
    τc = sum(acf.^2) * dt
    return τc
end


# ---------------------------------------------------------------------
# 4. SIMULATION AND ANALYSIS LOOP
# ---------------------------------------------------------------------

println("Starting Coherence Resonance simulation based on Pikovsky & Kurths (1997)...")

# --- Model Parameters from the paper ---
ε = 0.01
a = 1.002

# --- Simulation Parameters ---
u0 = [-1.0, -1.0] # Initial conditions near the stable fixed point
t_end = 1000.0   # Long simulation for good spectral resolution
dt_save = 0.1     # Save data at these time intervals
tspan = (0.0, t_end)

# --- Noise Levels (D) to Scan ---
# The paper finds resonance around D ≈ 0.06. We scan around this.
noise_intensities = 10 .^ range(-2.6, -0.6, length=50)
correlation_times = Float64[]
example_results = Dict() 

for (i, D) in enumerate(noise_intensities)
    print("Simulating for noise D = $(round(D, digits=5)) [$(i)/$(length(noise_intensities))]...")
    
    # Combine all parameters into a single tuple
    p = (ε, a, D)
    
    # Define and solve the SDE problem
    sde_prob = SDEProblem(fhn_drift_paper!, fhn_diffusion_paper!, u0, tspan, p)
    sol = solve(sde_prob, EM(), dt=0.001, saveat=dt_save)
    
    # Analyze the 'x' variable time series (voltage)
    x_series = [u[1] for u in sol.u]
    
    τc = calculate_correlation_time(x_series, dt_save)
    push!(correlation_times, τc)
    println(" τc = $(round(τc, digits=2))")

    # --- Store a few examples for later visualization ---
    if i == 4; example_results["low"] = (D=D, sol=sol, τc=τc); end
    if i == length(noise_intensities) - 2; example_results["high"] = (D=D, sol=sol, τc=τc); end
    if i > 1 && τc > maximum(@view correlation_times[1:end-1]); example_results["optimal"] = (D=D, sol=sol, τc=τc); end
end

println("Simulation complete.")

# ---------------------------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------------------------

println("Generating plots...")

# --- PLOT 1: The Coherence Resonance Curve ---
cr_plot = plot(
    noise_intensities,
    correlation_times,
    xaxis=:log,
    xlabel="Noise Intensity (D)",
    ylabel="Coherence (Q-factor)",
    title="Coherence Resonance (Pikovsky & Kurths Model)",
    label="Q-factor",
    marker=:circle,
    legend=:topleft
)

# Highlight the points for which we have example plots
scatter!(cr_plot, [example_results["low"].D], [example_results["low"].τc], label="Low Noise", color=:blue, markersize=8)
if haskey(example_results, "optimal"); scatter!(cr_plot, [example_results["optimal"].D], [example_results["optimal"].τc], label="Optimal Noise", color=:green, markersize=8); end
scatter!(cr_plot, [example_results["high"].D], [example_results["high"].τc], label="High Noise", color=:red, markersize=8)

# --- PLOT 2: Example Time Series and Power Spectra ---

function create_example_plot(data, color, title_str)
    sol, D, τc = data.sol, data.D, data.τc
    t = sol.t
    x_series = reduce(hcat, sol.u)'#[u[1] for u in sol.u]  # (time_length, 2) array
   
    
    if size(x_series, 2) == 1
        x_series = pecuzal_embedding(x_series; max_cycles=4)[1]
    end

    RP = RecurrenceMatrix(StateSpaceSet(x_series),  GlobalRecurrenceRate(0.02))

    time_shifts=0:1:200

    RPC = zeros(length(time_shifts))
    
    di = 0
    for (j,dj) in enumerate(time_shifts)
        

        RPC[j] = morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == dj ? 1 : 0), Δi_range = di:di, Δj_range = dj:dj)
    end

    shift_opt =time_shifts[findfirst(x -> x== maximum(RPC[6:end]), RPC[6:end])] + 5
    
    noise_shift= 6
    upo_shift= 32
    
    lRPC_noise = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == noise_shift ? 1 : 0), Δi_range = di:di, Δj_range = noise_shift:noise_shift)
    
      
    lRPC_upo = local_morans_I(RP; weight_function = (Δi, Δj) -> (Δi == di && Δj == upo_shift ? 1 : 0), Δi_range = di:di, Δj_range = upo_shift:upo_shift)
    
    circ = Shape(Plots.partialcircle(0, 2π))   
    # Top plot: Time series
    ts_plot = plot(t[end-699:end], x_series[end-699:end, 1], xlabel="t", ylabel="x",
                    label="",
                   title="$title_str (D=$(round(D, digits=4)), τc=$(round(τc, digits=2)))", color=color, size=(500,250))
    
    lRPC_plot = plot(t[end-699:end], lRPC_noise[end-699:end], xlabel="t", ylabel="lRPC", label="Noise period", color=:black)
        
    plot!(t[end-699:end], lRPC_upo[end-699:end], ls=:dash, xlabel="t", ylabel="lRPC", label="UPO period", color=:orange)
    

    RPC_plot = plot(time_shifts.*dt_save, RPC, xlabel="Dj * dt", ylabel="RPC", label="", color=color)
                vline!([noise_shift * dt_save], label="Noise Shift", ls=:dash, color=:black)            
                vline!([upo_shift * dt_save], label="UPO Shift", ls=:dash, color=:orange)      
    # Bottom plot: Recurrence Plot
    # Use the last 400 points for a clear plot (400x400)
    series_for_rp = x_series[end-699:end, :]

    if size(series_for_rp, 2) == 1
        #println("   Performing Pecuzal embedding for 1D time series")
        series_for_rp = pecuzal_embedding(series_for_rp; max_cycles=4)[1]
    end
    
    # Create the recurrence matrix and plot it
    R = RecurrenceMatrix(StateSpaceSet(series_for_rp),  GlobalRecurrenceRate(0.02))
    rp_plot = spy(R, title="Recurrence Plot", markersize=1.5, color=color, legend=false)




    return plot(ts_plot, lRPC_plot, RPC_plot, rp_plot, layout=grid(4, 1, heights=[0.2, 0.2, 0.2, 0.4]))
end

# Generate the three example plots
low_noise_plot = create_example_plot(example_results["low"], :blue, "Low Noise")
opt_noise_plot = haskey(example_results, "optimal") ? create_example_plot(example_results["optimal"], :green, "Optimal Noise") : plot(title="Optimal not captured", framestyle=:none)
high_noise_plot = create_example_plot(example_results["high"], :red, "High Noise")

# Combine all plots into a final layout
final_plot = plot(
    cr_plot, low_noise_plot, opt_noise_plot, high_noise_plot,
    layout = @layout([A{0.20h}; [B C D]]),
    size=(2200, 1400), 
    left_margin=5Plots.mm,
    plot_title = "Coherence Resonance Analysis with Recurrence Plots"
)

display(final_plot)
savefig(final_plot, joinpath(figures_path,"coherence_resonance_paper_model.png"))
println("Plot saved to coherence_resonance_paper_model.png")