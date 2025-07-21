

# Recurrence Recurrence Patterns Correlation, Marghoti G., 2025 
# Paper figures reproduction script




# plot fig 2
$ julia 
using Plots
using LaTeXStrings
default(fontfamily="Computer Modern", linewidth=2, label=nothing, grid=false, framestyle=:box)
x = range(0,1,101)
plot(x, [((1 .- x).^2) ((x.^2)) (-x.*(1 .-x))],
           label = [L"(1 - rr)^2" L"rr^2" L"-rr(1 - rr)"],
           lc = [:red :red :blue],
           frame_style = :box,
           grid = false,
           xlabel = L"rr",
           ylabel = L"\left(\textbf{R}_{ij} - rr\right) \left(\textbf{R}_{i'j'} - rr\right)",
           size = (500, 250),
           dpi = 300,
           lw = [2 2 2],
           legend = :top,
           ls = [:solid :dot :solid],
           ylims=(-0.275,1.02))


# plot fig 3
julia scripts/compare_sys_colorbar.jl


# plot fig 4
julia scripts/RPC_compare_systems.jl 


# plot fig 5
julia scripts/Logistic_bifurcation_lRPC_basis_panels.jl 


# plot fig 6
julia scripts/standard_map_local_RPC_multi_ic_panels.jl 

# plot fig 7
julia scripts/Lorenz_RPC_var_dj_autocorr.jl 