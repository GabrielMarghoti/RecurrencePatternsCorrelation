#
#
using RecurrenceAnalysis
using DifferentialEquations
using Plots
using Statistics
using ProgressBars
using LinearAlgebra
using LaTeXStrings
##
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end
function calcula_P_micro_S(X, rr, L)
    N = size(X)[1]

    hist = fill(1.0e-30, (N-L, 2^(L*L)))

    RP = RecurrenceMatrix(StateSpaceSet(X), GlobalRecurrenceRate(rr); metric = Euclidean(), parallel=true)
    MM = 0
    for id_x=1:(N-L)
        M=0
        for id_y=1:(N-L)
            id_micro_estado = 0
            for ly=1:(L)
                for lx=1:(L)
                    if RP[id_x+lx-1, id_y+ly-1] == 1
                        id_micro_estado += 2^((ly-1)*L+lx-1)
                    end
                end
            end
            hist[id_x, Int(1+id_micro_estado)] += 1
            M+=1
        end
        MM = M
    end
    P_A = hist/MM
    return P_A
end


function main()
    sigma = 10
    beta  = 8/3
    p = 28

    rr = 0.2

    PDRP_RR = 0.1

    Nf = 4000

    Δt = 0.1

    L = 2

    pasta = "/home/gabrielm/dados/tsukuba/instant_entropy/Nf$(Nf)_L$(L)"
    mkpath(pasta)
 
    σ = 10.0
    ρ = 28.0
    β = 8/3
    
    ds = ODEProblem(lorenz!, 20*randn(3), (0.0, round(Int, 2*Nf*Δt)), [σ, ρ, β])
    tr = solve(ds, Tsit5(), dt=Δt, saveat=Δt, reltol=1e-9, abstol=1e-9, maxiters = 1e7)
    

    ps =  calcula_P_micro_S(tr.u[end-Nf:end], rr, L)
    psx =  calcula_P_micro_S(tr[1, end-Nf:end], rr, L)
    psz =  calcula_P_micro_S(tr[3, end-Nf:end], rr, L)

    serie = tr[:, end-Nf:end-L]'
   # display(serie)

    var_comps = zeros(2^(L*L))
    #for comp_idx=1:(2^(L*L))
    #    var_comps[comp_idx] = std(ps[:, comp_idx])
    #end
    entrp = zeros(length(serie[:,1]))
    entrpx = zeros(length(serie[:,1]))
    entrpz = zeros(length(serie[:,1]))
    distances = []
    distances_x = []
    distances_z = []
    entropies = []
    entropies_x = []
    entropies_z = []
    for i=1:length(serie[:, 1])
        entrp[i] = -sum(ps[i, :].*log10.(ps[i,:]))
        entrpx[i] = -sum(psx[i, :].*log10.(psx[i,:]))
        entrpz[i] = -sum(psz[i, :].*log10.(psz[i,:]))
        for j=1:i
            push!(distances, sqrt(sum((serie[i,:].-serie[j,:]).^2)))
            push!(entropies, sqrt(sum((ps[i, :] .- ps[j, :]).^2)))

            push!(distances_x, sqrt(sum((serie[i,1].-serie[j,1]).^2)))
            push!(entropies_x, sqrt(sum((psx[i, :] .- psx[j, :]).^2)))
            push!(distances_z, sqrt(sum((serie[i,3].-serie[j,3]).^2)))
            push!(entropies_z, sqrt(sum((psz[i, :] .- psz[j, :]).^2)))
        end
    end

    #comps = sortperm(var_comps)

    png(plot(serie[:, 1], serie[:, 2], serie[:, 3], marker=:o, zcolor=entrp, lc=:gray, color=:jet1, la=0.6, lw=0.8, label="", ms=1.4, markerstrokewidth=0.2, size=(500,400), dpi=200, frame_style=:box), pasta*"/atractor_ps")
   
    png(plot(serie[:, 1], serie[:, 2], serie[:, 3], marker=:o, zcolor=entrpx, lc=:gray, color=:jet1, la=0.6, lw=0.8, label="", ms=1.4, markerstrokewidth=0.2, size=(500,400), dpi=200, frame_style=:box), pasta*"/atractor_psx")
    png(plot(serie[:, 1], serie[:, 2], serie[:, 3], marker=:o, zcolor=entrpz, lc=:gray, color=:jet1, la=0.6, lw=0.8, label="", ms=1.4, markerstrokewidth=0.2, size=(500,400), dpi=200, frame_style=:box), pasta*"/atractor_psz")

    png(scatter(distances, entropies, mc=:black, ms=0.4, ma=0.4, markerstrokewidth=0.01,  label="", xlabel="ΔX", ylabel="ΔP", size=(500,400), grid=false, dpi=200, frame_style=:box), pasta*"/SxD")

    png(scatter(distances_x, entropies_x, mc=:blue, ms=0.4, ma=0.4, markerstrokewidth=0.01,  label="", xlabel="ΔX", ylabel="ΔP", size=(500,400), grid=false, dpi=200, frame_style=:box), pasta*"/SxD_x")

    png(scatter(distances_z, entropies_z, mc=:red, ms=0.4, ma=0.4, markerstrokewidth=0.01,  label="", xlabel="ΔX", ylabel="ΔP", size=(500,400), grid=false, dpi=200, frame_style=:box), pasta*"/SxD_z")

###################

    png(scatter(distances, entropies, mc=:black, xscale=:log10, yscale=:log10, xlims=(1.0e-1, 6.0e1), ylims=(1.0e-3, 2.0e0), ms=0.4, ma=0.4, markerstrokewidth=0.01,  label="", xlabel="ΔX", ylabel="ΔP", size=(500,400), grid=false, dpi=200, frame_style=:box), pasta*"/log10_SxD")

    png(scatter(distances_x, entropies_x, mc=:blue, xscale=:log10, yscale=:log10, xlims=(1.0e-1, 6.0e1), ylims=(1.0e-3, 2.0e0), ms=0.4, ma=0.4, markerstrokewidth=0.01,  label="", xlabel="ΔX", ylabel="ΔP", size=(500,400), grid=false, dpi=200, frame_style=:box), pasta*"/log10_SxD_x")

    png(scatter(distances_z, entropies_z, mc=:red, xscale=:log10, yscale=:log10, xlims=(1.0e-1, 6.0e1), ylims=(1.0e-3, 2.0e0), ms=0.4, ma=0.4, markerstrokewidth=0.01,  label="", xlabel="ΔX", ylabel="ΔP", size=(500,400), grid=false, dpi=200, frame_style=:box), pasta*"/log10_SxD_z")
 
    return
end
main()