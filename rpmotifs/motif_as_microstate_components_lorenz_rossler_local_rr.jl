#
#
using RecurrenceAnalysis
using DifferentialEquations
using Plots
using Statistics
using ProgressBars
using LinearAlgebra
using LaTeXStrings

function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

function rossler!(du, u, p, t)
    a, b, c = p
    x, y, z = u
    du[1] = -y - z
    du[2] = x + a * y
    du[3] = b + z * (x - c)
end

function calcula_P_micro_S(X, rr, L)
    N = size(X)[1]

    hist = fill(1.0e-30, (N-L, 2^(L*L)))

    RP = RecurrenceMatrix(StateSpaceSet(X), LocalRecurrenceRate(rr); metric = Euclidean(), parallel=true)
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

    rr = 0.5

    PDRP_RR = 0.2

    Nf = 10000

    Δt = 0.1

    L = 2

    pasta = "/home/gabrielm/figuras/pik/motif_as_microstate_components_lorenz_rossler_local_rr/rr$(rr)_Nf$(Nf)_dt$(Δt)_L$(L)"
    mkpath(pasta)
    

    function motifs_probabilities_and_entropy(system, p)
        ds = ODEProblem(system, 2*rand(3), (0.0, round(Int, 2*Nf*Δt)), p)
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
        end

        #comps = sortperm(var_comps)

        
        png(plot(serie[:, 1], serie[:, 2], serie[:, 3], title="Entropy",
        marker=:o, zcolor=entrp, lc=:gray, color=:jet1, la=0.6, lw=0.8, 
        xlabel="x", ylabel="y", zlabel="z", label="", ms=1.4, markerstrokewidth=0.2, 
        size=(500,400), dpi=200, frame_style=:box), pasta*"/$(string(system))_atractor_ps")
        
        png(plot(serie[:, 1], serie[:, 2], serie[:, 3], title="Entropy",
        marker=:o, zcolor=entrpx, lc=:gray, color=:jet1, la=0.6, lw=0.8, 
        xlabel="x", ylabel="y", zlabel="z", label="", ms=1.4, markerstrokewidth=0.2, 
        size=(500,400), dpi=200, frame_style=:box), pasta*"/$(string(system))_atractor_psx")
        
        png(plot(serie[:, 1], serie[:, 2], serie[:, 3], title="Entropy",
        marker=:o, zcolor=entrpz, lc=:gray, color=:jet1, la=0.6, lw=0.8, 
        xlabel="x", ylabel="y", zlabel="z", label="", ms=1.4, markerstrokewidth=0.2, 
        size=(500,400), dpi=200, frame_style=:box), pasta*"/$(string(system))_atractor_psz")


        for motif_idx=1:length(ps[1, :])
            png(plot(serie[:, 1], serie[:, 2], serie[:, 3], zcolor=ps[:, motif_idx], title="Motif #$(motif_idx) Probability",
            marker=:o, lc=:gray, color=:jet1, la=0.6, lw=0.8, 
            xlabel="x", ylabel="y", zlabel="z", label="", ms=1.4, markerstrokewidth=0.2, 
            size=(500,400), dpi=200, frame_style=:box), pasta*"/$(string(system))_atractor_motif$(motif_idx)")

            png(plot(serie[:, 1], serie[:, 2], serie[:, 3], zcolor=psx[:, motif_idx], title="Motif #$(motif_idx) Probability",
            marker=:o, lc=:gray, color=:jet1, la=0.6, lw=0.8, 
            xlabel="x", ylabel="y", zlabel="z", label="", ms=1.4, markerstrokewidth=0.2, 
            size=(500,400), dpi=200, frame_style=:box), pasta*"/$(string(system))_atractor_motif$(motif_idx)_comp_X")

            png(plot(serie[:, 1], serie[:, 2], serie[:, 3], zcolor=psz[:, motif_idx], title="Motif #$(motif_idx) Probability",
            marker=:o, lc=:gray, color=:jet1, la=0.6, lw=0.8, 
            xlabel="x", ylabel="y", zlabel="z", label="", ms=1.4, markerstrokewidth=0.2, 
            size=(500,400), dpi=200, frame_style=:box), pasta*"/$(string(system))_atractor_motif$(motif_idx)_comp_Z")
        end
    end
    
    motifs_probabilities_and_entropy(lorenz!, [10.0, 28.0, 8/3])
    motifs_probabilities_and_entropy(rossler!, [0.2, 0.2, 5.7])
   
    return
end
main()