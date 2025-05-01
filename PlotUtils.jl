module PlotUtils

using Plots
using Measures

function plot_motifs_transition_joint_prob(probabilities, rr, LMAX; log_scale=true, figures_path=".")
    mkpath(figures_path)

    Xticks_nume = [-LMAX[1]:2:-2; 0 ; 2:2:LMAX[1]]
    Xticks_name = ["i".*string.(-LMAX[1]:2:-2); "i" ;"i+".*string.(2:2:LMAX[1])]
    Yticks_nume = [-LMAX[2]:2:-2; 0 ; 2:2:LMAX[2]]
    Yticks_name = ["j".*string.(-LMAX[2]:2:-2); "j" ;"j+".*string.(2:2:LMAX[2])]

    # Plot transition times matrix
    for motif_idx in 1:2^2
        Rij, R00 = divrem(motif_idx - 1, 2)
        
        probabilities_motif = R00==1 ? probabilities[:, :, motif_idx]/rr : probabilities[:, :, motif_idx]/(1-rr)
        if log_scale
            probabilities_motif = log.(probabilities_motif .+ 1e-10)
        end
    
        trans_matrix_plot = heatmap(-LMAX[1]:LMAX[1], -LMAX[2]:LMAX[2], 
                                    probabilities_motif, 
                                    aspect_ratio = 1, c = :viridis, xlabel = "i+i'", ylabel = "j+j'", 
                                    title = "P[R(i+i',j+j')=$(Rij) | R(i,j)= $(R00)]", colorbar_title = "Probability", 
                                    xticks = (Xticks_nume, Xticks_name), yticks = (Yticks_nume, Yticks_name),
                                    zlims = log_scale == true ? (-8,0) : (0,1),
                                    size = (800, 800), dpi=300, grid = false, transpose = true, frame_style=:box, 
                                    widen=false,  xrotation = 50)
        savefig(trans_matrix_plot, "$figures_path/joint_P_log_$(log_scale)_$(string(motif_idx-1, base=2)).png")
    end
end

function plot_motifs_transition_joint_prob_GIF(probabilities, rr, LMAX; log_scale=true, figures_path=".")
    mkpath(figures_path)

    Xticks_nume = [-LMAX[1]:2:-2; 0 ; 2:2:LMAX[1]]
    Xticks_name = [string.(-LMAX[1]:2:-2); "0" ;string.(2:2:LMAX[1])]
    Yticks_nume = [-LMAX[2]:2:-2; 0 ; 2:2:LMAX[2]]
    Yticks_name = [string.(-LMAX[2]:2:-2); "0" ;string.(2:2:LMAX[2])]

    for motif_idx in 1:2^2
        anim = @animate for t in 1:1:size(probabilities, 3)

            Rij, R00 = divrem(motif_idx - 1, 2)
            
            probabilities_motif = R00 == 1 ? probabilities[:, :, t, motif_idx] / rr : probabilities[:, :, t, motif_idx] / (1 - rr)
            if log_scale
                probabilities_motif = log.(probabilities_motif .+ 1e-10)
            end
        
            heatmap(-LMAX[1]:LMAX[1], -LMAX[2]:LMAX[2], 
                    probabilities_motif, 
                    aspect_ratio = 1, c = :viridis, xlabel = "i'", ylabel = "j'", 
                    title = "P[R(i+i',j+j')=$(Rij) | R(i,j)= $(R00)] (t=$t)", colorbar_title = "Probability", 
                    xticks = (Xticks_nume, Xticks_name), yticks = (Yticks_nume, Yticks_name),
                    zlims = log_scale ? (-4, 0) : (0, 1),
                    size = (800, 800), dpi=300, grid = false, transpose = true, frame_style=:box, 
                    widen=false, xrotation = 50)
        end
        gif(anim, "$figures_path/joint_P_log_$(log_scale)_$(string(motif_idx-1, base=2))_on_time.gif", fps=20)
    end
end

function plot_motifs_transition_cond_prob(joint_probabilities, isolated_probabilities, LMAX; log_scale=true, figures_path=".")
    mkpath(figures_path)

    Xticks_nume = [-LMAX[1]:2:-2; 0 ; 2:2:LMAX[1]]
    Xticks_name = ["i".*string.(-LMAX[1]:2:-2); "i" ;"i+".*string.(2:2:LMAX[1])]
    Yticks_nume = [-LMAX[2]:2:-2; 0 ; 2:2:LMAX[2]]
    Yticks_name = ["j".*string.(-LMAX[2]:2:-2); "j" ;"j+".*string.(2:2:LMAX[2])]

    # Plot transition times matrix
    for motif_idx in 1:2^2
        Rij, R00 = divrem(motif_idx - 1, 2)

        if log_scale
            probabilities = log.(probabilities .+ 1e-10)
        end
    
        trans_matrix_plot = heatmap(-LMAX[1]:LMAX[1], -LMAX[2]:LMAX[2], 
                                    probabilities[:, :, motif_idx], 
                                    aspect_ratio = 1, c = :viridis, xlabel = "i+i'", ylabel = "j+j'", 
                                    title = "P[R(i+i',j+j')=$(Rij) | R(i,j)= $(R00)]", colorbar_title = "Probability", 
                                    xticks = (Xticks_nume, Xticks_name), yticks = (Yticks_nume, Yticks_name),
                                    size = (800, 800), dpi=300, grid = false, transpose = true, frame_style=:box, 
                                    widen=false,  xrotation = 50)
        savefig(trans_matrix_plot, "$figures_path/conditional_P_log_$(log_scale)_$(string(motif_idx-1, base=2)).png")
    end
end



# Function to plot mutual information
function plot_mutual_information(series, label; max_delay=50)
    delays = -max_delay:max_delay
    mi_values = [mutual_information(series, delay) for delay in delays]
    
    plot(delays, mi_values, label=label, xlabel="Time Delay", ylabel="Mutual Information", legend=:topright)
end

function plot_cond_prob_level_curves(probabilities, rrs, is, js; log_scale=false, figures_path=".")
    mkpath(figures_path)

    i_equals_j_probs = zeros(length(is), length(rrs), 2^2)
    i_equals_minusj_probs = zeros(length(is), length(rrs), 2^2)
    vari_j0_probs = zeros(length(is), length(rrs), 2^2)
    i0_varj_probs = zeros(length(is), length(rrs), 2^2)
    for i in 1:length(is)
        for idx in 1:length(rrs)
            for motif_idx in 1:2^2
                Rij, R00 = divrem(motif_idx - 1, 2)
                i_equals_j_probs[i, idx, motif_idx] = R00==1 ?  probabilities[idx, i, i, motif_idx]/rrs[idx] : probabilities[idx, i, i, motif_idx]/(1-rrs[idx])
                i_equals_minusj_probs[i, idx, motif_idx] = R00==1 ?  probabilities[idx, i, end-i+1, motif_idx]/rrs[idx] : probabilities[idx, i, end-i+1, motif_idx]/(1-rrs[idx])
                vari_j0_probs[i, idx, motif_idx] = R00==1 ?  probabilities[idx, i, div(length(js),2)+1, motif_idx]/rrs[idx] : probabilities[idx, i, div(length(js),2)+1, motif_idx]/(1-rrs[idx])
                i0_varj_probs[i, idx, motif_idx] = R00==1 ?  probabilities[idx, div(length(is),2)+1, i, motif_idx]/rrs[idx] : probabilities[idx, div(length(is),2)+1, i, motif_idx]/(1-rrs[idx])
            end
        end
    end

    for motif_idx in 1:2^2
        Rij, R00 = divrem(motif_idx - 1, 2)
        if log_scale
            i_equals_j_probs = log.(i_equals_j_probs .+ 1e-12)
            i_equals_minusj_probs = log.(i_equals_minusj_probs .+ 1e-12)
            vari_j0_probs = log.(i_equals_j_probs .+ 1e-12)
            i0_varj_probs = log.(i_equals_j_probs .+ 1e-12)
        end
        i_equals_j_plot = plot(is, i_equals_j_probs[:, :, motif_idx], 
                                xlabel = "i'", ylabel = "Probability", 
                                title = "P[R(i+i',j+j')=$(Rij) | R(i,j)= $(R00)]; i'=j'",
                                label = "rr=".*string.(rrs'), 
                                 size = (600, 400), dpi=300, grid = false, frame_style=:box, widen=false)
        
        i_equals_minusj_plot = plot(is, i_equals_minusj_probs[:, :, motif_idx], 
                                xlabel = "i'", ylabel = "Probability", 
                                title = "P[R(i+i',j+j')=$(Rij) | R(i,j)= $(R00)]; i'=-j'",
                                label = "rr=".*string.(rrs'), 
                                 size = (600, 400), dpi=300, grid = false, frame_style=:box, widen=false)

        vari_j0_plot = plot(is, vari_j0_probs[:, :, motif_idx], 
                                xlabel = "i'", ylabel = "Probability", 
                                title = "P[R(i+i',j+j')=$(Rij) | R(i,j)= $(R00)]; -j'=0",
                                label = "rr=".*string.(rrs'), 
                                size = (600, 400), dpi=300, grid = false, frame_style=:box, widen=false)

        i0_varj_plot = plot(is, i0_varj_probs[:, :, motif_idx], 
                                xlabel = "j'", ylabel = "Probability", 
                                title = "P[R(i+i',j+j')=$(Rij) | R(i,j)= $(R00)]; i'=0'",
                                label = "rr=".*string.(rrs'), 
                                size = (600, 400), dpi=300, grid = false, frame_style=:box, widen=false)

        savefig(i_equals_j_plot, "$figures_path/i_equals_j_plot_log_$(log_scale)_$(string(motif_idx-1, base=2)).png")
        savefig(i_equals_minusj_plot, "$figures_path/i_equals_minusj_plot_log_$(log_scale)_$(string(motif_idx-1, base=2)).png")
        savefig(vari_j0_plot, "$figures_path/vari_j0_plot_log_$(log_scale)_$(string(motif_idx-1, base=2)).png")
        savefig(i0_varj_plot, "$figures_path/i0_varj_plot_log_$(log_scale)_$(string(motif_idx-1, base=2)).png")
    end

end