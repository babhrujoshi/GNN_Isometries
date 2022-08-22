using Plots
using LaTeXStrings
using Hyperopt

function plot_scatter_stats(results::AbstractArray{Tuple{N,N}}; kwargs...) where {N<:Number}
    resultarray = hcat(collect.(results)...)
    incoherences = resultarray[:, 1]
    means = [mean(results[2][i] for i in 1:length(results[1]) if results[1][i] == incoherence) for incoherence in incoherences]
    deviations = [std(results[2][i] for i in 1:length(results[1]) if results[1][i] == incoherence) for incoherence in incoherences]
    stats = cat(incoherences, means, deviations, dims=2)
    sorted_stats = sortslices(stats, dims=1)
    scatter(results[1], results[2]; kwargs...)
    plot!(sorted_stats[:, 1], sorted_stats[:, 2], linewidth=3.2, ribbon=sorted_stats[:, 3], fillalpha=0.5)
end

function plotmeasurementMNISTarray(; meas, recoveries, true_signal)
    plots = Array{Plots.Plot,1}(undef, length(meas))
    kwargs = Dict()
    kwargs[:axis] = ([], false)
    kwargs[:dpi] = 300
    kwargs[:titlefontsize] = 10
    for (i, m) in enumerate(meas)
        kwargs[:title] = L"m = %$(meas[i])"
        plots[i] = plot(colorview(Gray, reshape(recoveries[i], 28, 28)'); kwargs...)
    end
    kwargs[:title] = "signal"
    original_signal = plot(colorview(Gray, 1 .- true_signal)'; kwargs...)
    fig_save = plot(original_signal, plots..., size=(600, 100), layout=(1, length(meas) + 1))
    fig_save
end


##

logscale(params::AbstractVector{T}) where {T<:Real} = /(extrema(params)...) < 2e-2
@recipe function plot(ho::Hyperoptimizer)
    N = length(ho.params)
    layout --> N
    for i = 1:N
        params = getindex.(ho.history, i)
        if eltype(params) <: Real
            perm = sortperm(params)
            yguide --> "Function value"
            seriestype --> :scatter
            @series begin
                xguide --> ho.params[i]
                subplot --> i
                label --> "Sampled points"
                legend --> false
                xscale --> (logscale(params) ? :log10 : :identity)
                yscale --> (logscale(ho.results) ? :log10 : :identity)
                params[perm], ho.results[perm]
            end
        else
            params = Symbol.(params)
            uniqueparams = sort(unique(params))
            paramidxs = map(x -> findfirst(==(x), uniqueparams), params)
            yguide --> "Function value"
            seriestype --> :scatter
            @series begin
                xguide --> ho.params[i]
                subplot --> i
                label --> "Sampled points"
                legend --> false
                xticks --> (1:length(uniqueparams), uniqueparams)
                xscale --> :identity
                yscale --> (logscale(ho.results) ? :log10 : :identity)
                paramidxs, ho.results
            end
        end
    end
end