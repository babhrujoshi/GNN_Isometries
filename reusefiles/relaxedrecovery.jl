using Flux
using Flux: Chain

include("compressedsensing.jl")

"""
Recover with features at all layers of the network to convexify the problem; this may allow the use of other optimizers
This Function yields optimizations out of range. To get closer to the range, increase link strength

It seems more expensive to allocate the extra memory than the gain in optimization, so this does not result in a speedup.

returns (code, signal)
"""

function relaxedloss(measurements, A, generativenet::Flux.Chain, linkstrength::AbstractFloat, networkparts::AbstractArray, fullcode::Tuple{Vararg{<:AbstractArray}})
    @assert length(fullcode) == length(networkparts) "the lengths of  fullcode  and  networkparts  must match, they are $(length(fullcode)) and $(length(networkparts))"
    linkloss = 0
    for (i, networkpart) in enumerate(networkparts[1:end-1])
        linkloss += sum(abs2.(networkpart(fullcode[i]) .- fullcode[i+1]))
    end
    mismatchloss = sum(abs2.(A * networkparts[end](fullcode[end]) .- measurements))

    mismatchloss + linkstrength * linkloss
end


function relaxed_recover(measurements, A, generativenet::Flux.Chain, intermediate_optimlayers::AbstractArray{<:Integer}; linkstrength=1.0f0, kwargs...)

    optimloss(x, p::Tuple) = relaxedloss(p..., x)

    netparts = AbstractArray{Chain}([])
    push!(netparts, generativenet.layers[1:intermediate_optimlayers[1]] |> Chain)
    for i in 2:length(intermediate_optimlayers)
        push!(netparts, generativenet.layers[intermediate_optimlayers[i-1]+1:intermediate_optimlayers[i]] |> Flux.Chain)
    end
    push!(netparts, generativenet.layers[intermediate_optimlayers[end]+1:end] |> Chain)

    p = (measurements, A, generativenet, linkstrength, netparts)

    codes = [randn(Float32, size(generativenet.layers[1].weight)[2])]
    for index in intermediate_optimlayers
        push!(codes, randn(Float32, size(generativenet.layers[index].weight)[1]))
    end
    codes = Tuple(codes)
    # The problem is that 

    recoveredencodings = optimise!(optimloss, p, codes; kwargs...)

    netparts[end](recoveredencodings[end])
end


function accelerated_recovery(measurements, A, model; kwargs...)
    opt = ADAM()
    code, _ = relaxed_recover(measurements, A, model, opt=opt)
    recoversignal(measurements, A, model, init_code=code, opt=opt)
end



include("VAE_recovery.jl")
include("vaemodels.jl")
"""
Plot a matrix of recovery images by number for different measurement numbers
The VAE and VAE decoder should never have a final activation
VAE can be given as nothing if "inrange=false" is given.
"""
function plot_MNISTrecoveries_bynumber_bymeasurementnumber_relaxed(VAE, aimedmeasurementnumbers, numbers; linkstrength=1.0f0, intermediatelayers=collect(indexof(VAE.decoder.layers)), presigmoid=true, inrange=true, typeofdata=:test, plotwidth=600, kwargs...)
    #TODO incorporate this into the main mrecovery method with the recovery function as parameter.
    decoder = VAE.decoder
    if !presigmoid #preprocess the models
        VAE = sigmoid ∘ VAE
        decoder = sigmoid ∘ decoder
    end

    MNISTtestdata = MNIST(Float32, typeofdata)
    plots = Matrix{Plots.Plot}(undef, length(numbers), length(aimedmeasurementnumbers) + 1)

    @threads for (i, number) in collect(enumerate(numbers))

        numberset = MNISTtestdata.features[:, :, MNISTtestdata.targets.==number]
        img = numberset[:, :, rand(rng, 1:size(numberset)[end])]

        truesignal, plottedtruesignal = _preprocess_MNIST_truesignal(img, VAE, presigmoid, inrange)

        plots[i, 1] = i == 1 ? plot(colorview(Gray, 1.0f0 .- reshape(plottedtruesignal, 28, 28)'), title="signal") :
                      plot(colorview(Gray, 1.0f0 .- reshape(plottedtruesignal, 28, 28)'))

        @threads for (j, aimedm) in collect(enumerate(aimedmeasurementnumbers))
            F = sampleFourierwithoutreplacement(aimedm, length(truesignal))
            measurements = F * truesignal
            recovery = relaxed_recover(measurements, F, decoder, intermediatelayers, linkstrength=linkstrength; kwargs...)

            recoveryerror = @sprintf("%.1E", norm(recovery .- truesignal))
            plottedrecovery = presigmoid ? sigmoid(recovery) : recovery
            title = i == 1 ? "m:$aimedm er:$recoveryerror" : "er:$recoveryerror"
            plots[i, j+1] = plot(colorview(Gray, 1.0f0 .- (reshape(plottedrecovery, 28, 28)')), title=title)

        end
    end

    scale = plotwidth / length(aimedmeasurementnumbers)
    title_plot_margin = 100
    returnplot = plot(permutedims(plots)...,
        layout=(length(numbers), length(aimedmeasurementnumbers) + 1),
        size=((length(aimedmeasurementnumbers) + 1) * scale, length(numbers) * scale + title_plot_margin),
        background_color=:grey93,
        axis=([], false),
        titlefontsize=12)

    returnplot
end