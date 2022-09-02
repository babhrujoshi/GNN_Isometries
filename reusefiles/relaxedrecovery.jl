using Flux

include("compressedsensing.jl")

"""
Recover with features at all layers of the network to convexify the problem; this may allow the use of other optimizers
This Function yields optimizations out of range. To get closer to the range, increase link strength

returns (code, signal)
"""
function relaxed_recover(measurements, A, generativenet::Flux.Chain, encodingdims::AbstractVector{<:Integer}; linkstrength=0.1f0, kwargs...)
    function relaxedloss(generativechain::Flux.Chain, linkstrength::AbstractFloat, truemeasurements, fullcode::AbstractVector)
        @assert length(fullcode) == length(generativechain.layers)
        linkloss = 0
        for (i, layer) in enumerate(generativechain.layers[1:end-1])
            linkloss += sum(abs2.(layer(fullcode[i]) .- fullcode[i+1])) / encodingdims[i+1]
        end
        mismatchloss = sum(abs2.(A * generativechain.layers[end](fullcode[end]) .- truemeasurements))

        mismatchloss + linkstrength * linkloss
    end
    optimloss(x, p::Tuple) = relaxedloss(p..., x)

    p = (generativenet, linkstrength, measurements)
    z0 = [randn(dims) for dims in encodingdims]

    recoveredencodings = optimise!(optimloss, p, z0)

    (recoveredencodings[1], generativenet(recoveredencodings[1]))
end

function accelerated_recovery(measurements, A, model, encoding_dims; kwargs...)
    opt = ADAM()
    code, _ = relaxed_recover(measurements, A, model, encoding_dims, opt=opt)
    recoversignal(measurements, A, model, encoding_dims[1], init_code=code, opt=opt)
end


getlayerdims(ChainDecoder::Flux.Chain{<:Tuple{Vararg{Dense}}}) =
    vcat([size(layer.weight)[2] for layer in ChainDecoder.layers], [size(ChainDecoder.layers[end].weight)[1]])

include("VAE_recovery.jl")
"""
Plot a matrix of recovery images by number for different measurement numbers
The VAE and VAE decoder should never have a final activation
VAE can be given as nothing if "inrange=false" is given.
"""
function plot_MNISTrecoveries_bynumber_bymeasurementnumber_fast(VAE, VAEdecoder, aimedmeasurementnumbers, numbers, layerdims; presigmoid=true, inrange=true, typeofdata=:test, plotwidth=600, kwargs...)

    @assert !isnothing(VAE) || inrange == false "first field VAE caonnot be nothing when in range"
    if !presigmoid #preprocess the models
        VAE = sigmoid ∘ VAE
        VAEdecoder = sigmoid ∘ VAEdecoder
    end

    MNISTtestdata = MNIST(Float32, typeofdata)
    plots = Matrix{Plots.Plot}(undef, length(numbers), length(aimedmeasurementnumbers) + 1)

    @threads for (i, number) in collect(enumerate(numbers))

        numberset = MNISTtestdata.features[:, :, MNISTtestdata.targets.==number]
        img = numberset[:, :, rand(1:size(numberset)[end])]

        truesignal, plottedtruesignal = _preprocess_MNIST_truesignal(img, VAE, presigmoid, inrange)

        plots[i, 1] = i == 1 ? plot(colorview(Gray, 1.0f0 .- reshape(plottedtruesignal, 28, 28)'), title="signal") :
                      plot(colorview(Gray, 1.0f0 .- reshape(plottedtruesignal, 28, 28)'))

        @threads for (j, aimedm) in collect(enumerate(aimedmeasurementnumbers))
            F = sampleFourierwithoutreplacement(aimedm, layerdims[end])
            measurements = F * truesignal

            recovery = accelerated_recovery(measurements, F, VAEdecoder, layerdims[1:end-1], tolerance=5e-4; kwargs...)
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

    #if !isnothing(savefile)
    #this needs data that are not plots
    #@save savefile plots metadata = Dict(:inrange => inrange, :presigmoid => presigmoid, :aimedmeasurementnumbers => aimedmeasurementnumbers, :returnplot => returnplot, :VAE => VAE)
    #end
    returnplot
end