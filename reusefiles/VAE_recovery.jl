using Plots, Images
#using Plots.PlotMeasures
using Base.Threads

include("vaemodels.jl")
include("compressedsensing.jl")

"Makes the true signal of the correct type and shape"
function _preprocess_MNIST_truesignal(img, VAE, presigmoid, inrange)

    function inversesigmoid(y; clampmargin=1f-3)
        y = clamp(y, 0.0f0 + clampmargin, 1.0f0 - clampmargin)
        log(y / (1.0f0 - y))
    end

    img = reshape(img, 28^2, :)

    truesignal = img
    plottedtruesignal = img

    if inrange
        truesignal = VAE(img, 100)
        plottedtruesignal = presigmoid ? sigmoid(truesignal) : truesignal
    elseif presigmoid
        truesignal = inversesigmoid.(img)
    end

    return truesignal, plottedtruesignal
end

"""
Plot a matrix of recovery images by number for different measurement numbers
The VAE and VAE decoder should never have a final activation
VAE can be given as nothing if "inrange=false" is given.
"""
function plot_MNISTrecoveries_bynumber_bymeasurementnumber(VAE, VAEdecoder, aimedmeasurementnumbers, numbers, k, n; presigmoid=true, inrange=true, typeofdata=:test, plotwidth=600, kwargs...)

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
            F = samplefourierwithoutreplacement(aimedm, n)
            measurements = F * truesignal
            recovery = recoversignal(measurements, F, VAEdecoder, k, tolerance=5e-4; kwargs...)
            plottedrecovery = presigmoid ? sigmoid(recovery) : recovery
            plots[i, j+1] = i == 1 ? plot(colorview(Gray, 1.0f0 .- (reshape(plottedrecovery, 28, 28)')), title="m≈$aimedm") :
                                     plot(colorview(Gray, 1.0f0 .- (reshape(plottedrecovery, 28, 28)')))
        end
    end

    scale = plotwidth / length(aimedmeasurementnumbers)
    title_plot_margin = 100
    plot(permutedims(plots)...,
        layout=(length(numbers), length(aimedmeasurementnumbers) + 1),
        size=((length(aimedmeasurementnumbers) + 1) * scale, length(numbers) * scale + title_plot_margin),
        background_color=:grey93,
        axis=([], false),
        titlefontsize=24)
end

using BSON: @load
@load "reusefiles/savedmodels/incoherentepoch20" model

plot_MNISTrecoveries_bynumber_bymeasurementnumber(nothing, model.decoder, [128], [2,5], 16, 28^2, inrange=false)