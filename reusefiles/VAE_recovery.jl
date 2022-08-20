using Plots, Images
using Plots.PlotMeasures
using Base.Threads

include("vaemodels.jl")
include("compressedsensing.jl")

function plot_randomarrayoftestMNISTrecoveries(decoder, numimg, aimedm, k, n; plotwidth=800, kwargs...)
    MNISTtestdata = reshape(MNIST(Float32, :test).features, 28^2, :)

    function getimgpair()
        truesignal = MNISTtestdata[:, rand(1:size(MNISTtestdata)[2])]
        F = samplefourierwithoutreplacement(aimedm, n)
        measurements = F * truesignal
        recovery = recoversignal(measurements, F, decoder, k, tolerance=5e-4; kwargs...)
        return [reshape(truesignal, 28, 28)', reshape(recovery, 28, 28)']
    end

    plots = vcat([hcat((plot(colorview(Gray, 1.0f0 .- img), axis=([], false)) for img in getimgpair())...) for i in 1:numimg]...)
    scale = plotwidth / numimg
    plot(plots..., layout=(2, numimg), size=(numimg * scale, 2 * scale), background_color=:grey93)
end

function plot_randomarrayoftestMNIST_presigmoid_recoveries(VAE_withoutsigmoid, decoder_withoutsigmoid, numimg, aimedm, k, n; plotwidth=800, kwargs...)
    MNISTtestdata = reshape(MNIST(Float32, :test).features, 28^2, :)

    function getimgpair()
        img = MNISTtestdata[:, rand(1:size(MNISTtestdata)[2])]
        truesignal = VAE_withoutsigmoid(img, 100)
        F = samplefourierwithoutreplacement(aimedm, n)
        measurements = F * truesignal
        recovery = recoversignal(measurements, F, decoder_withoutsigmoid, k, tolerance=5e-4; kwargs...)
        return [reshape(sigmoid(truesignal), 28, 28)', reshape(sigmoid(recovery), 28, 28)']
    end

    plots = vcat([hcat((plot(colorview(Gray, 1.0f0 .- img), axis=([], false)) for img in getimgpair())...) for i in 1:numimg]...)
    scale = plotwidth / numimg
    plot(plots..., layout=(2, numimg), size=(numimg * scale, 2 * scale), background_color=:grey93)
end


""" if not isbeforesigmoid, then the sigmoid should be included in the model and decoder"""
function plot_testMNIST_inrange_bynumber_bymeasurementnumber_recoveries(VAE, VAEdecoder, aimedmeasurementnumbers, numbers, k, n; isbeforesigmoid=true, inrange=true, plotwidth=600, kwargs...)

    MNISTtestdata = MNIST(Float32, :test)
    plots = Matrix{Plots.Plot}(undef, length(numbers), length(aimedmeasurementnumbers) + 1)

    function inversesigmoid(y, clampmargin)
        y = clamp(y, 0.0f0 + clampmargin, 1.0f0 - clampmargin)
        log(y / (1.0f0 - y))
    end


    @threads for (i, number) in collect(enumerate(numbers))

        numberset = reshape(MNISTtestdata.features[:, :, MNISTtestdata.targets.==number], 28^2, :)
        img = numberset[:, rand(1:size(numberset)[end])]

        if inrange
            truesignal = VAE(img, 100)
            plottedtruesignal = isbeforesigmoid ? sigmoid(truesignal) : truesignal
        elseif isbeforesigmoid
            truesignal = inversesigmoid.(img, 1.0f-3)
            plottedtruesignal = img
        else #after sigmoid and not in range
            plottedtruesignal = img
            truesignal = img
        end

        if (i == 1)
            plots[i, 1] = plot(colorview(Gray, 1.0f0 .- reshape(plottedtruesignal, 28, 28)'), title="signal")
        else
            plots[i, 1] = plot(colorview(Gray, 1.0f0 .- reshape(plottedtruesignal, 28, 28)'))
        end

        @threads for (j, aimedm) in collect(enumerate(aimedmeasurementnumbers))
            F = samplefourierwithoutreplacement(aimedm, n)
            measurements = F * truesignal
            recovery = recoversignal(measurements, F, VAEdecoder, k, tolerance=5e-4; kwargs...)
            plottedrecovery = isbeforesigmoid ? sigmoid(recovery) : recovery
            plots[i, j+1] = plot(colorview(Gray, 1.0f0 .- (reshape(plottedrecovery, 28, 28)')))
            if i == 1
                title!("m≈$aimedm")
            end
        end
    end

    scale = plotwidth / length(aimedmeasurementnumbers)
    title_plot_margin = 50
    plot(permutedims(plots)...,
        layout=(length(numbers), length(aimedmeasurementnumbers) + 1),
        size=((length(aimedmeasurementnumbers) + 1) * scale, length(numbers) * scale + title_plot_margin),
        background_color=:grey93,
        axis=([], false),
        titlefontsize=12)
end

using BSON: @load
@load "reusefiles/savedmodels/incoherentepoch20" model

@time plot_testMNIST_inrange_bynumber_bymeasurementnumber_recoveries(model, model.decoder, [16, 32, 64, 128, 256, 512], [2, 4, 5, 7, 8], 16, 28^2, isbeforesigmoid=true, inrange=false)
#     function getimgpair()
#         img = MNISTtestdata[:, rand(1:size(MNISTtestdata)[2])]
#         truesignal = VAE_withoutsigmoid(img, 100)
#         F = samplefourierwithoutreplacement(aimedm, n)
#         measurements = F * truesignal
#         recovery = recoversignal(measurements, F, decoder_withoutsigmoid, k, tolerance=5e-4; kwargs...)
#         return [reshape(sigmoid(truesignal), 28, 28)', reshape(sigmoid(recovery), 28, 28)']
#     end

#     plots = vcat([hcat((plot(colorview(Gray, 1.0f0 .- img), axis=([], false)) for img in getimgpair())...) for i in 1:numimg]...)
#     scale = plotwidth / numimg
#     plot(plots..., layout=(2, numimg), size=(numimg * scale, 2 * scale), background_color=:grey93)
# end

# using Plots
# numimg = AbstractArray{Plots.Plot,2}(undef, (2, 2))

# plot_testMNIST_bynumber_bymeasurementnumber_presigmoid_recoveries