using BSON: @save
using Base.Threads
using Images
using LsqFit
using Plots
using Printf
using DataFrames
using Random
using CairoMakie
using CairoMakie: Axis
using Flux
using MLDatasets

include("vaemodels.jl")

using .VaeModels

include("compressedsensing.jl")
#include("plottingfunctions.jl")

"Makes the true signal of the correct type and shape"
function _preprocess_MNIST_truesignal(img, VAE::Union{FullVae,ComposedFunction}, presigmoid, inrange; rng=TaskLocalRNG())

    function inversesigmoid(y; clampmargin=1.0f-3)
        y = clamp(y, 0.0f0 + clampmargin, 1.0f0 - clampmargin)
        log(y / (1.0f0 - y))
    end

    img = reshape(img, :)

    truesignal = img
    plottedtruesignal = img

    if inrange
        truesignal = VAE(img, 100, rng=rng)
        plottedtruesignal = presigmoid ? sigmoid(truesignal) : truesignal
    elseif presigmoid
        truesignal = inversesigmoid.(img)
    end

    return truesignal, plottedtruesignal
end


function imagesfromnumbers(numbers::AbstractArray{<:Integer}, typeofdata; rng=TaskLocalRNG())
    data = MNIST(Float32, typeofdata)
    images = []
    for number in numbers
        numberset = data.features[:, :, data.targets.==number]
        push!(images, numberset[:, :, rand(rng, 1:size(numberset)[end])])
    end
    convert(AbstractArray{typeof(images[1])}, images)
end

function imagesfromnumbers(numbers::Integer, typeofdata; rng=TaskLocalRNG())
    data = MNIST(Float32, typeofdata)
    numberset = data.features[:, :, data.targets.==numbers]
    numberset[:, :, rand(rng, 1:size(numberset)[end])]
end


getlayerdims(ChainDecoder::Flux.Chain{<:Tuple{Vararg{Dense}}}) =
    vcat([size(layer.weight)[2] for layer in ChainDecoder.layers], [size(ChainDecoder.layers[end].weight)[1]])

"""
Plot a matrix of recovery images by number for different measurement numbers
The VAE and VAE decoder should never have a final activation
VAE can be given as nothing if "inrange=false" is given.
"""
function plot_MNISTrecoveries(VAE::FullVae, aimedmeasurementnumbers::Vector{<:Integer}, images::Vector{<:AbstractArray}; recoveryfn=recoversignal, presigmoid=true, inrange=true, typeofdata=:test, rng=TaskLocalRNG(), plotwidth=600, kwargs...)
    #TODO incorporate this into the main mrecovery method with the recovery function as parameter.
    if !presigmoid #preprocess the models
        lastlayer = VAE.decoder.layers[end]
        newlastlayer = Dense(lastlayer.weight, false, sigmoid)
        VAE = FullVae(VAE.encoder, Chain(VAE.decoder.layers[1:end-1]..., newlastlayer))
    end
    decoder = VAE.decoder

    signalimages = Vector{Vector{Float32}}(undef, length(images))
    plotimages = Matrix{Vector{Float32}}(undef, length(images), length(aimedmeasurementnumbers))
    recoveryerrors = Matrix{Float32}(undef, length(images), length(aimedmeasurementnumbers))
    F = fouriermatrix(length(images[1]))
    n = length(images[1])
    @threads for (i, img) in collect(enumerate(images))
        truesignal, plottedtruesignal = _preprocess_MNIST_truesignal(img, VAE, presigmoid, inrange, rng=rng)
        signalimages[i] = plottedtruesignal
        @threads for (j, aimedm) in collect(enumerate(aimedmeasurementnumbers))
            freq = rand(rng, Bernoulli(aimedm / n), n)
            @views sampledF = F[freq, :]
            measurements = sampledF * truesignal
            recovery = recoveryfn(measurements, sampledF, decoder; kwargs...)
            recoveryerrors[i, j] = norm(recovery .- truesignal)
            plotimages[i, j] = presigmoid ? sigmoid(recovery) : recovery
        end
    end

    f = Figure(resolution=(200 * (length(aimedmeasurementnumbers) + 1), 200 * length(images) + 100), backgroundcolor=:lightgrey)
    Label(f[1, 1], "signal", tellheight=true, tellwidth=false, textsize=20)
    for (i, signalimage) in enumerate(signalimages)
        ax = Axis(f[i+1, 1], aspect=1)
        hidedecorations!(ax)
        CairoMakie.heatmap!(ax, 1.0f0 .- reshape(signalimage, 28, 28)[:, end:-1:1], colormap=:grays)
    end
    for i in 1:size(plotimages)[1], j in 1:size(plotimages)[2]
        ax = Axis(f[i+1, j+1], aspect=1, title="err: $(@sprintf("%.1E", recoveryerrors[i, j]))")
        hidedecorations!(ax)
        CairoMakie.heatmap!(ax, 1.0f0 .- reshape(plotimages[i, j], 28, 28)[:, end:-1:1], colormap=:grays)
    end
    for (i, m) in enumerate(aimedmeasurementnumbers)
        Label(f[1, i+1], "m:$m", tellheight=true, tellwidth=false, textsize=20)
    end
    f
end

"""
Deal with inputs as arrays or single entries
"""
@generated function plot_MNISTrecoveries(VAE, aimedmeasurementnumbers, numbers; typeofdata=:test, rng=TaskLocalRNG(), kwargs...)

    if aimedmeasurementnumbers <: Integer
        measnum = :([aimedmeasurementnumbers])
    elseif aimedmeasurementnumbers <: Vector{<:Integer}
        measnum = :(aimedmeasurementnumbers)
    else
        throw(MethodError(plot_MNISTrecoveries, (VAE, aimedmeasurementnumbers, numbers)))
    end

    if !(numbers <: Vector)
        imagesexpr = :([images])
    else
        imagesexpr = :(images)
    end

    return quote
        images = imagesfromnumbers(numbers, typeofdata, rng=rng)
        plot_MNISTrecoveries(VAE, $measnum, $imagesexpr, rng=rng; kwargs...)
    end
end

"Compare many models; this plots the recoveries for each model, keeping the measurements and signal images consistent as much as possible"
function compare_models_MNISTrecoveries(models::Vector{<:FullVae}, aimedmeasurementnumbers, numbers; typeofdata=:test, rng=TaskLocalRNG(), kwargs...)
    returnplots = []
    images = imagesfromnumbers(numbers, typeofdata, rng=rng)
    seed = rand(rng, 1:500)
    for vae in models
        rng = Xoshiro(seed)
        push!(returnplots, plot_MNISTrecoveries(vae, aimedmeasurementnumbers, numbers, rng=rng; kwargs...))
    end
    returnplots
end


function plot_MNISTrecoveries(recoveryfns::Vector{<:Function}, VAE::FullVae, aimedmeasurementnumbers::AbstractArray{<:Integer}, numbers::AbstractArray{<:Integer}; rng=TaskLocalRNG(), typeofdata=:test, kwargs...)
    #TODO incorporate this into the main mrecovery method with the recovery function as parameter.
    images = imagesfromnumbers(numbers, typeofdata, rng=rng)
    plots = []
    for fnpick in recoveryfns
        @time myplot = plot_MNISTrecoveries(VAE, aimedmeasurementnumbers, images; recoveryfn=fnpick, rng=rng, kwargs...)
        push!(plots, myplot)
    end
    plots
end


function plot_MNISTrecoveries(VAE::FullVae, aimedmeasurementnumbers::AbstractArray{<:Integer}, numbers::AbstractArray{<:Integer}, recoveryfunctions::AbstractArray; seed=53, kwargs...)
    plots = []

    for recoveryfn in recoveryfunctions
        rng = Xoshiro(seed)
        #numbersets = [MNISTtestdata.features[:,:, MNISTtestdata.targets.== number] for number in 1:9]
        push!(plots, plot_MNISTrecoveries(VAE, aimedmeasurementnumbers, numbers, recoveryfn=recoveryfn, rng=rng; kwargs...))
    end
    plots
end


"""Fit a sigmoid to data with log in y only"""
function threshold_through_fit(xdata, ydata; sigmoid_x_scale=2.5f0)
    ylogdata = log.(ydata)
    @. curve(x, p) = p[1] * sigmoid((x - p[2]) / (-sigmoid_x_scale)) + p[3]
    p0 = [3.0f0, 1.0f2, 1.5f0]
    fit = curve_fit(curve, xdata, ylogdata, p0)
    scatter(xdata, ydata, yaxis=:log) #although we do not fit for log(x) we still plot x in log scale for clarity
    (coef(fit)[2], fit, plot!(x -> exp(curve(x, coef(fit)))))
end

"""Scatter plot recovery errors for a single image, fit a sigmoid in the log-log scale, return the recovery threshold from the fit"""
function recoverythreshold_fromrandomimage(VAE, aimedmeasurementnumbers; img=nothing, presigmoid=true, inrange=true, typeofdata=:test, savefile="reusefiles/experiment_data/ansdata.BSON", kwargs...)

    # pick image at random
    decoder = VAE.decoder
    if !presigmoid #preprocess the models
        VAE = sigmoid ∘ VAE
        decoder = sigmoid ∘ decoder
    end

    if isnothing(img)
        dataset = MNIST(Float32, typeofdata).features
        img = dataset[:, :, rand(1:size(dataset)[3])]
    end

    truesignal, _ = _preprocess_MNIST_truesignal(img, VAE, presigmoid, inrange)

    true_ms = Vector{Float32}(undef, length(aimedmeasurementnumbers))
    recoveryerrors = Vector{Float32}(undef, length(aimedmeasurementnumbers))

    @threads for (i, aimedm) in collect(enumerate(aimedmeasurementnumbers))
        true_m, F = sampleFourierwithoutreplacement(aimedm, getlayerdims(decoder)[end], true)
        measurements = F * truesignal
        recovery = recoversignal(measurements, F, decoder; kwargs...)

        true_ms[i] = true_m
        recoveryerrors[i] = norm(recovery - truesignal)
    end

    threshold, fit, returnplot = threshold_through_fit(true_ms, recoveryerrors)

    datapoints = hcat(true_ms, recoveryerrors)

    if !isnothing(savefile)
        @save savefile true_ms recoveryerrors threshold truesignal inrange presigmoid aimedmeasurementnumbers VAE fit
    end

    (threshold=threshold, fitplot=returnplot, fitdata=datapoints, fitobject=fit) #threshold, and things to check if threshold is accurate
end

"""Compare models through the recovery threshold of a small number of images"""
function compare_models_from_thresholds(modelstocompare, modellabels, aimedmeasurementnumbers, numimages::Integer; typeofdata=:test, savefile="reusefiles/experiment_data/ansdata.BSON", kwargs...)
    # Still need to debug this

    dataset = MNIST(Float32, typeofdata).features

    numexperiments = numimages * length(modelstocompare)

    results = DataFrame(threshold=Vector{Float32}(undef, numexperiments),
        fitplot=Vector{Plots.Plot}(undef, numexperiments),
        fitdata=Vector{Matrix{Float32}}(undef, numexperiments),
        fitobject=Vector{LsqFit.LsqFitResult}(undef, numexperiments),
        image=Vector{Matrix{Float32}}(undef, numexperiments),
        modelname=Vector{String}(undef, numexperiments))

    images = [dataset[:, :, rand(1:size(dataset)[3])] for i in 1:numimages]
    @threads for (i, img) in collect(enumerate(images))
        @threads for (j, model) in collect(enumerate(modelstocompare))
            returnobj = recoverythreshold_fromrandomimage(model, aimedmeasurementnumbers, img=img, savefile=nothing; kwargs...)
            results[i+(j-1)*numimages, collect(keys(returnobj))] = returnobj
            results[i+(j-1)*numimages, :modelname] = modellabels[j]
            results[i+(j-1)*numimages, :image] = img
        end
        @info i #give some idea of progress
    end

    if !isnothing(savefile)
        @save savefile aimedmeasurementnumbers results
    end

    return results
end
#used to choose measurement number in a smart way

logrange(low_meas, high_meas, num_meas) = convert.(Int, floor.(exp.(LinRange(log(low_meas), log(high_meas), num_meas))))
#collect(0:10:220)
#@time recoverythreshold_fromrandomimage(model, model.decoder, collect(0:10:40), 16, 28^2)

"""
Make a scatter plot of recovery errors for random images for different numbers of measurements.
"""
function plot_models_recovery_errors(models::Vector{<:FullVae}, modellabels::Vector{<:AbstractString}, aimedmeasurementnumbers::AbstractArray;
    presigmoid=true, inrange=true, typeofdata=:test, savefile="reusefiles/experiment_data/ansdata.BSON", kwargs...)

    if !presigmoid #preprocess the models
        for model in models
            model.decoder = sigmoid ∘ model.decoder
        end
    end

    dataset = MNIST(Float32, typeofdata).features

    returnplot = plot()
    returndata = Dict()

    returnplot = plot()
    for (label, model) in zip(modellabels, models)
        recoveryerrors = Vector{Float32}(undef, length(aimedmeasurementnumbers))
        true_ms = Vector{Float32}(undef, length(aimedmeasurementnumbers))

        @threads for (i, aimedm) in collect(enumerate(aimedmeasurementnumbers))
            img = dataset[:, :, rand(1:size(dataset)[3])]
            truesignal, _ = _preprocess_MNIST_truesignal(img, model, presigmoid, inrange)

            true_m, F = sampleFourierwithoutreplacement(aimedm, getlayerdims(model.decoder)[end], true)
            measurements = F * truesignal
            recovery = recoversignal(measurements, F, model.decoder; kwargs...)

            true_ms[i] = true_m
            recoveryerrors[i] = norm(recovery .- truesignal)
        end

        returndata[label] = (true_ms, recoveryerrors)
        returnplot = scatter!(true_ms, recoveryerrors, yaxis=:log, label=label)
    end

    if !isnothing(savefile)
        @save savefile returndata inrange presigmoid aimedmeasurementnumbers
    end
    returnplot
end
