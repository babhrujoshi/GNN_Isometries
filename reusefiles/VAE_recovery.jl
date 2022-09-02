using BSON: @save
using Base.Threads
using Images
using LsqFit
using Plots
using Printf
using DataFrames

include("vaemodels.jl")
include("compressedsensing.jl")
#include("plottingfunctions.jl")

"Makes the true signal of the correct type and shape"
function _preprocess_MNIST_truesignal(img, VAE, presigmoid, inrange)

    function inversesigmoid(y; clampmargin=1.0f-3)
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
            F = sampleFourierwithoutreplacement(aimedm, n)
            measurements = F * truesignal
            recovery = recoversignal(measurements, F, VAEdecoder, k, tolerance=5e-4; kwargs...)
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

#using BSON: @load
#@load "reusefiles/savedmodels/incoherentepoch20" model

#@time plot_MNISTrecoveries_bynumber_bymeasurementnumber(model, model.decoder, [2, 64], [2], 16, 28^2, savefile="reusefiles/experiment_data/ansdata.BSON")

"""Fit a sigmoid to data with log in y only"""
function threshold_through_fit(xdata, ydata; sigmoid_x_scale=2.5f0)
    ylogdata = log.(ydata)
    @. curve(x, p) = p[1] * sigmoid((x - p[2]) / (-sigmoid_x_scale)) + p[3]
    p0 = [3.0f0, 1.0f2, 1.5f0]
    fit = curve_fit(curve, xdata, ylogdata, p0)
    scatter(xdata, ydata, xaxis=:log, yaxis=:log) #although we do not fit for log(x) we still plot x in log scale for clarity
    (coef(fit)[2], fit, plot!(x -> exp(curve(x, coef(fit)))))
end

"""Scatter plot recovery errors for a single image, fit a sigmoid in the log-log scale, return the recovery threshold from the fit"""
function recoverythreshold_fromrandomimage(VAE, VAEdecoder, aimedmeasurementnumbers, k, n; img=nothing, presigmoid=true, inrange=true, typeofdata=:test, savefile="reusefiles/experiment_data/ansdata.BSON", kwargs...)

    # pick image at random
    @assert !isnothing(VAE) || inrange == false "first field VAE caonnot be nothing when in range"
    if !presigmoid #preprocess the models
        VAE = sigmoid ∘ VAE
        VAEdecoder = sigmoid ∘ VAEdecoder
    end

    if isnothing(img)
        dataset = MNIST(Float32, typeofdata).features
        img = dataset[:, :, rand(1:size(dataset)[3])]
    end

    truesignal, _ = _preprocess_MNIST_truesignal(img, VAE, presigmoid, inrange)

    true_ms = Vector{Float32}(undef, length(aimedmeasurementnumbers))
    recoveryerrors = Vector{Float32}(undef, length(aimedmeasurementnumbers))

    @threads for (i, aimedm) in collect(enumerate(aimedmeasurementnumbers))
        true_m, F = sampleFourierwithoutreplacement(aimedm, n, true)
        measurements = F * truesignal
        recovery = recoversignal(measurements, F, VAEdecoder, k, tolerance=5e-4; kwargs...)

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
function compare_models_from_thresholds(modelstocompare, modellabels, aimedmeasurementnumbers, numimages::Integer, k, n; typeofdata=:test, savefile="reusefiles/experiment_data/ansdata.BSON", kwargs...)
    # Still need to debug this

    dataset = MNIST(Float32, typeofdata).features

    numexperiments = numimages * length(modelstocompare)

    results = DataFrame(threshold=Vector{Float32}(undef, numexperiments),
        fitplot=Vector{Plots.Plot}(undef, numexperiments),
        fitdata=Vector{Dict{Int,Float32}}(undef, numexperiments),
        fitobject=Vector{LsqFit.LsqFitResult}(undef, numexperiments),
        image=Vector{Matrix{Float32}}(undef, numexperiments),
        modelname=Vector{String}(undef, numexperiments))

    images = [dataset[:, :, rand(1:size(dataset)[3])] for i in 1:numimages]
    @threads for (i, img) in collect(enumerate(images))

        @threads for (j, model) in collect(enumerate(modelstocompare))
            returnobj = recoverythreshold_fromrandomimage(model, model.decoder, aimedmeasurementnumbers, k, n, img=img, savefile=nothing; kwargs...)
            results[i*j, collect(keys(returnobj))] = returnobj
            results[i*j, :modelname] = modellabels[j]
            results[i*j, :image] = img
        end
        @info i #give some idea of progress
    end

    if !isnothing(savefile)
        @save savefile aimedmeasurementnumbers k n results
    end

    return results
end
#used to choose measurement number in a smart way

logrange(low_meas, high_meas, num_meas) = convert.(Int, floor.(exp.(LinRange(log(low_meas), log(high_meas), num_meas))))
#collect(0:10:220)
#@time recoverythreshold_fromrandomimage(model, model.decoder, collect(0:10:40), 16, 28^2)

"""
Make a scatter plot of recovery errors for random images for different numbers of measurements.

arguments:
models is an array of (name, model, decoder)
"""
function plot_models_recovery_errors(models::AbstractArray, aimedmeasurementnumbers::AbstractArray, k::Integer, n::Integer;
    presigmoid=true, inrange=true, typeofdata=:test, savefile="reusefiles/experiment_data/ansdata.BSON", kwargs...)

    if !presigmoid #preprocess the models
        models = [(sigmoid ∘ model[1], sigmoid ∘ model[2]) for model in models]
    end

    dataset = MNIST(Float32, typeofdata).features



    returnplot = plot()
    returndata = Dict()

    for model in models
        recoveryerrors = Vector{Float32}(undef, length(aimedmeasurementnumbers))
        true_ms = Vector{Float32}(undef, length(aimedmeasurementnumbers))

        @threads for (i, aimedm) in collect(enumerate(aimedmeasurementnumbers))
            img = dataset[:, :, rand(1:size(dataset)[3])]
            truesignal, _ = _preprocess_MNIST_truesignal(img, model[2], presigmoid, inrange)

            true_m, F = sampleFourierwithoutreplacement(aimedm, n, true)
            measurements = F * truesignal
            recovery = recoversignal(measurements, F, model[3], k, tolerance=5e-4; kwargs...)

            true_ms[i] = true_m
            recoveryerrors[i] = norm(recovery .- truesignal)
        end

        returndata[model[1]] = (true_ms, recoveryerrors)
        returnplot = scatter!(true_ms, recoveryerrors, xaxis=:log, yaxis=:log, label=model[1])
    end

    if !isnothing(savefile)
        @save savefile returndata inrange presigmoid aimedmeasurementnumbers
    end
    returnplot
end