using Plots, Images
#using Plots.PlotMeasures
using Base.Threads
using Printf
using Plots
using BSON: @save
using LsqFit

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


#Scatter plot the measurement recoveries for a single image
function recoverythreshold_fromrandomimage(VAE, VAEdecoder, aimedmeasurementnumbers, k, n; presigmoid=true, inrange=true, typeofdata=:test, savefile="reusefiles/experiment_data/ansdata.BSON", kwargs...)

    # pick image at random
    @assert !isnothing(VAE) || inrange == false "first field VAE caonnot be nothing when in range"
    if !presigmoid #preprocess the models
        VAE = sigmoid ∘ VAE
        VAEdecoder = sigmoid ∘ VAEdecoder
    end

    dataset = MNIST(Float32, typeofdata).features
    img = dataset[:, :, rand(1:size(dataset)[3])]
    truesignal, _ = _preprocess_MNIST_truesignal(img, VAE, presigmoid, inrange)

    recoveryerrors = Vector{Float32}(undef, length(aimedmeasurementnumbers))
    true_ms = Vector{Float32}(undef, length(aimedmeasurementnumbers))

    @threads for (i, aimedm) in collect(enumerate(aimedmeasurementnumbers))
        true_m, F = sampleFourierwithoutreplacement(aimedm, n, true)
        measurements = F * truesignal
        recovery = recoversignal(measurements, F, VAEdecoder, k, tolerance=5e-4; kwargs...)

        true_ms[i] = true_m
        recoveryerrors[i] = norm(recovery .- truesignal)
    end


    returnplot = nothing
    threshold = NaN
    #err = NaN
    @assert length(aimedmeasurementnumbers) ≥ 5 "need at least 5 recoveries to fit error (for invertible Hessian)"
    #try
    #fit a sigmoid in double log scale
    returnplot = scatter(true_ms, recoveryerrors, xaxis=:log, yaxis=:log)
    @. curve(x, p) = exp(p[1] * sigmoid((log.(x) - p[2]) / p[3]) + p[4])
    p0 = [2.5f0, 4.2f0, -0.15f0, 1.5f0]

    fit = curve_fit(curve, true_ms, recoveryerrors, p0)
    #plot!(x->curve(x, p0))
    plot!(x -> curve(x, coef(fit)))
    threshold = exp(coef(fit)[2])

    #err = stderror(fit)[2] * threshold
    @info "threshold" threshold
    #scatter!([threshold], [curve(threshold, coef(fit))], xerror=err) #scale by derivative

    #    catch
    #        returnplot = scatter(true_ms, log.(recoveryerrors), title="Recovery Errors", yaxis=:log, xaxis=:log)
    #        @warn "could not fit the data"
    #    end

    if !isnothing(savefile)
        @save savefile true_ms recoveryerrors threshold metadata = Dict(:truesignal => truesignal, :inrange => inrange, :presigmoid => presigmoid, :aimedmeasurementnumbers => aimedmeasurementnumbers, :VAE => VAE)
    end
    returnplot
end

#used to choose measurement number in a smart way
logrange(low_meas, high_meas, num_meas) = convert.(Int, floor.(exp.(LinRange(log(low_meas), log(high_meas), num_meas))))
#collect(0:10:220)
#@time scatter_MNISTimage_recoveryerrors(model, model.decoder, collect(0:10:40), 16, 28^2)