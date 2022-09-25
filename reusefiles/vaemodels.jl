module VaeModels

using Flux, MLDatasets
using Flux: logitbinarycrossentropy, pullback, DataLoader, params
using BSON: @save
using Statistics, LinearAlgebra, FFTW
using Logging, ProgressLogging, TensorBoardLogger
using Random

export FullVae, VaeEncoder, makeVae, trainVae, trainstdVaeonMNIST, train_incoherentVAE_onMNIST

struct VaeEncoder{T,V,L}
    encoderbody::T
    splitedμ::V
    splitedlogvar::L
end
Flux.@functor VaeEncoder


function (m::VaeEncoder)(x::AbstractArray)
    intermediate = m.encoderbody(x)
    μ = m.splitedμ(intermediate)
    logvar = m.splitedlogvar(intermediate)
    return μ, logvar
end

struct FullVae{T}
    encoder::VaeEncoder
    decoder::T
end
Flux.@functor FullVae


#forward pass
function (m::FullVae)(x::AbstractArray; rng=TaskLocalRNG())
    μ, logvar = m.encoder(x)
    randcoeffs = randn(rng, Float32, size(logvar))
    z = μ .+ randcoeffs .* exp.(0.5f0 .* logvar)
    m.decoder(z)
end

#averaged forward pass
function (m::FullVae)(x::AbstractArray, n::Integer; rng=TaskLocalRNG())
    #preformance gain available by getting mu and logvar once, and sampling many times
    acc = zero(x)
    μ, logvar = m.encoder(x)

    for i in 1:n
        randcoeffs = randn(rng, Float32, size(logvar))
        z = μ .+ randcoeffs .* exp.(0.5f0 .* logvar)
        m.decoder(z)
        acc .+= m.decoder(m.encoder(x)[1])
    end
    acc ./ n
end


function makeVae(hidden, secondhidden, zlayer)
    FullVae(
        VaeEncoder(
            Chain(
                Dense(28^2 => hidden, relu),
                Dense(hidden => secondhidden)
            ),
            Dense(secondhidden => zlayer),
            Dense(secondhidden => zlayer)
        ),
        Chain(
            Dense(zlayer => secondhidden, bias=false, relu),
            Dense(secondhidden => hidden, bias=false, relu),
            Dense(hidden => 28^2, bias=false)
        )
    )
end



function trainVae(β, λ, model, pars::Flux.Params, traindata, opt::Flux.Optimise.AbstractOptimiser, numepochs, savedir, tblogdir; loginterval=10, label="")
    # The training loop for the model
    tblogger = TBLogger(tblogdir)
    saveindex = 0

    function savemodel()
        @save string(savedir, label, "intrain", saveindex) model opt
        saveindex += 1
    end

    function klfromgaussian(μ, logvar)
        0.5 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0))
    end

    function l2reg(pars)
        sum(x -> sum(abs2, x), pars)
    end

    #numbatches = length(data)
    @progress for epochnum in 1:numepochs
        for (step, x) in enumerate(traindata)

            loss, back = pullback(pars) do
                intermediate = model.encoder.encoderbody(x)
                μ = model.encoder.splitedμ(intermediate)
                logvar = model.encoder.splitedlogvar(intermediate)
                z = μ .+ randn(Float32, size(logvar)) .* exp.(0.5f0 .* logvar)
                x̂ = model.decoder(z)
                logitbinarycrossentropy(x̂, x; agg=sum) + β * klfromgaussian(μ, logvar) + λ * l2reg(pars)
            end
            gradients = back(1.0f0)
            Flux.Optimise.update!(opt, pars, gradients)

            if step % loginterval == 0
                with_logger(tblogger) do
                    @info "loss" loss
                end
            end


            # if step % saveinterval == 0
            #     savemodel()
            #     @save string(savedir, label, "_epoch_", epochnum) model opt
            # end

        end
        savemodel()
    end
    savemodel()
    @info "training complete!"
end

function trainstdVaeonMNIST()
    model = makeVae(512, 512, 16)
    batchsize = 64

    traindata = reshape(MNIST(Float32, :train).features[:, :, 1:end], 28^2, :)
    trainloader = DataLoader(traindata, batchsize=batchsize)

    trainVAE(1.0f0, 0.01f0, model, params(model), trainloader, Flux.Optimise.ADAM(), 40, "./reusefiles/models/", "./reusefiles/logs/", label="working", loginterval=100)
end

function VAEloss_unitarycoherence(x, F, model::FullVae, pars, lastlayer, β, λ, α)
    function klfromgaussian(μ, logvar)
        0.5 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0))
    end

    function l2reg(pars)
        sum(x -> sum(abs2, x), pars)
    end
    intermediate = model.encoder.encoderbody(x)
    μ = model.encoder.splitedμ(intermediate)
    logvar = model.encoder.splitedlogvar(intermediate)
    z = μ .+ randn(Float32, size(logvar)) .* exp.(0.5f0 .* logvar)
    x̂ = model.decoder(z)

    lastlayercoherence(lastlayer) = sqrt(maximum(sum(abs2, F * lastlayer, dims=2))) + sum(abs2, lastlayer' * lastlayer - I)

    logitbinarycrossentropy(x̂, x; agg=sum) + β * klfromgaussian(μ, logvar) + α * lastlayercoherence(lastlayer) + λ * l2reg(pars)
end

function VAEloss_boundedcoherence(x, F, model::FullVae, pars, lastlayer, β, λ, α)

    intermediate = model.encoder.encoderbody(x)
    μ = model.encoder.splitedμ(intermediate)
    logvar = model.encoder.splitedlogvar(intermediate)
    z = μ .+ randn(Float32, size(logvar)) .* exp.(0.5f0 .* logvar)
    x̂ = model.decoder(z)

    klgaussian = 0.5 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0))
    l2reg = sum(x -> sum(abs2, x), pars)
    boundedcoherence = log(sum(exp, (sum(abs2, F * lastlayer, dims=2)))) #taking the softmax

    logitbinarycrossentropy(x̂, x; agg=sum) + β * klgaussian + α * boundedcoherence + λ * l2reg
end

function VAEloss_boundedcoherence(x, F, model::FullVae, pars, lastlayer, β, λ, α, tblogger)

    intermediate = model.encoder.encoderbody(x)
    μ = model.encoder.splitedμ(intermediate)
    logvar = model.encoder.splitedlogvar(intermediate)
    z = μ .+ randn(Float32, size(logvar)) .* exp.(0.5f0 .* logvar)
    x̂ = model.decoder(z)

    klgaussian = 0.5 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0))
    l2reg = sum(x -> sum(abs2, x), pars)
    boundedcoherence = log(sum(exp, (sum(abs2, F * lastlayer, dims=2)))) #taking the softmax

    with_logger(tblogger) do
        @info "loss terms" logitbinarycrossentropy(x̂, x; agg=sum) β * klgaussian α * boundedcoherence λ * l2reg
    end
end


function trainincoherentVae(vaelossfn, β, λ, α, F, model, pars::Flux.Params, traindata, opt::Flux.Optimise.AbstractOptimiser, numepochs, savedir, tblogdir; loginterval=10, label="", logginglossterms=false)
    # The training loop for the model
    tblogger = TBLogger(tblogdir)

    function klfromgaussian(μ, logvar)
        0.5 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0))
    end

    function l2reg(pars)
        sum(x -> sum(abs2, x), pars)
    end

    lastlayer = params(model.decoder[end])[1]
    #lastlayercoherence() = maximum(sqrt.(sum((F * lastlayer) .* (F * lastlayer), dims=2))) + norm(lastlayer' * lastlayer - I(500), 2)^2
    lastlayercoherence(lastlayer) = sqrt(maximum(sum(abs2, F * lastlayer, dims=2))) + sum(abs2, lastlayer' * lastlayer - I)


    #numbatches = length(data)
    @progress for epochnum in 1:numepochs
        for (step, x) in enumerate(traindata)

            loss, back = pullback(pars) do
                vaelossfn(x, F, model, pars, lastlayer, β, λ, α)
            end
            gradients = back(1.0f0)
            Flux.Optimise.update!(opt, pars, gradients)

            if step % loginterval == 0
                with_logger(tblogger) do

                    if logginglossterms
                        vaelossfn(x, F, model, pars, lastlayer, β, λ, α, tblogger)
                    end
                    @info "loss" epochnum loss
                end
            end
        end
        @save string(savedir, label, "epoch", epochnum) model opt epochnum β λ α label
    end

    @info "training complete!"
end

function train_incoherentVAE_onMNIST(; vaelossfn=VAEloss_boundedcoherence, numepochs=20, β=1.0f0, λ=1.0f-2, α=1.0f4, kwargs...)
    model = makeVae(512, 512, 16)
    batchsize = 64

    traindata = reshape(MNIST(Float32, :train).features[:, :, 1:end], 28^2, :)
    trainloader = DataLoader(traindata, batchsize=batchsize)

    F = dct(diagm(ones(28^2)), 2)

    trainincoherentVae(vaelossfn, β, λ, α, F, model, params(model), trainloader, Flux.Optimise.ADAM(), numepochs, "./reusefiles/models/", "./reusefiles/logs/"; kwargs...)
end

end


