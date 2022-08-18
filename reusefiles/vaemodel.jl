using Flux
using Flux: binarycrossentropy, pullback
using BSON: @save
using Statistics
using ProgressLogging
using TensorBoardLogger
using Logging


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
    randcoeffs = randn(Float32, size(logvar))
    z = μ .+ randcoeffs .* exp.(0.5f0 .* logvar)
    return z, μ, logvar
end

struct FullVae{T}
    encoder::VaeEncoder
    decoder::T
end
Flux.@functor FullVae

#forward pass
(m::FullVae)(x::AbstractArray) = m.decoder(m.encoder(x)[1])

#averaged forward pass
function (m::FullVae)(x::AbstractArray, n::Integer)
    acc = zero(x)
    for i in 1:n
        acc .+= m.decoder(m.encoder(x)[1])
    end
    acc ./ n
end

function makeVAE(hidden, secondhidden, zlayer)
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
            Dense(hidden => 28^2, bias=false, sigmoid)
        )
    )
end



function trainVAE(β, λ, model, pars::Flux.Params, traindata, opt::Flux.Optimise.AbstractOptimiser, numepochs, savedir, tblogdir; loginterval=10, label="")
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
        for (step, x::AbstractArray{Float32}) in enumerate(traindata)

            loss, back = pullback(pars) do
                len = length(x)
                intermediate = model.encoder.encoderbody(x)
                μ = model.encoder.splitedμ(intermediate)
                logvar = model.encoder.splitedlogvar(intermediate)
                z = μ .+ randn(Float32, size(logvar)) .* exp.(0.5f0 .* logvar)
                x̂ = model.decoder(z)
                binarycrossentropy(x̂, x; agg=sum) / len + β * klfromgaussian(μ, logvar) / len + λ * l2reg(pars)
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