using Flux
using Flux: binarycrossentropy, train!
using Statistics

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
    randcoeffs = randn(size(logvar)...)
    z = μ + randcoeffs .* exp.(0.5 .* logvar)
    return z, μ, logvar
end

struct FullVae{T}
    encoder::VaeEncoder
    decoder::T
end
Flux.@functor FullVae

(m::FullVae)(x::AbstractArray) = m.decoder(m.encoder(x)[1])

function makevae()
    encoderbody = Chain(
        Dense(784, 500, relu),
        Dense(500, 500, relu)
    )
    splitedμ = Chain(Dense(500, 20))
    splitedlogvar = Chain(Dense(500, 20))
    encoder = VaeEncoder(encoderbody, splitedμ, splitedlogvar)
    gaussinit(out, in) = randn(Float32, out, in) / sqrt(out)
    decoder = Chain(
        Dense(20 => 500, bias=false, relu, init=gaussinit),
        Dense(500 => 500, bias=false, relu, init=gaussinit),
        Dense(500 => 784, bias=false, sigmoid, init=gaussinit)
    )

    FullVae(encoder, decoder) |> gpu
end

function klfromgaussian(μ, logvar)
    0.5 * sum(@. exp(logvar) + μ^2 - logvar - 1.0)
end

function l2reg(pars)
    sum(x -> sum(abs2, x), pars)
end

Flux.Losses.mse

#loss function
function vaeloss(vaenetwork, β, λ)
    function loss(x)
        z, μ, logvar = vaenetwork.encoder(x)
        x̂ = vaenetwork.decoder(z)
        binarycrossentropy(x̂, x) + β * klfromgaussian(μ, logvar) + λ * l2reg(params(vaenetwork))
    end
end