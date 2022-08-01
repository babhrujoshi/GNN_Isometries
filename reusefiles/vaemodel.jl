using Flux
using Flux: binarycrossentropy, train!

struct VaeEncoder
    encoderbody::Chain
    splitedμ::Chain
    splitedlogvar::Chain
end

Flux.@functor VaeEncoder

function (m::VaeEncoder)(x::AbstractArray)
    intermediate = m.encoderbody(x)
    μ = m.splitedμ(intermediate)
    logvar = m.splitedlogvar(intermediate)
    z = μ + map(t -> randn() * t, exp.(0.5 .* logvar))
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

    gaussinit(out, in) = randn(out, in) / sqrt(out)
    decoder = Chain(
        Dense(20 => 500, bias=false, relu, init=gaussinit),
        Dense(500 => 500, bias=false, relu, init=gaussinit),
        Dense(500 => 784, bias=false, sigmoid, init=gaussinit)
    )

    FullVae(encoder, decoder)
end


#loss function
function vaeloss(vaenetwork, β, λ)
    @assert size(x)[end] != 0

    function loss(x)
        z, μ, logvar = vaenetwork.encoder(x)
        x̂ = vaenetwork.decoder(z)

        #mismatch + kl from gaussian + l2 regularisation
        binarycrossentropy(x̂, x, agg=sum) - 0.5β * sum(@. exp(logvar) + μ^2 - logvar - 1.0) + sum(t -> sum(abs2, t), params(vaenetwork))
    end
end



