using Flux
using Flux: binarycrossentropy, train!

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
    gaussinit(out, in) = randn(Float32, out, in) / sqrt(out)
    decoder = Chain(
        Dense(20 => 500, bias=false, relu, init=gaussinit),
        Dense(500 => 500, bias=false, relu, init=gaussinit),
        Dense(500 => 784, bias=false, sigmoid, init=gaussinit)
    )

    FullVae(encoder, decoder) |> gpu
end


#loss function
function vaeloss(vaenetwork, β, λ)
    function loss(x)
        z, μ, logvar = vaenetwork.encoder(x)
        x̂ = vaenetwork.decoder(z)

        #mismatch + kl from gaussian + l2 regularisation
        mismatch = binarycrossentropy(x̂, x, agg=sum)
        klfromgaussian = - 0.5β * sum(@. exp(logvar) + μ^2 - logvar - 1.0)
        l2reg = λ*sum(t -> sum(abs2, t), params(vaenetwork))

        mismatch + klfromgaussian + l2reg
    end
end



