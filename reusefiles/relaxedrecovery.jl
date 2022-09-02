using Flux

include("compressedsensing.jl")

"""
Recover with features at all layers of the network to convexify the problem; this may allow the use of other optimizers
This Function yields optimizations out of range. To get closer to the range, increase link strength

returns (code, signal)
"""
function relaxed_recover(measurements, A, generativenet::Flux.Chain, encodingdims::AbstractVector{Integer};linkstrength = 0.1f0, kwargs...)
    function relaxedloss(generativechain::Flux.Chain, linkstrength::AbstractFloat, truemeasurements, fullcode::AbstractVector)
        @assert length(fullcode) == length(generativechain.layers)
        linkloss = 0
        for (i,layer) in enumerate(generativechain.layers[1:end-1])
            linkloss += sum(abs2.(layer(fullcode[i]) .- fullcode[i+1]))/encodingdims[i+1]
        end
        mismatchloss = sum(abs2.(A*generativechain.layers[end](fullcode[end]) .- truemeasurements))

        mismatchloss + linkstrength*linkloss
    end
    optimloss(x, p::Tuple) = relaxedloss(p..., x)

    p = (generativenet, linkstrength, measurements)
    z0 = [randn(dims) for dims in encodingdims]

    recoveredencodings = optimise!(optimloss, p, z0)
    
    (recoveredencodings[1], generativenet(recoveredencodings[1]))
end

function accelerated_recovery(measurements, A, model, encoding_dims; kwargs...)
    opt = ADAM()
    code , _ = relaxed_recover(measurements,A, model, encoding_dims, opt = opt)
    recoversignal(measurements, A, model, encoding_dims[1], init_code = code, opt = opt)
end