using Flux: params, gradient, update!
using FFTW
using Distributions
using LinearAlgebra
using TensorBoardLogger
using Logging
##



"""
    optimise(init_z, loss, opt, tolerance, [out_toggle = 0,][max_iter = 1_000_000])

    Optimization that stops when the gradient is small enough

    loss takes z as an argument.
"""
function optimise!(loss, z; opt=Flux.Optimise.ADAM(0.001), tolerance=5e-4, out_toggle=1e2, max_iter=15_000, tblogdir=nothing)
    tol2 = tolerance^2
    usingtb = !isnothing(tblogdir)
    logger = usingtb ? TBLogger(tblogdir) : current_logger()

    ps = params(z)
    iter = 1
    arglessloss() = loss(z)
    succerror = 0.0f0
    with_logger(logger) do
        while true
            if iter > max_iter
                @warn "Max num. iterations reached"
                return missing
            end
            grads = gradient(arglessloss, ps) #loss cannot have any arguments
            update!(opt, ps, grads)
            succerror = sum(abs2, grads[z])
            if usingtb && out_toggle != 0 && iter % out_toggle == 0
                @info "recovery optimization step" iter grad_size = sqrt(succerror) lossval = sqrt(loss(z))
            end
            if succerror < tol2

                break
            end
            iter += 1
        end
    end
    @info "final stats" error = sqrt(succerror) iter
    return z
end


##

function recoversignal(measurements, A, model, code_dim; kwargs...)
    @debug "Starting Image Recovery"
    function loss(codeguess)
        return sum(abs2, A * model(codeguess) - measurements)
    end
    #model(optimise!(loss, randn(code_dim) / sqrt(code_dim); kwargs...))
    model(optimise!(loss, randn(code_dim) / sqrt(code_dim); kwargs...))
    #return opt_code != nothing ? model(opt_code) : nothing
end


function recoveryerror(x₀, model, A, k; kwargs...)
    y = A * x₀
    norm(recoversignal(y, A, model, k; kwargs...) - x₀)
end

function samplefourierwithoutreplacement(aimed_m, n)
    F = dct(diagm(ones(n)), 2)
    sampling = rand(Bernoulli(aimed_m / n), n)
    true_m = sum(sampling)
    F[sampling, :] * sqrt(n / true_m)
    #return get_true_m ? (true_m, normalized_F) : normalized_F # normalize it
end

function fullfourier(dim)
    dct(diagm(ones(dim)), 2)
end

function singlerecoveryfourierexperiment(x₀, model, k, n, aimed_m; kwargs...)
    A = samplefourierwithoutreplacement(aimed_m, n)
    measurements = A * x₀
    recoveredsignal = recoversignal(measurements, A, model, k; kwargs...)
    return recoveredsignal, norm(recoveredsignal - x₀)
end