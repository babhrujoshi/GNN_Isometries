using Flux: params, gradient, update!

##
"""
    optimise(init_z, loss, opt, tolerance, [out_toggle = 0,][max_iter = 1_000_000])

    Optimization that stops when the gradient is small enough

    loss takes z as an argument.
"""
function optimise!(loss, z; opt::Flux.Optimise.AbstractOptimiser=Flux.Optimise.ADAM(0.001), tolerance=1e-5, out_toggle=1e5, max_iter::Integer=500_000)
    tol2 = tolerance^2
    ps = params(z)
    iter = 1
    arglessloss() = loss(z)
    while true
        if iter > max_iter
            @warn "Max num. iterations reached"
            return missing
        end
        grads = gradient(arglessloss, ps) #loss cannot have any arguments
        update!(opt, ps, grads)
        succ_error = sum(abs2, grads[z])
        if out_toggle != 0 && iter % out_toggle == 0
            @info "====> In Gradient:" iter grad_size = (sqrt(succ_error)) tolerance error = string(sqrt(loss(z)))
        end
        if succ_error < tol2
            @info sqrt(succ_error)
            break
        end
        iter += 1
    end
    return z
end


##

function recoversignal(measurements, A, model, code_dim; kwargs...)
    function loss(codeguess)
        return sum(abs2, A * model(codeguess) - measurements)
    end

    model(optimise!(loss, randn(code_dim) / sqrt(code_dim); kwargs...))
    #return opt_code != nothing ? model(opt_code) : nothing
end


function recoveryerror(x₀, model, A, k; kwargs...)
    y = A * x₀
    norm(recoversignal(y, A, model, k; kwargs...) - x₀)
end

function samplefourierwithoutreplacement(aimed_m, n, get_true_m=false)
    F = dct(diagm(ones(n)), 2)
    sampling = rand(Bernoulli(aimed_m / n), n)
    return get_true_m ? (sum(sampling), F[sampling, :] * sqrt(n / true_m)) : F[sampling, :] * sqrt(n / true_m) # normalize it
end

function fourier(dim)
    dct(diagm(ones(dim)), 2)
end