using Flux

function optimise!(z::AbstractArray{Float64,1}, opt::Flux.Optimise.AbstractOptimiser, loss::Function; tolerance = 1e-4 ::Float64, out_toggle = 0 ::Integer, max_iter = 10_000 ::Integer)
    tol2 = tolerance^2
    ps = params(z)
    iter=1
    while true
        if iter > max_iter
            @warn "Max num. iterations reached"
            return nothing
        end
        grads = gradient(loss, ps) #loss cannot have any arguments
        update!(opt, ps, grads)
        succ_error = sum(abs2, grads[z])
        if out_toggle != 0 && iter % out_toggle == 0
            println("====> In Gradient: Iteration: $iter grad_size: $(sqrt(succ_error)) tolerance: $tolerance  loss: ", string(loss()))
        end
        if succ_error < tol2
            break
        end
        iter += 1
    end
    return z
end

function recover_signal(x₀::AbstractArray{Float64, 1}, k::Integer, A::Matrix{Float64}, model::Function; opt = Flux.Optimise.ADAM(1e-3)::Flux.Optimise.AbstractOptimiser, kwargs...)
    k̂ = randn(k)/sqrt(k)
    y= A*x₀
    function loss()
        return sum(abs2, A*model(k̂) - y)
    end
    optimise!(k̂,opt, loss; kwargs...)
    return model(k̂)
end