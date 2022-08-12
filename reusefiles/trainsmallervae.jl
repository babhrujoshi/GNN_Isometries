using Flux: Chain, Dense, params, DataLoader
using MLDatasets

include("vaemodel.jl")

gaussinit(out, in) = randn(Float32, out, in) / sqrt(out)


model = FullVae(
    VaeEncoder(Chain(
            Dense(784 => 512, relu, init=gaussinit),
            Dense(512 => 256, relu, init=gaussinit)
        ),
        Dense(256 => 2, init=gaussinit),
        Dense(256 => 2, init=gaussinit)
    ),
    Chain(
        Dense(2 => 256, bias=false, relu, init=gaussinit),
        Dense(256 => 512, bias=false, relu, init=gaussinit),
        Dense(512 => 784, bias=false, sigmoid, init=gaussinit)
    )
)

loss = vaeloss(model, 1.0, 1.0)

traindata = DataLoader(reshape(MNIST(Float32, :train).features, 28^2, :))
validatedata = DataLoader(reshape(MNIST(Float32, :test).features, 28^2, :))


include("trainloops.jl")

trainvalidatelognsave(loss, model, params(model), traindata, validatedata, Flux.Optimise.ADAM(0.0005), 100, "./reusefiles/models/", "./reusefiles/logs/", label="small", validateinterval=5)