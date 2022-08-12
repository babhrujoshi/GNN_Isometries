using Flux: Chain, Dense, params, DataLoader
using MLDatasets

include("vaemodel.jl")
include("trainloops.jl")

gaussinit(out, in) = randn(Float32, out, in) / sqrt(out)

#function trainmodel()

# model = FullVae(
#     VaeEncoder(Chain(
#             Dense(784 => 512, relu, init=gaussinit),
#             Dense(512 => 256, relu, init=gaussinit)
#         ),
#         Dense(256 => 2, init=gaussinit),
#         Dense(256 => 2, init=gaussinit)
#     ),
#     Chain(
#         Dense(2 => 256, bias=false, relu, init=gaussinit),
#         Dense(256 => 512, bias=false, relu, init=gaussinit),
#         Dense(512 => 784, bias=false, sigmoid, init=gaussinit)
#     )
# )

model = FullVae(
    VaeEncoder(Chain(
            Dense(784 => 128, relu, init=gaussinit),
            Dense(128 => 32, relu, init=gaussinit)
        ),
        Dense(32 => 2, init=gaussinit),
        Dense(32 => 2, init=gaussinit)
    ),
    Chain(
        Dense(2 => 32, bias=false, relu, init=gaussinit),
        Dense(32 => 128, bias=false, relu, init=gaussinit),
        Dense(128 => 784, bias=false, sigmoid, init=gaussinit)
    )
)

loss = vaeloss(model, 1.0f0, 0.01f0)
traindata = reshape(MNIST(Float32, :train).features[:, :, 1:end], 28^2, :)
testdata = reshape(MNIST(Float32, :test).features[:, :, 1:64], 28^2, :)
trainloader = DataLoader(traindata, batchsize=32)
validateloader = DataLoader(testdata, batchsize=32)
trainvalidatelognsave(loss, model, params(model), trainloader, validateloader, Flux.Optimise.ADAM(0.001), 4, "./reusefiles/models/", "./reusefiles/logs/", label="small", validateinterval=1, saveinterval=1)
#end

trainedModel = trainmodel()