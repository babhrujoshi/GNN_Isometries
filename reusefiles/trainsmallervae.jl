using Flux: Chain, Dense, params, DataLoader
using MLDatasets

include("vaemodel.jl")
include("trainloops.jl")

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
# hidden = 512
# secondhidden = convert(Integer, hidden / 2)
# zlayer = 16

# model = FullVae(
#     VaeEncoder(Chain(
#             Dense(28^2 => hidden, relu),
#             Dense(hidden => secondhidden, relu)
#         ),
#         Dense(secondhidden => zlayer),
#         Dense(secondhidden => zlayer)
#     ),
#     Chain(
#         Dense(zlayer => secondhidden, bias=false, relu),
#         Dense(secondhidden => hidden, bias=false, relu),
#         Dense(hidden => 28^2, bias=false, sigmoid)
#     )
# )

hidden = 512
secondhidden = 512
zlayer = 16

model = FullVae(
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


loss = vaeloss(model, 1.0f0, 0.005f0)
traindata = reshape(MNIST(Float32, :train).features[:, :, 1:128], 28^2, :)
testdata = reshape(MNIST(Float32, :test).features[:, :, 1:128], 28^2, :)
trainloader = DataLoader(traindata, batchsize=64)
validateloader = DataLoader(testdata, batchsize=64)
trainvalidatelognsave(loss, model, params(model), trainloader, validateloader, Flux.Optimise.ADAM(0.001), 2, "./reusefiles/models/", "./reusefiles/logs/", label="tinyv5", loginterval=5, saveinterval=50)
#end
