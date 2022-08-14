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
secondhidden = 256
zlayer = 16

model = FullVae(
    VaeEncoder(
        Chain(Dense(28^2 => hidden, relu),
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


loss = vaeloss(model, 1.0f0, 0.001f0)
traindata = reshape(MNIST(Float32, :train).features[:, :, 1:end], 28^2, :)
testdata = reshape(MNIST(Float32, :test).features[:, :, 1:end], 28^2, :)
trainloader = DataLoader(traindata, batchsize=32)
validateloader = DataLoader(testdata, batchsize=32)
trainvalidatelognsave(loss, model, params(model), trainloader, validateloader, Flux.Optimise.ADAM(0.001), 20, "./reusefiles/models/", "./reusefiles/logs/", label="tinyv3", loginterval=20, saveinterval=1000)
#end
