using BSON: @load

@load "./reusefiles/savedmodels/vae100epoch" model

using Plots
using Images
using MLDatasets


img = MNIST(Float32, :test).features[:, :, 32]

recodedimg = reshape(model(reshape(img, :)), 28, 28)

plot(
    plot(colorview(Gray, img)), plot(colorview(Gray, recodedimg))
)