
module classifier
    include("classifier/main.jl")
end

module regressor
    include("regressor/main.jl")
end

const REGRESSING = false
const lib = REGRESSING ? regressor : classifier

X, Y = lib.util.loaddata()
X = Float32.(X)
if REGRESSING
    Y = Float32.(Y) # convert things to the right type
else
    Y = UInt32.(Y) # convert things to the right type
end

n_samples, n_features = size(X)
# number of features to
# subselect for each split
# in this case, we are using all available features
max_features = 64

# the number of classes in the label dataset
# TODO: autodetect the number of classes
n_classes = 10



# these follow directly from the 
# meaning of the same keyword arguments
# in scikit learn
max_depth           = typemax(UInt32)
min_samples_leaf    = UInt32(1)
min_samples_split   = UInt32(2)
# this is the same is min impurity decrease
min_purity_increase = Float32(0.0)

# this is not yet supported
# but will be supported soon
max_leaf_nodes      = UInt32(0)

meta = lib.TreeMeta(n_classes, max_features)
stop = lib.StopCondition(
        max_depth, 
        max_leaf_nodes, 
        min_samples_leaf, 
        min_samples_split, 
        min_purity_increase)


# this is a "burn in" run to have everything 
# be compiled first
tree = lib.build_tree(X, Y, meta, stop)
@time lib.build_tree(X, Y, meta, stop)
@time lib.build_tree(X, Y, meta, stop)
@time for i in 1:100
    lib.build_tree(X, Y, meta, stop)
end

function predict(tree, v :: Array{Float32})
    node = tree.root
    while !node.is_leaf
        node = if v[node.feature] <= node.threshold
            node.l
        else
            node.r
        end
    end

    return tree.list[node.label]
end

# for classification
pred = [predict(tree, X[i, 1:n_features]) for i in 1:n_samples]
loss = if REGRESSING
    sum((pred - Y).^2)
else
    1 - sum(pred .== Y) / n_samples
end
println("Loss: ", loss)