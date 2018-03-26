
mutable struct Node
    l           :: Node  # right child
    r           :: Node  # left child
    
    label       :: Float32  # most likely label
    feature     :: UInt32  # feature used for splitting
    threshold   :: Float32 # threshold value
    is_leaf     :: Bool

    depth       :: UInt32
    region      :: UnitRange{UInt32} # a slice of the samples used to decide the split of the node
    features    :: Array{UInt32}     # a list of features not known to be constant

    # added by buid_tree
    purity      :: Float32
    split_at    :: UInt32            # index of samples

    Node() = new()
    Node(features, region, depth) = (
            node = new();
            node.depth = depth;
            node.region = region;
            node.features = features;
            node.is_leaf = false;
            node)
end

struct TreeMeta
    n_classes    :: UInt32 # number of classes to predict
    max_features :: UInt32 # number of features to subselect
end

struct StopCondition
    max_depth           :: UInt32
    max_leaf_nodes      :: UInt32
    min_samples_leaf    :: UInt32
    min_samples_split   :: UInt32
    min_purity_increase :: Float32

    StopCondition(a, b, c, d, e) = new(a, b, c, d, e)
    StopCondition() = new(typemax(UInt32), 0, 1, 2, 0.0)
end

mutable struct FF # float float int lol
    purity  :: Float32
    value   :: Float32
end

mutable struct Tree
    meta :: TreeMeta
    root :: Node
end