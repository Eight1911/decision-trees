# What does presorting even mean ?
# stop conditions are satisfied
    # check stop conditions
    # o maximum depth
    # o minimum of samples for split
    # o output values homogenous
    # o input values homogenous
    # o minimum of samples in each leaf
    # x minimum impurity decrease
# all fields of node are filled
# constants condiion is satisfied

# checks :
#   max_features < n_total_features
#   dimensions

# TODO: implement multiple output trees
# TODO: implement weights on samples
# TODO: add rng seeding

include("util.jl")
include("struct.jl")
include("hypgeo.jl")

# all features given in 'features'
# are not found to be constant
# while space for samples can be saved using
# recursive partitioning, the space of list of
# non-constant features cannot.
function split!(X    :: Array{Float32, 2},
                Y    :: Array{UInt32, 1},
                node :: Node,
                meta :: TreeMeta,
                indX :: Array{UInt64, 1},
                stop :: StopCondition,
                ncs  :: Tuple{Array{UInt32,1},Array{UInt32,1},Array{UInt32,1},Array{Float32,1},Array{UInt32,1}})
    region = node.region
    n_samples = length(region)
    n_classes = meta.n_classes
    r_start = region.start - 1

    # nc = zeros(UInt64, n_classes)
    nc, ncl, ncr, Xf, Yf = ncs
    @simd for lab in 1:n_classes
        @inbounds nc[lab] = 0
    end

    @simd for i in region
        @inbounds nc[Y[indX[i]]] += 1
    end

    node.label = indmax(nc)

    min_samples_leaf = stop.min_samples_leaf
    if (min_samples_leaf * 2   >  n_samples
     || stop.min_samples_split >  n_samples
     || stop.max_depth         <= node.depth
     || n_samples in nc)
        node.is_leaf = true
        return
    end

    # ncl = Array{UInt32}(meta.n_classes)
    # ncr = Array{UInt32}(meta.n_classes)
    features = node.features
    n_features = length(features)
    max_features = meta.max_features
    best_purity = -Inf
    best_feature = -1
    threshold_lo = Inf32
    threshold_hi = Inf32

    indf = 1
    n_constant = 0
    # Xf = Array{Float32}(n_samples)
    # Yf = Array{UInt32}(n_samples)
    unsplittable = true
    r_start = region.start - 1
    # the number of non constant features we will see if
    # only sample n_features used features
    # is a hypergeometric random variable
    total_features = size(X, 2)
    non_constants_used = hypergeometric(n_features, total_features-n_features, max_features)
    @inbounds while (unsplittable || indf <= non_constants_used) && indf <= n_features
        feature = let
            indr = rand(indf:n_features)
            features[indf], features[indr] = features[indr], features[indf]
            features[indf]
        end

        @simd for lab in 1:n_classes
            ncl[lab] = 0
            ncr[lab] = nc[lab]
        end

        @simd for i in 1:n_samples
            sub_i = indX[i + r_start]
            Yf[i] = Y[sub_i]
            Xf[i] = X[sub_i, feature]
        end

        util.q_bi_sort!(Xf, Yf, 1, n_samples)
        nl, nr = 0, n_samples
        lo, hi = 0, 0
        is_constant = true
        while hi < n_samples
            lo = hi + 1
            curr_f = Xf[lo]
            hi = (lo < n_samples && curr_f == Xf[lo+1]
                ? searchsortedlast(Xf, curr_f, lo, n_samples, Base.Order.Forward)
                : lo)

            (nl != 0) && (is_constant = false)
            # honor min_samples_leafs
            if nl >= min_samples_leaf && nr >= min_samples_leaf
                unsplittable = false
                purity = -(nl * util.entropy(ncl, nl)
                         + nr * util.entropy(ncr, nr))
                if purity > best_purity
                    # will take average at the end
                    threshold_lo = last_f
                    threshold_hi = curr_f
                    best_purity = purity
                    best_feature = feature
                end
            end

            let delta = hi - lo + 1
                nl += delta
                nr -= delta
            end

            if hi - lo < n_samples - hi
                @simd for i in lo:hi
                    ncr[Yf[i]] -= 1
                end
                @simd for lab in 1:n_classes
                    ncl[lab] = nc[lab] - ncr[lab]
                end
            else
                @simd for lab in 1:n_classes
                    ncr[lab] = 0
                end
                @simd for i in (hi+1):n_samples
                    ncr[Yf[i]] += 1
                end
                @simd for lab in 1:n_classes
                    ncl[lab] = nc[lab] - ncr[lab]
                end
            end

            last_f = curr_f
        end

        if is_constant
            n_constant += 1
            features[indf], features[n_constant] = features[n_constant], features[indf]
        end

        indf += 1
    end

    # no splits honor min_samples_leaf
    @inbounds if unsplittable
        node.is_leaf = true
        return
    else
        node.purity = best_purity / n_samples
        if (node.purity + util.entropy(nc, n_samples)
            < stop.min_purity_increase)
            node.is_leaf = true
            return
        end
        bf = Int64(best_feature)
        @simd for i in 1:n_samples
            Xf[i] = X[indX[i + r_start], bf]
        end
        node.threshold = (threshold_lo + threshold_hi) / 2.0
        node.split_at = util.partition!(indX, Xf, node.threshold, region)
        node.feature = best_feature
        node.features = features[(n_constant+1):n_features]
    end

end

@inline function fork!(node :: Node)
    ind = node.split_at
    region = node.region
    features = node.features
    # no need to copy because we will copy at the end
    node.l = Node(features, region[    1:ind], node.depth + 1)
    node.r = Node(features, region[ind+1:end], node.depth + 1)
end


# To do: check that Y actually has
# meta.n_classes classes
function check_input(X    :: Array{Float32, 2},
                     Y    :: Array{UInt32, 1},
                     meta :: TreeMeta,
                     stop :: StopCondition)
    n_samples, n_features = size(X)
    if length(Y) != n_samples
        throw("dimension mismatch between X and Y ($(size(X)) vs $(size(Y))")
    elseif n_features < meta.max_features
        throw("number of features $(n_features) ",
              "is less than the number of ",
              "max features $(meta.max_features)")
    elseif stop.min_samples_leaf < 1
        throw("min_samples_leaf must be a positive integer ",
              "(given $(stop.min_samples_leaf))")
    elseif stop.min_samples_split < 2
        throw("min_samples_split must be at least 2 ",
              "(given $(stop.min_samples_split))")
    end
end


function build_tree(X    :: Array{Float32, 2},
                    Y    :: Array{UInt32, 1},
                    meta :: TreeMeta,
                    stop :: StopCondition)
    check_input(X, Y, meta, stop)
    n_samples, n_features = size(X)
    indX = collect(UInt64(1):UInt64(n_samples))
    stack = Node[]

    tree = let
        @inbounds root = Node(collect(1:n_features), 1:n_samples, 1)
        push!(stack, root)
        Tree(meta, root)
    end

    ncs = (Array{UInt32}(meta.n_classes),
           Array{UInt32}(meta.n_classes),
           Array{UInt32}(meta.n_classes),
           Array{Float32}(n_samples),
           Array{UInt32}(n_samples))
    @inbounds while length(stack) > 0
        node = pop!(stack)
        split!(X, Y, node, meta, indX, stop, ncs)
        if !node.is_leaf
            fork!(node)
            push!(stack, node.r)
            push!(stack, node.l)
        end
    end
    return tree
end

