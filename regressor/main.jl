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

@inline function sum_squared(Y, sub)
    yssq = 0.0
    ysum = 0.0
    @simd for i in sub
        y = Y[i]
        yssq += y*y
        ysum += y
    end

    yssq, ysum
end

@inline function nvariance(lssq, lsum, nl)
    return lsum * lsum / nl - lssq
end

function split!(X    :: Array{Float32, 2},
                Y    :: Array{Float32, 1},
                node :: Node,
                meta :: TreeMeta,
                indX :: Array{UInt64, 1},
                stop :: StopCondition,
                ncs  :: Tuple{Array{Float32,1},Array{Float32,1}})
    region = node.region
    # sub = view(indX, region)
    n_samples = length(region)

    tssq, tsum = 0.0, 0.0
    @simd for i in region
        y = Y[indX[i]]
        tssq += y*y
        tsum += y
    end
    # the labels are homogenous
    node.label = tsum / n_samples
    old_purity = (tsum * node.label - tssq) / n_samples
    min_samples_leaf = stop.min_samples_leaf
    if (min_samples_leaf * 2   >  n_samples
     || stop.min_samples_split >  n_samples
     || stop.max_depth         <= node.depth
     || -old_purity < 1e-7)
        node.is_leaf = true
        return
    end

    features = node.features
    n_features = length(features)
    max_features = meta.max_features

    best_purity = -Inf
    best_feature = -1
    threshold_lo = Inf32
    threshold_hi = Inf32

    indf = 1
    n_constant = 0
    Xf, Yf = ncs
    unsplittable = true

    # the number of non constant features we will see if
    # only sample n_features used features
    # is a hypergeometric random variable
    r_start = region.start - 1
    total_features = size(X, 2)
    non_constants_used = hypergeometric(n_features, total_features-n_features, max_features)
    @inbounds while (unsplittable || indf <= non_constants_used) && indf <= n_features
        feature = let
            indr = rand(indf:n_features)
            features[indf], features[indr] = features[indr], features[indf]
            features[indf]
        end

        lssq = lsum = 0.0
        rsum = rssq = 0.0
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
            # honor min_samples_leaf
            if nl >= min_samples_leaf && nr >= min_samples_leaf
                unsplittable = false
                # purity is negative variance
                # should have a minus tssq but well add it later
                # since its a constant
                purity = (rsum / nr * rsum) + (lsum / nl * lsum)

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

            if hi - lo <= n_samples - hi
                @simd for i in lo:hi
                    lab  = Yf[i]
                    lsum += lab
                    lssq += lab*lab
                end

                rsum = tsum - lsum
                rssq = tssq - lssq
            else
                rsum = rssq = 0.0
                @simd for i in (hi+1):n_samples
                    lab  = Yf[i]
                    rsum += lab
                    rssq += lab*lab
                end

                lsum = tsum - rsum
                lssq = tssq - rssq
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
        node.purity = (best_purity - tssq) / n_samples
        if (node.purity - old_purity < stop.min_purity_increase)
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

@inline function fork!(node)
    ind = node.split_at
    region = node.region
    features = node.features
    # no need to copy because we will copy at the end
    node.l = Node(features, region[    1:ind], node.depth + 1)
    node.r = Node(features, region[ind+1:end], node.depth + 1)
end

function check_input(X, Y, meta, stop)
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
                    Y    :: Array{Float32, 1},
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

    ncs = (Array{Float32}(n_samples),
           Array{Float32}(n_samples))

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
