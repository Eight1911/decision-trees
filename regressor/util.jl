module util
    export gini, entropy, q_bi_sort!, depth, num_nodes

    function loaddata(large=false)
        data = if large
            f = open("../mnist_train.csv")
            data = readlines(f)[1:2000]
            [map(parse, split(row, ",")) for row in data]
        else
            f = open("../digits.csv")
            data = readlines(f)[2:end]
            [[parse(Float32, i) 
                for i in split(row, ",")]
                for row in data]
        end

        data = hcat(data...)
        Y = Int.(data[1, 1:end]) + 1
        X = transpose(data[2:end, 1:end])

        return X, Y

    end

    @inline function gini(ns, n)
        s = 0.0
        @simd for k in ns
            s += k * (n - k)
        end
        return s / (n * n)
        # return sum(k * (n - k) for k in ns) / (n * n)
    end


    @inline function entropy(ns, n)
        # =
        s = 0.0
        log_n = log(n)
        @simd for k in ns
            if k > 0
                s += k * log(k)
            end
        end
        return log(n) - s / n
        # =#
        # return -sum(k * log(k/n) for k in ns) / n

    end

    @inline function partition!(v, w, pivot, region)
        i, j = 1, length(region)
        r_start = region.start - 1
        @inbounds while true
            while w[i] <= pivot; i += 1; end;
            while w[j]  > pivot; j -= 1; end;
            i >= j && break
            ri = r_start + i
            rj = r_start + j
            v[ri], v[rj] = v[rj], v[ri]
            w[i], w[j] = w[j], w[i]
            i += 1; j -= 1
        end
        return j
    end

    function insert_sort!(v, w, lo, hi)
        @inbounds for i = lo+1:hi
            j = i
            x = v[i]
            y = w[i]
            while j > lo
                if x < v[j-1]
                    v[j] = v[j-1]
                    w[j] = w[j-1]
                    j -= 1
                    continue
                end
                break
            end
            v[j] = x
            w[j] = y
        end
        return v
    end

    @inline function _selectpivot!(v, w, lo, hi)
        @inbounds begin
            mi = (lo+hi)>>>1

            # sort the values in v[lo], v[mi], v[hi]

            if v[mi] < v[lo]
                v[mi], v[lo] = v[lo], v[mi]
                w[mi], w[lo] = w[lo], w[mi]
            end
            if v[hi] < v[mi]
                if v[hi] < v[lo]
                    v[lo], v[mi], v[hi] = v[hi], v[lo], v[mi]
                    w[lo], w[mi], w[hi] = w[hi], w[lo], w[mi]
                else
                    v[hi], v[mi] = v[mi], v[hi]
                    w[hi], w[mi] = w[mi], w[hi]
                end
            end

            # move v[mi] to v[lo] and use it as the pivot
            v[lo], v[mi] = v[mi], v[lo]
            w[lo], w[mi] = w[mi], w[lo]
            pivot = v[lo]
            w_piv = w[lo]
        end

        # return the pivot
        return pivot, w_piv
    end


    @inline function _bi_partition!(v, w, lo, hi)
        pivot, w_piv = _selectpivot!(v, w, lo, hi)
        # pivot == v[lo], v[hi] > pivot
        i, j = lo, hi
        @inbounds while true
            i += 1; j -= 1
            while v[i] < pivot; i += 1; end;
            while pivot < v[j]; j -= 1; end;
            i >= j && break
            v[i], v[j] = v[j], v[i]
            w[i], w[j] = w[j], w[i]
        end
        v[j], v[lo] = pivot, v[j]
        w[j], w[lo] = w_piv, w[j]

        # v[j] == pivot
        # v[k] >= pivot for k > j
        # v[i] <= pivot for i < j
        return j
    end


    const SMALL_THRESHOLD  = 20
    function q_bi_sort!(v, w, lo, hi)
        @inbounds while lo < hi
            hi-lo <= SMALL_THRESHOLD && return insert_sort!(v, w, lo, hi)
            j = _bi_partition!(v, w, lo, hi)
            if j-lo < hi-j
                # recurse on the smaller chunk
                # this is necessary to preserve O(log(n))
                # stack space in the worst case (rather than O(n))
                lo < (j-1) && q_bi_sort!(v, w, lo, j-1)
                lo = j+1
            else
                j+1 < hi && q_bi_sort!(v, w, j+1, hi)
                hi = j-1
            end
        end
        return v
    end

    function depth(node)
        return node.is_leaf ? 1 : 1 + max(depth(node.l), depth(node.r))
    end

    function num_nodes(node)
        return node.is_leaf ? 1 : 1 + num_nodes(node.l) + num_nodes(node.r)
    end

end
