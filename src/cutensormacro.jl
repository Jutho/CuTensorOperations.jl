macro cutensor(ex::Expr)
    ex = expandconj(ex)
    ex = processcontractorder(ex, nothing)
    intensors, outtensors = extracttensors!(ex)

    tensors = [:($(esc(v)) = $k) for (k,v) in intensors]
    for (k,v) in outtensors
        haskey(intensors, k) || push!(tensors, :($(esc(v)) = $k))
    end
    pre = Expr(:block, tensors...)

    ex = _flatten(tensorify(ex))

    for (k,v) in intensors
        v2 = gensym()
        replace_and_splice!(ex, v, v2, :(CuArray($(esc(v)))))
        if haskey(outtensors, k)
            outtensors[k] = v2
        end
    end
    for (k,v) in outtensors
        if !haskey(intensors, k)
            v2 = gensym()
            replace_and_splice!(ex, v, v2, :(CuArray{eltype($(esc(v)))}(undef, size($(esc(v))))))
            outtensors[k] = v2
        end
    end
    if length(outtensors) > 0
        post = Expr(:block, [:(copyto!($k, $(esc(v)))) for (k,v) in outtensors]...)
        return Expr(:block, pre, ex, post)
    else
        return Expr(:block, pre, ex)
    end
end

# check if a subexpression contains/uses the variable s, but ignore `eltype` calls
function _contains(ex::Expr, s)
    if ex.head == :call && ex.args[1] == :eltype
        return false
    else
        any(e->_contains(e, s), ex.args)
    end
end
_contains(ex::Symbol, s) = ex == s
_contains(ex, s) = false

function _replace!(ex::Expr, s1, s2)
    args = ex.args
    for i = 1:length(args)
        if args[i] isa Symbol && args[i] == s1
            args[i] = s2
        else
            _replace!(args[i], s1, s2)
        end
    end
    return nothing
end
_replace!(ex, s1, s2) = nothing

function replace_and_splice!(ex::Expr, s1, s2, rhs)
    if ex.head == :block
        args = ex.args
        found = false
        i = 1
        while i <= length(args)
            if !found && _contains(args[i], s1)
                insert!(args, i, :($(esc(s2)) = $rhs))
                found = true
            elseif found
                _replace!(args[i], s1, s2)
            end
            i += 1
        end
    elseif !(ex.head == :call && ex.args[1] == :throw)
        @show ex
        error("unexpected expression")
    end
end
