module TSNE

##############################################################################
##
## Dependencies and Reexports
##
##############################################################################

# None

##############################################################################
##
## Exported methods and types
##
##############################################################################

export tsne

##############################################################################
##
## Source code
##
##############################################################################

"""
Distance matrix D containing the pairwise squared Euclidean distances. Each
element is Dᵢⱼ = ∥xᵢ−xⱼ∥² with the diagonal elements given by the input `diagval`.
"""
function pairwise_sq_euc_dist(X::AbstractMatrix{F}, diagval::F) where {F<:AbstractFloat}
    n = size(X, 2)
    s = sum(abs2, X, 1)                # sᵢ = ||xᵢ||²
    D = BLAS.syrk('L', 'T', F(-2), X)  # Computes lower triangle of D = - 2 .* (X' * X)

    # Complete the full distance matrix: D = - 2 .* (X' * X) .+ s .+ s'
    for j in 1:n

        for i in 1:(j - 1)  # Upper triangle - copy from lower triangle (matrix is symmetric)
            D[i, j] = D[j, i]
        end

        # Diagonal - often 0, but can use for tricks e.g. set to -Inf to underflow to 0 under exp or inv
        D[j, j] = diagval

        sj = s[j]
        for i in (j + 1):n  # Lower triangle
            D[i, j] += (s[i] + sj)
        end
    end

    return D
end

"""
Calculates the entropy of the distribution around point j as well as the conditional
probability of each point i given j (pᵢ∣ⱼ) with distances Dⱼ and parameter βⱼ = 1 / 2σⱼ².
Pⱼ is changed in-place and Hⱼ is returned.
"""
function entropy_and_conditional_p!(Pj::AbstractVector{F}, Dj::AbstractVector{F},
                                    βj::F) where {F<:AbstractFloat}
    sumPj, dotDjPj = F(0), F(0)
    for i in eachindex(Pj)
        sumPj += (Pj[i] = exp(-βj * Dj[i]))
        dotDjPj += Dj[i] * Pj[i]
    end
    Hj = (βj * dotDjPj / sumPj + log(sumPj)) / log(F(2))
    Pj ./= sumPj
    return Hj
end

"""
Reduces each vector by it's minimum value. Has the effect of moving the range of
−βⱼDᵢⱼ (the input into the exponential function) to (−∞,0).
"""
function reducemin!(Dj)
    minDj = minimum(Dj)
    for i in eachindex(Dj)
        Dj[i] -= minDj
    end
end

"""
Performs, for each column (point j), a binary search for the value  βⱼ = 1 / 2σⱼ² and
returns the matrix of conditional probabilities pᵢ∣ⱼ. Each column sums to 1 i.e. ∑ᵢpᵢ∣ⱼ = 1.
"""
function conditional_p(D::AbstractMatrix{F}, perplexity::F, tol::F,
                       max_iter::T) where {T<:Integer, F<:AbstractFloat}

    # Number of observations - columns of X
    n = size(D, 1)

    # Initialise P matrix and β vector
    P = zeros(F, n, n)
    β = zeros(F, n)

    # Initialise column vectors
    Dj = zeros(F, n)
    Pj = zeros(F, n)

    # Binary search target
    target = log2(perplexity)

    # Binary search over each column
    for j in 1:n

        # Re-initialise column vectors
        fill!(Pj, F(0))
        copy!(Dj, view(D, :, j))

        # Subtract minimum value for better stability
        reducemin!(Dj)

        # Initial binary search values
        βj_min = F(0)
        βj_max = F(Inf)
        βj = F(1)

        # Binary search loop
        for iter in 1:max_iter

            # Calculate conditional probabilities and entropy given βj
            Hj = entropy_and_conditional_p!(Pj, Dj, βj)

            # Break from loop if within tolerence
            abs(target - Hj) < tol && break

            # Search branches
            if target < Hj
                βj_min = βj
                βj = isinf(βj_max) ? 2βj : (βj_min + βj_max) / 2
            else
                βj_max = βj
                βj = (βj_min + βj_max) / 2
            end

            # Warn if max iterations reached
            iter == max_iter && warn("Max iterations reached for column $j with perplexity = $(2^Hj)")
        end

        # Save the results
        P[:, j] = Pj
        β[j] = βj
    end

    # Return conditional probabilities
    return P
end

"""
Replaces the conditional P matrix in-place with the joint P matrix
"""
function joint_p!(P::AbstractMatrix{F}) where {F<:AbstractFloat}
    n = size(P, 2)
    for j = 1:n
        for i = (j + 1):n  # Lower triangle
            P[i, j] += P[j, i]
            P[j, i] = P[i, j]
        end
    end
    scale!(P, F(inv(2n)))
end

"""
Create the matrix P of joint probabilities in high dimension space
"""
function create_P(X::AbstractMatrix{F}, perplexity::F, tol::F, max_iter::T) where {T<:Integer, F<:AbstractFloat}
    D = pairwise_sq_euc_dist(X,  prevfloat(F(Inf)))
    P = conditional_p(D, perplexity, tol, max_iter)
    joint_p!(P)
    return P
end

"""
Create the matrix Q of joint probabilities in low dimension space
For computation reasons `joint_q!` doesn't return the full Q matrix but instead returns
the matrix of the numerators (1 + ∥yᵢ−yⱼ∥²)⁻¹ - called Qtop - and separately the
denominator ∑ₖ∑ₗ≠ₖ(1 + ∥yₖ−yₗ∥²)⁻¹ - called sumQ - which is identical for all Qᵢⱼ.
"""
function joint_q!(Qtop::AbstractMatrix{F}, s::AbstractMatrix{F},
                  Y::AbstractMatrix{F}) where {F<:AbstractFloat}
    n = size(Qtop, 2)
    sum!(abs2, s, Y)                            # sᵢ = ||Yᵢ||² (in-place)
    BLAS.syrk!('L', 'T', F(-2), Y, F(0), Qtop)  # Lower triangle of Qtop = - 2 .* (Y' * Y) (in-place)

    sumQ = zero(F)
    @inbounds for j = 1:n
        Qtop[j, j] = F(0)  # Diagonal = 0
        sj = s[j]
        @simd for i = (j + 1):n  # Lower triangle
            Qtop[i, j] += (s[i] + sj)
            sumQ += (Qtop[i, j] = inv(one(F) + Qtop[i, j]))
        end
    end
    sumQ = 2 * sumQ
end

"""
Cost function - KL Divergence
"""
function kl_div(P::AbstractMatrix{F}, Qtop::AbstractMatrix{F}, sumQ::F, exag::F) where {F<:AbstractFloat}
    n = size(P, 2)
    cost = zero(F)
    for j = 1:n
        for i = (j + 1):n  # Lower triangle
            @inbounds cost += P[i, j] * log(P[i, j] / Qtop[i, j])
        end
    end
    return (2 * cost) / exag + log(sumQ / exag)
end

"""
Gradients
"""
function grad!(ΔY::AbstractMatrix{F}, Y::AbstractMatrix{F}, P::AbstractMatrix{F},
               Qtop::AbstractMatrix{F}, sumQ::F)  where {F<:AbstractFloat}
    fill!(ΔY, zero(F))
    invsumQ = inv(sumQ)
    n = size(P, 2)
    for j in 1:n
        for i = (j+1):n  # Lower Triangle
            @inbounds tmp1 = 4 * (P[i, j] - Qtop[i, j] * invsumQ) * Qtop[i, j]
            for d in indices(ΔY, 1)
                tmp2 = tmp1 * (Y[d, j] - Y[d, i])
                ΔY[d, j] += tmp2
                ΔY[d, i] -= tmp2
            end
        end
    end
end

"""
Check the analytic gradients with a numeric approximation
"""
function check_gradients(X, d)

    # Number of observations
    n = size(X, 2)

    # Create P Matrix
    P = create_P(X, 30.0, 1e-5, 50)

    # Initialise Y matrix - sample from Normal(0, 1e-4)
    Y = randn(d, n) * 1e-4

    # Initialise re-use matrices
    s, Qtop, ΔY = zeros(1, n), zeros(n, n), zeros(d, n)

    # Calculate gradients analytically
    sumQ = joint_q!(Qtop, s, Y)
    grad!(ΔY, Y, P, Qtop, sumQ)

    # Size of perturbation
    ϵ = 1e-5

    # Initialise numeric gradient array
    numeric_grad = zeros(ΔY)

    # Estimate gradient for each parameter separately
    for i in eachindex(Y)

        # Store value of parameter
        tmp = Y[i]

        # Forward
        Y[i] += ϵ
        sumQ = joint_q!(Qtop, s, Y)
        cost1 = kl_div(P, Qtop, sumQ, 1.0)

        # Backward
        Y[i] -= 2ϵ
        sumQ = joint_q!(Qtop, s, Y)
        cost2 = kl_div(P, Qtop, sumQ, 1.0)

        # Calculate numeric gradient using central difference
        numeric_grad[i] = (cost1 - cost2) ./ 2ϵ

        # Reset the parameter
        Y[i] = tmp
    end

    # Display results
    # Display the difference between numeric and analytic gradient for each parameter
    @printf "\e[1m%s\e[0m" " Numeric Gradient    Analytic Gradient           Difference\n"
    for i=1:min(15, length(numeric_grad))
        @printf "%17e %20e %20e \n" numeric_grad[i] ΔY[i] numeric_grad[i] - ΔY[i]
    end
    @printf "%17s %20s %20s \n\n" "..." "..." "..."
    @printf "\e[1m%s\e[0m" " Largest differences:\n"
    @printf "%17e %20e \n" extrema(numeric_grad .- ΔY)...
end

"""
Perform PCA to reduce the number of dimensions before applying t-SNE
"""
function pca(X, k)
    Σ = X * X' ./ size(X, 1)         # Covariance natrix
    F = svdfact(Σ)                   # Factorise into Σ = U * diagm(S) * V'
    Xrot = F.U' * X                  # Rotate onto the basis defined by U
    pvar = sum(F.S[1:k]) / sum(F.S)  # Percentage of variance retained with top k vectors
    X̃ = Xrot[1:k, :]                 # Keep top k vectors
    return X̃, pvar
end

"""
Gradient descent update
Uses both momentum and adaptive learning
"""
function update!(Y::AbstractArray{F}, Yμ::AbstractArray{F}, ∇Y::AbstractArray{F},
                 Yg::AbstractArray{F}, μ::F, η::F, min_gain::F) where {F<:AbstractFloat}
    for i in eachindex(Y)

        # Update gains - adaptive learning params
        ∇Y[i] * Yμ[i] > 0 ? Yg[i] = max(Yg[i] * 0.8, min_gain) : Yg[i] += 0.2

        # Momentum gradient descent update with adaptive learning
        Yμ[i] = μ * Yμ[i] - η * Yg[i] * ∇Y[i]
        Y[i] += Yμ[i]
    end
end

"""
`t()` is for displaying the time during the gradient descent optimisation.
"""
t() = Dates.format(now(), "HH:MM:SS")

"""
    tsne(X, d, [perplexity = 30.0, perplexity_tol = 1e-5, perplexity_max_iter = 50,
                pca_init = true, pca_dims = 30, exag = 12.0, stop_exag = 250,
                μ_init = 0.5, μ_final = 0.8, μ_switch = 250,
                η = 100.0, min_gain = 0.01, num_iter = 1000])

### Input types
- `F::AbstractFloat`
- `T::Integer`

### Arguments
- `X::Matrix{F}`: Data matrix - where each row is a feature and each column is an observation / point
- `d::T`: The number of dimensions to reduce X to (e.g. 2 or 3)

### Keyword arguments and default values
- `perplexity::F = 30.0`: User specified perplexity for each conditional distribution
- `perplexity_tol::F = 1e-5`: Tolerence for binary search of bandwidth
- `perplexity_max_iter::T = 50`: Maximum number of iterations used in binary search for bandwidth
- `pca_init::Bool = true`: Choose whether to perform PCA before running t-SNE
- `pca_dims::T = 30`: Number of dimensions to reduce to using PCA before applying t-SNE
- `exag::T = 12`: Early exaggeration - multiply all pᵢⱼ's by this constant
- `stop_exag::T = 250`: Stop the early exaggeration after this many iterations
- `μ_init::F = 0.5`: Initial momentum parameter
- `μ_final::F = 0.8`: Final momentum parameter
- `μ_switch::T = 250`: Switch from initial to final momentum parameter at this iteration
- `η::F = 100.0`: Learning rate
- `min_gain::F = 0.01`: Minimum gain for adaptive learning
- `num_iter::T = 1000`: Number of iterations
- `show_every::T = 100`: Display progress at intervals of this number of iterations
"""
function tsne(X::AbstractMatrix{F}, d::T;
              perplexity::F = 30.0, perplexity_tol::F = 1e-5, perplexity_max_iter::T = 50,
              pca_init::Bool = true, pca_dims::T = 30,
              exag::F = 12.0, stop_exag::T = 250,
              μ_init::F = 0.5, μ_final::F = 0.8, μ_switch::T = 250,
              η::F = 100.0, min_gain::F = 0.01, num_iter::T = 1000,
              show_every::T = 100) where {T<:Integer, F<:AbstractFloat}

    # Number of observations
    n = size(X, 2)

    # Run PCA
    if pca_init
        print(t())
        @printf(" Running PCA ...")
        X, pvar = pca(X, pca_dims)
        @printf(" completed ... percentage of variance retained = %.1f%%\n", pvar*100)
    end

    # Create P Matrix
    print(t())
    @printf(" Computing high dimension joint probabilities ...")
    P = create_P(X, perplexity, perplexity_tol, perplexity_max_iter)
    @printf(" completed\n\n")
    scale!(P, exag)             # early exaggeration

    # Initialise matrices
    Y = randn(F, d, n) * 1e-4   # low dimension map - initialise by sampling from Normal(0, 1e-4)
    s = zeros(F, 1, n)          # sum(abs2, Y, 1)
    Qtop = zeros(F, n, n)       # ||Yⱼ - Yⱼ||² etc
    ∇Y = zeros(F, d, n)         # gradient of Y
    Yμ = zeros(F, d, n)         # velocity of Y - momentum gradient descent
    Yg = ones(F, d, n)          # gains of Y - adaptive learning rate params

    # Initial momentum
    μ = μ_init

    # Gradient descent updates
    println("Running gradient descent updates ...")
    for iter in 1:num_iter

        # Calculate Q probabilities
        sumQ = joint_q!(Qtop, s, Y)

        # Calculate gradients
        grad!(∇Y, Y, P, Qtop, sumQ)

        # Update Y
        update!(Y, Yμ, ∇Y, Yg, μ, η, min_gain)

        # Show progress
        if mod(iter, show_every) == 0
            # print progress
            cost = kl_div(P, Qtop, sumQ, exag)
            print(t())
            @printf(" Iteration %d: cost = %.5f, gradient norm = %.5f\n", iter, cost, norm(∇Y))
        end

        # Change momentum param after μ_switch iterations
        iter == μ_switch && (μ = μ_final)

        # Stop exaggeration
        if iter == stop_exag
            scale!(P, inv(exag))
            exag = F(1)
        end
    end

    # Completed results
    return Y
end

end # module
