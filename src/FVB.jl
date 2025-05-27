module FVB

using LinearAlgebra, SparseArrays, Random, Statistics,
    LoopVectorization,
    Gases, LXCatParser, AndersonAcceleration,
    Base.Threads


export pt, pts,
    Gas, CrossSection, CollisionFrequency, parseLXCat, mix, samplemix,
    σₘ, σₑ, σₑₓ, σₐₜ, σᵢₒ,
    νₘ, νₑ, νₑₓ, νₐₜ, νᵢₒ,
    MaxwellModel, HardSphereModel, PowerLawAttachment,
    ReidConstant, ReidModel2, ReidRamp,
    LucasSaelee,
    NessRobson,
    SuperModel,
    WhiteMorrisonMason

const σ₀ = 1E-20 # 1Å² reference cross-section in m^2
const m = 5.48579909065E-4 # electron mass in amu
const e = 1.602176634E-19 # elementary charge in C
const mkg = 9.1093837015E-31 # electron mass in kg
const γ = √(2e / mkg) # speed scale in m/s
include("weights.jl")

function __init__()
    BLAS.set_num_threads(1) # Single-threaded BLAS.
    #SparseArrays.UMFPACK.umf_ctrl[8] = 0 # Disable iterative refinement of LU solution. Default = 2. #I believe it is disabled by default now.
end

@fastmath function LinearAlgebra.ldiv!(F::SparseArrays.SPQR.QRSparse{Float64,Int64}, B::Vector{Float64})
    @inbounds begin
        # Apply left permutation
        invpermute!(B, F.rpivinv)

        # Apply F.Q' to B
        for l = 1:length(F.τ)
            h = @view F.factors[:, l]
            axpy!(-F.τ[l] * (h ⋅ B), h, B)
        end

        # Apply F.R⁻¹ to B
        #ldiv!(UpperTriangular(F.R),B)
        aa = F.R.nzval
        ja = F.R.rowval
        ia = F.R.colptr
        for j = length(B):-1:1
            i1 = ia[j]
            i2 = ia[j+1] - one(eltype(ia))

            # find diagonal element
            ii = searchsortedlast(ja, j, i1, i2, Base.Order.Forward)

            # divide with pivot
            bj = B[j] / aa[ii]
            B[j] = bj

            # update remaining part
            for i = ii-1:-1:i1
                B[ja[i]] -= bj * aa[i]
            end
        end

        # Apply right permutation
        invpermute!(B, F.cpiv)
    end
end

function LinearAlgebra.:\(F::SparseArrays.SPQR.QRSparse{Float64,Int64}, B::Vector{Float64})
    X = copy(B)
    ldiv!(F, X)
end

function oracle(gas::Gas; E, T, lmax)
    μE, σE = 0.0, 4.0
    μT, σT = 2.0, 2.0
    μmass, σmass = 1.0, 2.0
    μt, σt = -1.7958800173440752, 4.204119982655925
    μω, σω = -2.820343287958481, 5.00637725113053
    μσ, σσ = 0.4961060916886306, 5.496106091688631
    με, σε = 2.5, 5.5
    μl, σl = 4.0, 3.0

    scaleE(E) = @. (log10(clamp(E, 1E-4, 1E4)) - μE) / σE
    scaleT(T) = @. (log10(clamp(T, 1E0, 1E4)) - μT) / σT
    scalemass(mass) = @. (log10(clamp(mass, 1E0, 1E3)) - μmass) / σmass
    scalet(t) = @. (log10(clamp(t, 1E-6, 256.0)) - μt) / σt
    scaleσ(σ) = @. (log10(clamp(σ, 1E-5, 1E7)) - μσ) / σσ
    scalel(l) = @. (clamp(l, 1.0, 7.0) - μl) / σl
    unscaleω(x) = @. clamp(exp10(μω + σω * x), sqrt(eps()), 200.0)
    unscaleε(x) = @. clamp(1.4090169943749475exp10(με + σε * x), 1E-3, 1E8)

    εdata = 10 .^ range(-4, 4, length=150)

    A = Vector{Float32}(undef, 619)
    A[1] = E |> scaleE
    A[2] = T |> scaleT
    A[3] = lmax |> scalel
    A[4] = sum(w * m for (w, m) in zip(gas.wₑₗ, gas.Mₑₗ)) |> scalemass
    exlosses = Tuple(t for t in gas.exlosses if !isinf(t) && !isnan(t))
    n = length(exlosses)
    A[5] = (n == 0 ? Inf : minimum(exlosses)) |> scalet
    A[6] = (n == 0 ? Inf : median(exlosses)) |> scalet
    A[7] = (n == 0 ? Inf : mean(exlosses)) |> scalet
    A[8] = (n == 0 ? Inf : prod(exp(log(t) / n) for t in exlosses)) |> scalet
    A[9] = (n == 0 ? Inf : n / sum(1 / t for t in exlosses)) |> scalet
    A[10] = (n == 0 ? Inf : sqrt(sum(t^2 for t in exlosses) / n)) |> scalet
    A[11] = (n == 0 ? Inf : maximum(exlosses)) |> scalet
    iolosses = Tuple(t for t in gas.iolosses if !isinf(t) && !isnan(t))
    n = length(iolosses)
    A[12] = (n == 0 ? Inf : minimum(iolosses)) |> scalet
    A[13] = (n == 0 ? Inf : median(iolosses)) |> scalet
    A[14] = (n == 0 ? Inf : mean(iolosses)) |> scalet
    A[15] = (n == 0 ? Inf : prod(exp(log(t) / n) for t in iolosses)) |> scalet
    A[16] = (n == 0 ? Inf : n / sum(1 / t for t in iolosses)) |> scalet
    A[17] = (n == 0 ? Inf : sqrt(sum(t^2 for t in iolosses) / n)) |> scalet
    A[18] = (n == 0 ? Inf : maximum(iolosses)) |> scalet
    A[19] = gas.atthresh |> scalet
    for i = 1:150
        A[19+i] = σₘ(gas, εdata[i]) |> scaleσ
    end
    for i = 1:150
        A[19+150+i] = σₑₓ(gas, εdata[i]) |> scaleσ
    end
    for i = 1:150
        A[19+150+150+i] = σₐₜ(gas, εdata[i]) |> scaleσ
    end
    for i = 1:150
        A[19+150+150+150+i] = σᵢₒ(gas, εdata[i]) |> scaleσ
    end

    softplus(x) = ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
    mish(x) = x * tanh(softplus(x))

    B = W₆ * mish.(W₅ * mish.(W₄ * mish.(W₃ * mish.(W₂ * mish.(W₁ * A .+ b₁) .+ b₂) .+ b₃) .+ b₄) .+ b₅) .+ b₆

    return unscaleε(B[1]), unscaleω(B[2])
end

function pt(gas::Gas; T, E, εmax=NaN, μ=NaN, lmax=1, N=200, superelastic=false, supertype=:tseparate, verbose=false)
    @inbounds begin

        ex_types = (:ROTATION, :VIBRATION, :EXCITATION, :DISSOCIATION)
        (iseven(lmax) || lmax < 1) && throw("lmax must be odd.")

        ifactor = 1 + (gas.species === :electron)

        εguess::Float64, ω₀guess::Float64 = oracle(gas; E, T, lmax)
        εguess *= 10
        adjust_εmax = isnan(εmax)
        εmax::Float64 = adjust_εmax ? εguess : εmax

        T_og = T
        TK, ETd = max(sqrt(eps()), T), max(sqrt(eps()), E)
        T = 1380649TK / 16021766340 # T *= kB/e
        E = ETd / 10 # E *= 1E-21/σ₀
        if superelastic
            Z_total = Z_T(gas.exlosses, T_og, gas.ex_nos)
        end

        attempts = 0
        @label start

        attempts += 1
        #attempts > 100 && throw("Convergence failed.")
        #attempts > 100 && throw("Convergence failed.")

        μ = isnan(μ) ? εmax > gas.iothresh ? ω₀guess : sqrt(eps()) : μ

        verbose && println("Solving with εmax = $εmax eV")

        vmax = sqrt(εmax)
        Δv = vmax / N
        v = range(Δv / 2, vmax - Δv / 2, length=N)
        vb = range(Δv, vmax, length=N)
        ε = v .^ 2

        # Initialise rate terms?
        νₗₒₛₛ = Vector{Float64}(undef, N)
        νₜₒₜ = Vector{Float64}(undef, N)
        νₙₑₜ = Vector{Float64}(undef, N)
        νₗₒₛₛb = Vector{Float64}(undef, N)
        νₜₒₜb = Vector{Float64}(undef, N)
        # TODO: What are W/A/B?
        Wb = Vector{Float64}(undef, N)
        Ab = Vector{Float64}(undef, N)
        Bb = Vector{Float64}(undef, N)
        if superelastic
            for i = 1:N
                nuat, nuio = νₐₜ(gas, v[i]), νᵢₒ(gas, v[i])
                νₗₒₛₛ[i] = νₑₓ(gas, v[i], T_og, supertype) + νₛᵤ(gas, v[i], T_og, supertype) + nuat + nuio
                νₜₒₜ[i] = νₘ(gas, v[i]) + νₗₒₛₛ[i]
                νₙₑₜ[i] = nuio - nuat
                νₗₒₛₛb[i] = νₑₓ(gas, vb[i], T_og, supertype) + νₛᵤ(gas, vb[i], T_og, supertype) + νₐₜ(gas, vb[i]) + νᵢₒ(gas, vb[i])
                νₜₒₜb[i] = νₘ(gas, vb[i]) + νₗₒₛₛb[i] # momentum + loss terms (i.e. excitation, ionisation, attachment)
                z = 2(1 / vb[i] - vb[i] / T)Δv
                Wb[i] = z * νₑ(gas, vb[i]) * T / 4Δv # vₑ is the 'not' momentum transfer elastic cross-section
                Ab[i] = -Wb[i] / expm1(-z) # expm1 is exp(x-1)
                Bb[i] = -Wb[i] / expm1(+z)
            end
        else
            for i = 1:N
                nuat, nuio = νₐₜ(gas, v[i]), νᵢₒ(gas, v[i])
                νₗₒₛₛ[i] = νₑₓ(gas, v[i]) + nuat + nuio
                νₜₒₜ[i] = ν̃(gas, v[i], 2) + νₗₒₛₛ[i]
                νₙₑₜ[i] = nuio - nuat
                νₗₒₛₛb[i] = νₑₓ(gas, vb[i]) + νₐₜ(gas, vb[i]) + νᵢₒ(gas, vb[i])
                νₜₒₜb[i] = ν̃(gas, vb[i], 2) + νₗₒₛₛb[i] # use the second column here
                z = 2(1 / vb[i] - vb[i] / T)Δv
                Wb[i] = z * νₑ(gas, vb[i], 1) * T / 4Δv
                Ab[i] = -Wb[i] / expm1(-z)
                Bb[i] = -Wb[i] / expm1(+z)
            end
        end
        #νₑₓ(gas,v[end],T_og,supertype) + νₛᵤ(gas,v[end],T_og,supertype)
        #= νₜₒₜv = νₜₒₜ #gas.elastic === gas.viscosity ? νₜₒₜ : @. νᵥ(gas,v) + νₗₒₛₛ =#
        #= νₜₒₜvb = νₜₒₜb #gas.elastic === gas.viscosity ? νₜₒₜb : @. νᵥ(gas,vb) + νₗₒₛₛb =#
        #= νₜₒₜv = @. ν̃(gas, v, 1) + νₗₒₛₛ #gas.elastic === gas.viscosity ? νₜₒₜ : @. νᵥ(gas,v) + νₗₒₛₛ =#
        #= νₜₒₜvb = @. ν̃(gas, vb, 1) + νₗₒₛₛb #gas.elastic === gas.viscosity ? νₜₒₜb : @. νᵥ(gas,vb) + νₗₒₛₛb =#


        siz = (3 + 5 + 10(lmax ÷ 2))N #TODO: For each grid point, we have 3 (), 5 () and 10 () items...
        for exloss in gas.exlosses
            if exloss < εmax
                siz += 2N
            end
        end
        for ioloss in gas.iolosses
            if ioloss < εmax
                siz += (1 + N ÷ 2)N
            end
        end
        if μ != 0
            siz += (lmax + 1)N
        end

        if superelastic
            for exloss in gas.exlosses
                if exloss < εmax
                    siz += 2N
                end
            end
        end

        is = Vector{Int}(undef, siz) #TODO: is and js?
        js = Vector{Int}(undef, siz)
        Ls = Vector{Float64}(undef, siz) #TODO: ls ?

        c = 0
        c += 1
        is[c] = 1
        js[c] = 1
        Ls[c] = -(νₗₒₛₛ[1] + Ab[1] / Δv)
        c += 1
        is[c] = 1
        js[c] = 2
        Ls[c] = -Bb[1] / Δv
        for i = 2:N-1
            c += 1
            is[c] = i
            js[c] = i - 1
            Ls[c] = Ab[i-1] / Δv
            c += 1
            is[c] = i
            js[c] = i
            Ls[c] = -(νₗₒₛₛ[i] + (Ab[i] - Bb[i-1]) / Δv)
            c += 1
            is[c] = i
            js[c] = i + 1
            Ls[c] = -Bb[i] / Δv
        end
        c += 1
        is[c] = N
        js[c] = N - 1
        Ls[c] = Ab[N-1] / Δv
        c += 1
        is[c] = N
        js[c] = N
        Ls[c] = -(νₗₒₛₛ[N] - Bb[N-1] / Δv)

        # TODO: Functional gas cross-sections don't have the type attribute

        if isempty(gas.excitation)
            no_gases = 1
        else
            no_gases = maximum(gas.ex_nos)
        end
        Z_trv = ones(no_gases) # Total partition function of the rotation/vibration states for the bolsig method
        if supertype === :tseparatebolsig
            for j in 1:no_gases
                for k in eachindex(gas.excitation)
                    if gas.excitation[k].type === :ROTATION && gas.ex_nos[k] == j
                        #for k in eachindex(gas.excitation) if gas.excitation[k].type ∈ [:VIBRATION, :ROTATION]
                        Z_trv[j] += Z(gas.exlosses[k], T_og)
                    end
                end
            end
        end

        # Excitations
        #@show "Excitation processes"
        Z_robust = ones(no_gases, length(gas.excitation))
        # For each process of the same type and lower threshold, add to the population
        Z_robust_super = ones(no_gases, length(gas.excitation))
        # iterate over gas numbers
        for ng in 1:no_gases
            for t in ex_types
                Z_t = 1.0
                for k in eachindex(gas.excitation)
                    if gas.excitation[k].type === t && gas.ex_nos[k] == ng
                        Z_t += Z(gas.exlosses[k], T_og)
                        for ki in eachindex(gas.excitation)
                            if ki != k && gas.excitation[ki].type === t && gas.ex_nos[ki] == ng
                                # if cs is the same type and lower threshold, add to partition
                                if gas.excitation[ki].energy_loss < gas.excitation[k].energy_loss
                                    Z_robust[ng, k] += Z(gas.exlosses[ki], T_og)
                                    Z_robust_super[ng, k] += 1
                                end
                            end
                        end
                    end
                end
                #@show t

                for k in eachindex(gas.excitation)
                    if gas.excitation[k].type === t && gas.ex_nos[k] == ng
                        νₑₓ, wₑₓ, εₖ = CollisionFrequency(gas.excitation[k]), gas.wₑₓ[k], gas.exlosses[k]
                        for i = 1:N
                            εshift = ε[i] + εₖ
                            εshift >= εmax && break
                            vshift = Base.sqrt_llvm(εshift)
                            if superelastic
                                if gas.custom_population
                                    #i== 1 && @show gas.excitation[k].population
                                    prefactor = wₑₓ * v[i] / vshift * νₑₓ(vshift) * gas.excitation[k].population
                                else
                                    if supertype === :separate
                                        # Each process is separate
                                        prefactor = wₑₓ * v[i] / vshift * νₑₓ(vshift) / (1 + Z(εₖ, T_og))
                                        #i == 1 && @show 1/(1+Z(εₖ,T_og))

                                    elseif supertype === :tseparate
                                        # Each process type is separate
                                        prefactor = wₑₓ * v[i] / vshift * νₑₓ(vshift) / Z_t
                                        #i == 1 && @show 1/Z_t

                                    elseif supertype === :tseparatebolsig
                                        # Each process type is separate in a bolsig way
                                        #if t ∈ [:VIBRATION, :ROTATION]
                                        if t === :ROTATION
                                            prefactor = wₑₓ * v[i] / vshift * νₑₓ(vshift) / Z_trv[ng]
                                            #i == 1 && @show 1/Z_trv
                                        else
                                            prefactor = wₑₓ * v[i] / vshift * νₑₓ(vshift) / (1 + Z(εₖ, T_og))
                                            #i == 1 && @show 1/(1+Z(εₖ,T_og))
                                        end

                                    elseif supertype === :together
                                        # Each process is together
                                        prefactor = wₑₓ * v[i] / vshift * νₑₓ(vshift) / Z_total[ng]
                                    elseif supertype === :robust
                                        # Each process type is separate + pure state approximation
                                        prefactor = Z_robust[ng, k] * wₑₓ * v[i] / vshift * νₑₓ(vshift) / Z_t
                                        #i == 1 && @show Z_robust[k]/Z_t
                                    end
                                end
                            else
                                prefactor = wₑₓ * v[i] / vshift * νₑₓ(vshift)
                            end
                            j = floor(Int, (vshift + Δv / 2) / (vmax + Δv) * (N + 1)) #searchsortedlast(v,vshift)
                            w = (vshift - v[j]) / Δv
                            c += 1
                            is[c] = i
                            js[c] = j
                            Ls[c] = prefactor * (1 - w)
                            j == N && break
                            c += 1
                            is[c] = i
                            js[c] = j + 1
                            Ls[c] = prefactor * w
                        end
                    end
                end
            end
        end
        # Superelastics
        #@show "superelastic processes"
        if superelastic
            for ng in 1:no_gases
                for t in ex_types
                    Z_t = 1.0
                    for k in eachindex(gas.excitation)
                        if gas.excitation[k].type === t && gas.ex_nos[k] == ng
                            Z_t += Z(gas.exlosses[k], T_og)
                        end
                    end
                    #@show t
                    #@show Z_t

                    for k in eachindex(gas.excitation)
                        if gas.excitation[k].type === t && gas.ex_nos[k] == ng
                            νₑₓ, wₑₓ, εₖ = CollisionFrequencySuper(gas.excitation[k]), gas.wₑₓ[k], gas.exlosses[k]
                            if gas.custom_population
                                Z_k = gas.excitation[k].super_population
                            else
                                if supertype === :separate
                                    # Each process is separate
                                    Z_k = Z(εₖ, T_og) / (1 + Z(εₖ, T_og))
                                    #@show Z_k

                                elseif supertype === :tseparate
                                    # Each process type is separate
                                    Z_k = Z(εₖ, T_og) / Z_t
                                    #@show Z_k

                                elseif supertype === :tseparatebolsig
                                    # Each process type is separate
                                    # 
                                    #if t ∈ [:VIBRATION, :ROTATION]
                                    if t === :ROTATION
                                        Z_k = Z(εₖ, T_og) / Z_trv[ng]
                                    else
                                        Z_k = Z(εₖ, T_og) / (1 + Z(εₖ, T_og))
                                    end
                                    #@show Z_k
                                elseif supertype === :together
                                    # Each process is together
                                    Z_k = Z(εₖ, T_og) / Z_total[ng]
                                elseif supertype === :robust
                                    # Each process is together
                                    Z_k = Z_robust_super[ng, k] * Z(εₖ, T_og) / Z_t
                                    #Z_k = Z_prefactor_robust_super[k]/Z_total_robust
                                    #@show Z_k
                                end
                            end
                            for i = 1:N
                                εshift = ε[i] - εₖ
                                εshift <= 0 && continue
                                vshift = Base.sqrt_llvm(εshift)
                                prefactor = wₑₓ * v[i] / vshift * Z_k * νₑₓ(vshift, εₖ)
                                j = floor(Int, (vshift + Δv / 2) / (vmax + Δv) * (N + 1)) #searchsortedlast(v,vshift)
                                w = (vshift - v[j]) / Δv
                                #c += 1; is[c] = i; js[c] = j; Ls[c] = prefactor*(1-w)
                                #j == N && break
                                #c += 1; is[c] = i; js[c] = j+1; Ls[c] = prefactor*w
                                c += 1
                                is[c] = i
                                js[c] = j + 1
                                Ls[c] = prefactor * w
                                j == 0 && continue
                                c += 1
                                is[c] = i
                                js[c] = j
                                Ls[c] = prefactor * (1 - w)
                            end
                        end
                    end
                end
            end
        end

        #if superelastic
        #for k in eachindex(gas.excitation)
        #νₑₓ, wₑₓ, εₖ = CollisionFrequencySuper(gas.excitation[k]), gas.wₑₓ[k], gas.exlosses[k]
        #for i = 1:N
        #εshift = ε[i] - εₖ
        #εshift <= 0 && continue
        #vshift = Base.sqrt_llvm(εshift)
        #prefactor = wₑₓ*ε[i]/εshift*Z_k*νₑₓ(v[i],εₖ)/Z_total
        #j = floor(Int,(vshift+Δv/2)/(vmax+Δv)*(N+1)) #searchsortedlast(v,vshift)
        #w = (vshift-v[j])/Δv
        #c += 1; is[c] = i; js[c] = j+1; Ls[c] = prefactor*w
        #j == 0 && continue
        #c += 1; is[c] = i; js[c] = j; Ls[c] = prefactor*(1-w)
        #end
        #end
        #end

        #All fractions equiprobable energy sharing
        for k in eachindex(gas.ionisation)
            νᵢₒ, wᵢₒ, εₖ = CollisionFrequency(gas.ionisation[k]), gas.wᵢₒ[k], gas.iolosses[k]
            for i = 1:N
                εshift = ε[i] + εₖ
                εshift >= εmax && break
                vshift = Base.sqrt_llvm(εshift)
                prefactor = ifactor * 2v[i] * wᵢₒ * Δv
                j = floor(Int, (vshift + Δv / 2) / (vmax + Δv) * (N + 1)) #searchsortedlast(v,vshift)
                w = 1 / 2 + (v[j] - vshift) / Δv
                h = w > 0 ? j : j + 1
                c += 1
                is[c] = i
                js[c] = h
                Ls[c] = w * prefactor * νᵢₒ(v[h]) / (ε[h] - εₖ)
                for h = j+1:N
                    c += 1
                    is[c] = i
                    js[c] = h
                    Ls[c] = prefactor * νᵢₒ(v[h]) / (ε[h] - εₖ)
                end
            end
        end


        # Equal energy sharing
        #for k in eachindex(gas.ionisation)
        #νᵢₒ, wᵢₒ, εₖ = CollisionFrequency(gas.ionisation[k]), gas.wᵢₒ[k], gas.iolosses[k]
        #for i = 1:N
        #εshift = 2ε[i] + εₖ
        #εshift >= εmax && break
        #vshift = Base.sqrt_llvm(εshift)
        #prefactor = ifactor*wᵢₒ*v[i]/vshift*νᵢₒ(vshift)
        #j = floor(Int,(vshift+Δv/2)/(vmax+Δv)*(N+1)) #searchsortedlast(v,vshift)
        #w = (vshift-v[j])/Δv
        #c += 1; is[c] = i; js[c] = j; Ls[c] = prefactor*(1-w)
        #j == N && break
        #c += 1; is[c] = i; js[c] = j+1; Ls[c] = prefactor*w
        #end
        #end


        # L01
        l = 0
        c += 1
        is[c] = l * N + 1
        js[c] = (l + 1) * N + 1
        Ls[c] = +E * (l + 1) / 2(2l + 3) * (1 / Δv)
        for i = 2:N-1
            c += 1
            is[c] = l * N + i
            js[c] = (l + 1) * N + i - 1
            Ls[c] = +E * (l + 1) / 2(2l + 3) * (l / 2v[i] - 1 / Δv)
            c += 1
            is[c] = l * N + i
            js[c] = (l + 1) * N + i
            Ls[c] = +E * (l + 1) / 2(2l + 3) * (l / 2v[i] + 1 / Δv)
        end
        c += 1
        is[c] = l * N + N
        js[c] = (l + 1) * N + N - 1
        Ls[c] = +E * (l + 1) / 2(2l + 3) * (-1 / Δv)

        # L10
        l = 1
        c += 1
        is[c] = l * N + 1
        js[c] = (l - 1) * N + 1
        Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2vb[1])
        c += 1
        is[c] = l * N + 1
        js[c] = (l - 1) * N + 1 + 1
        Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2vb[1] - 1 / Δv)
        for i = 2:N-1
            c += 1
            is[c] = l * N + i
            js[c] = (l - 1) * N + i
            Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2vb[i] + 1 / Δv)
            c += 1
            is[c] = l * N + i
            js[c] = (l - 1) * N + i + 1
            Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2vb[i] - 1 / Δv)
        end
        c += 1
        is[c] = l * N + N
        js[c] = (l - 1) * N + N
        Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2vb[N] + 1 / Δv)

        # L11
        for i = 1:N
            c += 1
            is[c] = l * N + i
            js[c] = l * N + i
            Ls[c] = -νₜₒₜb[i]
        end

        # Multiterms
        for l = 1:2:lmax-2
            c += 1
            is[c] = l * N + 1
            js[c] = (l + 1) * N + 1
            Ls[c] = +E * (l + 1) / 2(2l + 3) * (l / 2vb[1])
            c += 1
            is[c] = l * N + 1
            js[c] = (l + 1) * N + 1 + 1
            Ls[c] = +E * (l + 1) / 2(2l + 3) * (l / 2vb[1] + 1 / Δv)
            for i = 2:N-1
                c += 1
                is[c] = l * N + i
                js[c] = (l + 1) * N + i
                Ls[c] = +E * (l + 1) / 2(2l + 3) * (l / 2vb[i] - 1 / Δv)
                c += 1
                is[c] = l * N + i
                js[c] = (l + 1) * N + i + 1
                Ls[c] = +E * (l + 1) / 2(2l + 3) * (l / 2vb[i] + 1 / Δv)
            end
            c += 1
            is[c] = l * N + N
            js[c] = (l + 1) * N + N
            Ls[c] = +E * (l + 1) / 2(2l + 3) * (l / 2vb[N] - 1 / Δv)

            #? Even l
            l += 1
            c += 1
            is[c] = l * N + 1
            js[c] = (l - 1) * N + 1
            Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2v[1] - 1 / Δv)
            for i = 2:N-1
                c += 1
                is[c] = l * N + i
                js[c] = (l - 1) * N + i - 1
                Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2v[i] + 1 / Δv)
                c += 1
                is[c] = l * N + i
                js[c] = (l - 1) * N + i
                Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2v[i] - 1 / Δv)
            end
            c += 1
            is[c] = l * N + N
            js[c] = (l - 1) * N + N - 1
            Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2v[N] + 1 / Δv)
            c += 1
            is[c] = l * N + N
            js[c] = (l - 1) * N + N
            Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2v[N])
            for i = 1:N
                c += 1
                is[c] = l * N + i
                js[c] = l * N + i
                Ls[c] = -ν̃(gas, v[i], l + 1) - νₗₒₛₛ[i] # start at the third
            end
            c += 1
            is[c] = l * N + 1
            js[c] = (l + 1) * N + 1
            Ls[c] = +E * (l + 1) / 2(2l + 3) * (l / 2v[1] + 1 / Δv)
            for i = 2:N-1
                c += 1
                is[c] = l * N + i
                js[c] = (l + 1) * N + i - 1
                Ls[c] = +E * (l + 1) / 2(2l + 3) * (l / 2v[i] - 1 / Δv)
                c += 1
                is[c] = l * N + i
                js[c] = (l + 1) * N + i
                Ls[c] = +E * (l + 1) / 2(2l + 3) * (l / 2v[i] + 1 / Δv)
            end
            c += 1
            is[c] = l * N + N
            js[c] = (l + 1) * N + N - 1
            Ls[c] = +E * (l + 1) / 2(2l + 3) * (l / 2v[N] - 1 / Δv)
            c += 1
            is[c] = l * N + N
            js[c] = (l + 1) * N + N
            Ls[c] = +E * (l + 1) / 2(2l + 3) * (l / 2v[N])

            #? Odd l
            l += 1
            c += 1
            is[c] = l * N + 1
            js[c] = (l - 1) * N + 1
            Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2vb[1])
            c += 1
            is[c] = l * N + 1
            js[c] = (l - 1) * N + 1 + 1
            Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2vb[1] - 1 / Δv)
            for i = 2:N-1
                c += 1
                is[c] = l * N + i
                js[c] = (l - 1) * N + i
                Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2vb[i] + 1 / Δv)
                c += 1
                is[c] = l * N + i
                js[c] = (l - 1) * N + i + 1
                Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2vb[i] - 1 / Δv)
            end
            c += 1
            is[c] = l * N + N
            js[c] = (l - 1) * N + N
            Ls[c] = -E * l / 2(2l - 1) * ((l + 1) / 2vb[N] + 1 / Δv)
            for i = 1:N
                c += 1
                is[c] = l * N + i
                js[c] = l * N + i
                Ls[c] = -ν̃(gas, vb[i], l + 1) - νₗₒₛₛb[i]
            end
        end

        if μ != 0
            for i = 1:(lmax+1)N
                c += 1
                is[c] = i
                js[c] = i
                Ls[c] = -μ
            end
        end

        L = sparse(@view(is[1:c]), @view(js[1:c]), @view(Ls[1:c]), (lmax + 1)N, (lmax + 1)N)
        dropzeros!(L)

        F = qr(L)

        G = Vector{Float64}(undef, (lmax + 1)N)
        randexp!(@view G[1:N])
        randn!(@view G[N+1:end])
        G ./= Δv * sum(G[i] for i = 1:N)

        #    return L, F, G

        ω₀ₚₚ = ω₀ₚ = ω₀ = NaN
        ε̄ₚₚ = ε̄ₚ = ε̄ = NaN
        ω₁Fₚₚ = ω₁Fₚ = ω₁F = NaN
        σ²ω₀ₚ = σ²ω₀ = NaN
        σ²ε̄ₚ = σ²ε̄ = NaN
        σ²ω₁Fₚ = σ²ω₁F = NaN
        iterations = 0
        a = Anderson(G, m=3)
        converged = true
        while true
            iterations += 1
            #iterations > 10000 && throw("Convergence failed.")
            if iterations > 10000
                converged = false
                break
            end

            ldiv!(F, G)
            G ./= Δv * sum(G[i] for i = 1:N)
            accelerate!(a, G)
            G ./= Δv * sum(G[i] for i = 1:N)

            ω₀ₚₚ, ω₀ₚ, ω₀ = ω₀ₚ, ω₀, Δv * sum(νₙₑₜ[i] * G[i] for i = 1:N)
            ε̄ₚₚ, ε̄ₚ, ε̄ = ε̄ₚ, ε̄, Δv * sum(ε[i] * G[i] for i = 1:N)
            ω₁Fₚₚ, ω₁Fₚ, ω₁F = ω₁Fₚ, ω₁F, Δv * sum(vb[i] / 3 * G[N+i] for i = 1:N)
            verbose && @show (ω₀, ε̄, ω₁F)
            if isapprox(ε̄, ε̄ₚ, atol=10eps(), rtol=1E-6) && isapprox(ω₁F, ω₁Fₚ, atol=10eps(), rtol=1E-6) && isapprox(ω₀, ω₀ₚ, atol=10eps(), rtol=1E-6)
                if adjust_εmax
                    Gmax, imax = findmax(@view(G[1:N]))
                    Ggoal = 1E-10 * Gmax
                    igoal = findfirst(<=(Ggoal), @view(G[imax:N]))
                    if isnothing(igoal) # εmax is too small, extrapolate to larger
                        iend = imax - 1 + findlast(>(0), @view(G[imax:N]))
                        ihalf = (imax + iend) ÷ 2
                        ithreeq = (imax + 3iend) ÷ 4
                        v1, v2 = v[ihalf], v[ithreeq]
                        lG1, lG2 = log(G[ihalf]), log(G[ithreeq])
                        εproposed = clamp(1.25((lG1 * v2 - lG2 * v1 + (v1 - v2) * log(Ggoal)) / (lG1 - lG2))^2, 1.1εmax, 10εmax)
                        εmax = isnan(εproposed) ? 2εmax : εproposed

                        μ = max(μ, sqrt(eps()), ω₀)
                        attempts < 100 && @goto start
                    else
                        εgoal = ε[imax-1+igoal]
                        if εgoal < εmax / 2 # εmax is too large, truncate to smaller
                            εmax = 1.1εgoal

                            μ = max(μ, sqrt(eps()), ω₀)
                            attempts < 100 && @goto start
                        end
                    end
                end

                break
            end

            σ²ω₀ₚ, σ²ω₀ = σ²ω₀, (-2ω₀ₚₚ + ω₀ₚ + ω₀)^2 + (ω₀ₚₚ - 2ω₀ₚ + ω₀)^2 + (ω₀ₚₚ + ω₀ₚ - 2ω₀)^2
            σ²ε̄ₚ, σ²ε̄ = σ²ε̄, (-2ε̄ₚₚ + ε̄ₚ + ε̄)^2 + (ε̄ₚₚ - 2ε̄ₚ + ε̄)^2 + (ε̄ₚₚ + ε̄ₚ - 2ε̄)^2
            σ²ω₁Fₚ, σ²ω₁F = σ²ω₁F, (-2ω₁Fₚₚ + ω₁Fₚ + ω₁F)^2 + (ω₁Fₚₚ - 2ω₁Fₚ + ω₁F)^2 + (ω₁Fₚₚ + ω₁Fₚ - 2ω₁F)^2
            if (σ²ω₀ >= σ²ω₀ₚ && σ²ε̄ >= σ²ε̄ && σ²ω₁F >= σ²ω₁Fₚ) || (iterations >= 20 && iterations % 2 == 0)
                restart!(a)
            end
        end

        verbose && @show iterations
        WF = γ * ω₁F

        verbose && println("Solving HOH equations...")

        Lᴴᴼᴴ = L
        if μ != ω₀
            for i = 1:(lmax+1)N
                Lᴴᴼᴴ[i, i] += μ - ω₀
            end
        end
        for j = 1:N
            Lᴴᴼᴴ[N, j] = Δv
        end
        Lᴴᴼᴴ[N, 2N-1] = 0
        Lᴴᴼᴴ[N, 2N] = 0
        dropzeros!(Lᴴᴼᴴ)

        Fᴴᴼᴴ = qr(Lᴴᴼᴴ)

        G[N], B = G[N], (G[N] = 0;
        Fᴴᴼᴴ \ G)

        # Building G^(T) system
        c = 0
        for i = 1:N
            c += 1
            is[c] = i
            js[c] = i
            Ls[c] = 1
        end

        # L10
        l = 1
        c += 1
        is[c] = l * N + 1
        js[c] = (l - 1) * N + 1
        Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2vb[1])
        c += 1
        is[c] = l * N + 1
        js[c] = (l - 1) * N + 1 + 1
        Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2vb[1] - 1 / Δv)
        for i = 2:N-1
            c += 1
            is[c] = l * N + i
            js[c] = (l - 1) * N + i
            Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2vb[i] + 1 / Δv)
            c += 1
            is[c] = l * N + i
            js[c] = (l - 1) * N + i + 1
            Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2vb[i] - 1 / Δv)
        end
        c += 1
        is[c] = l * N + N
        js[c] = (l - 1) * N + N
        Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2vb[N] + 1 / Δv)

        # L11
        for i = 1:N
            c += 1
            is[c] = l * N + i
            js[c] = l * N + i
            Ls[c] = -ω₀ - νₜₒₜb[i]
        end

        # Multiterms
        for l = 1:2:lmax-2
            c += 1
            is[c] = l * N + 1
            js[c] = (l + 1) * N + 1
            Ls[c] = E * (l + 2) / 2(2l + 3) * (l / 2vb[1])
            c += 1
            is[c] = l * N + 1
            js[c] = (l + 1) * N + 1 + 1
            Ls[c] = E * (l + 2) / 2(2l + 3) * (l / 2vb[1] + 1 / Δv)
            for i = 2:N-1
                c += 1
                is[c] = l * N + i
                js[c] = (l + 1) * N + i
                Ls[c] = E * (l + 2) / 2(2l + 3) * (l / 2vb[i] - 1 / Δv)
                c += 1
                is[c] = l * N + i
                js[c] = (l + 1) * N + i + 1
                Ls[c] = E * (l + 2) / 2(2l + 3) * (l / 2vb[i] + 1 / Δv)
            end
            c += 1
            is[c] = l * N + N
            js[c] = (l + 1) * N + N
            Ls[c] = E * (l + 2) / 2(2l + 3) * (l / 2vb[N] - 1 / Δv)

            l += 1
            c += 1
            is[c] = l * N + 1
            js[c] = (l - 1) * N + 1
            Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2v[1] - 1 / Δv)
            for i = 2:N-1
                c += 1
                is[c] = l * N + i
                js[c] = (l - 1) * N + i - 1
                Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2v[i] + 1 / Δv)
                c += 1
                is[c] = l * N + i
                js[c] = (l - 1) * N + i
                Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2v[i] - 1 / Δv)
            end
            c += 1
            is[c] = l * N + N
            js[c] = (l - 1) * N + N - 1
            Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2v[N] + 1 / Δv)
            c += 1
            is[c] = l * N + N
            js[c] = (l - 1) * N + N
            Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2v[N])
            for i = 1:N
                c += 1
                is[c] = l * N + i
                js[c] = l * N + i
                Ls[c] = -ω₀ - ν̃(gas, v[i], l + 1) - νₗₒₛₛ[i]#νₜₒₜv[i]
            end
            c += 1
            is[c] = l * N + 1
            js[c] = (l + 1) * N + 1
            Ls[c] = E * (l + 2) / 2(2l + 3) * (l / 2v[1] + 1 / Δv)
            for i = 2:N-1
                c += 1
                is[c] = l * N + i
                js[c] = (l + 1) * N + i - 1
                Ls[c] = E * (l + 2) / 2(2l + 3) * (l / 2v[i] - 1 / Δv)
                c += 1
                is[c] = l * N + i
                js[c] = (l + 1) * N + i
                Ls[c] = E * (l + 2) / 2(2l + 3) * (l / 2v[i] + 1 / Δv)
            end
            c += 1
            is[c] = l * N + N
            js[c] = (l + 1) * N + N - 1
            Ls[c] = E * (l + 2) / 2(2l + 3) * (l / 2v[N] - 1 / Δv)
            c += 1
            is[c] = l * N + N
            js[c] = (l + 1) * N + N
            Ls[c] = E * (l + 2) / 2(2l + 3) * (l / 2v[N])

            l += 1
            c += 1
            is[c] = l * N + 1
            js[c] = (l - 1) * N + 1
            Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2vb[1])
            c += 1
            is[c] = l * N + 1
            js[c] = (l - 1) * N + 1 + 1
            Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2vb[1] - 1 / Δv)
            for i = 2:N-1
                c += 1
                is[c] = l * N + i
                js[c] = (l - 1) * N + i
                Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2vb[i] + 1 / Δv)
                c += 1
                is[c] = l * N + i
                js[c] = (l - 1) * N + i + 1
                Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2vb[i] - 1 / Δv)
            end
            c += 1
            is[c] = l * N + N
            js[c] = (l - 1) * N + N
            Ls[c] = E * (1 - l) / 2(2l - 1) * ((l + 1) / 2vb[N] + 1 / Δv)
            for i = 1:N
                c += 1
                is[c] = l * N + i
                js[c] = l * N + i
                Ls[c] = -ω₀ - ν̃(gas, vb[i], l + 1) - νₗₒₛₛb[i]#νₜₒₜvb[i]
            end
        end

        Lᵀ = sparse(@view(is[1:c]), @view(js[1:c]), @view(Ls[1:c]), (lmax + 1)N, (lmax + 1)N)
        dropzeros!(Lᵀ)

        Fᵀ = qr(Lᵀ)

        # G^(T)
        # A[l*N+i] = -v[i]*((G[(l-1)*N+i-1]+G[(l-1)*N+i])/2(2l-1) - (G[(l+1)*N+i-1]+G[(l+1)*N+i])/2(2l+3))
        # A[1:N] .= 0
        A = similar(G)
        @turbo for i = 1:N
            A[i] = 0
        end
        for l = 2:2:lmax-1
            for i = 1
                A[l*N+i] = -v[i] * (-(G[(l+1)*N+i]) / 2(2l + 3)) - v[i] * ((G[(l-1)*N+i]) / 2(2l - 1))
            end
            @turbo for i = 2:N
                A[l*N+i] = -v[i] * ((G[(l-1)*N+i-1]) / 2(2l - 1)) - v[i] * ((G[(l-1)*N+i]) / 2(2l - 1)) - v[i] * (-(G[(l+1)*N+i-1]) / 2(2l + 3)) - v[i] * (-(G[(l+1)*N+i]) / 2(2l + 3))
            end
        end
        for l = 1:2:lmax-2
            @turbo for i = 1:N-1
                A[l*N+i] = -vb[i] * (-(G[(l+1)*N+i]) / 2(2l + 3)) - vb[i] * (-(G[(l+1)*N+i+1]) / 2(2l + 3)) - vb[i] * ((G[(l-1)*N+i]) / 2(2l - 1)) - vb[i] * ((G[(l-1)*N+i+1]) / 2(2l - 1))
            end
            for i = N
                A[l*N+i] = -vb[i] * (-(G[(l+1)*N+i]) / 2(2l + 3)) - vb[i] * ((G[(l-1)*N+i]) / 2(2l - 1))
            end
        end
        for l = lmax
            @turbo for i = 1:N-1
                A[l*N+i] = -vb[i] * ((G[(l-1)*N+i]) / 2(2l - 1)) - vb[i] * ((G[(l-1)*N+i+1]) / 2(2l - 1))
            end
            for i = N
                A[l*N+i] = -vb[i] * ((G[(l-1)*N+i]) / 2(2l - 1))
            end
        end
        Gᵀ = Fᵀ \ A

        # G^(L)
        # A[l*N+i] = -v[i]*((l+1)/(2l+3)*(G[(l+1)*N+i-1]+G[(l+1)*N+i])/2+l/(2l-1)*(G[(l-1)*N+i-1]+G[(l-1)*N+i])/2)
        # A[N] = 0
        for l = 0
            for i = 1
                A[l*N+i] = ω₁F * G[l*N+i] - v[i] * ((l + 1) / (2l + 3) * (G[(l+1)*N+i]) / 2)
            end
            @turbo for i = 2:N-1
                A[l*N+i] = ω₁F * G[l*N+i] - v[i] * ((l + 1) / (2l + 3) * (G[(l+1)*N+i-1]) / 2) - v[i] * ((l + 1) / (2l + 3) * (G[(l+1)*N+i]) / 2)
            end
            A[N] = 0
        end
        for l = 2:2:lmax-1
            for i = 1
                A[l*N+i] = ω₁F * G[l*N+i] - v[i] * ((l + 1) / (2l + 3) * (G[(l+1)*N+i]) / 2) - v[i] * (l / (2l - 1) * (G[(l-1)*N+i]) / 2)
            end
            @turbo for i = 2:N
                A[l*N+i] = ω₁F * G[l*N+i] - v[i] * ((l + 1) / (2l + 3) * (G[(l+1)*N+i-1]) / 2) - v[i] * ((l + 1) / (2l + 3) * (G[(l+1)*N+i]) / 2) - v[i] * (l / (2l - 1) * (G[(l-1)*N+i-1]) / 2) - v[i] * (l / (2l - 1) * (G[(l-1)*N+i]) / 2)
            end
        end
        for l = 1:2:lmax-2
            @turbo for i = 1:N-1
                A[l*N+i] = ω₁F * G[l*N+i] - vb[i] * ((l + 1) / (2l + 3) * (G[(l+1)*N+i]) / 2) - vb[i] * ((l + 1) / (2l + 3) * (G[(l+1)*N+i+1]) / 2) - vb[i] * (l / (2l - 1) * (G[(l-1)*N+i]) / 2) - vb[i] * (l / (2l - 1) * (G[(l-1)*N+i+1]) / 2)
            end
            for i = N
                A[l*N+i] = ω₁F * G[l*N+i] - vb[i] * ((l + 1) / (2l + 3) * (G[(l+1)*N+i]) / 2) - vb[i] * (l / (2l - 1) * (G[(l-1)*N+i]) / 2)
            end
        end
        for l = lmax
            @turbo for i = 1:N-1
                A[l*N+i] = ω₁F * G[l*N+i] - vb[i] * (l / (2l - 1) * (G[(l-1)*N+i]) / 2) - vb[i] * (l / (2l - 1) * (G[(l-1)*N+i+1]) / 2)
            end
            for i = N
                A[l*N+i] = ω₁F * G[l*N+i] - vb[i] * (l / (2l - 1) * (G[(l-1)*N+i]) / 2)
            end
        end
        ldiv!(Fᴴᴼᴴ, A)
        ω₁R = sum(νₙₑₜ[i] * A[i] for i = 1:N) / (1 / Δv - sum(νₙₑₜ[i] * B[i] for i = 1:N))
        ω₁ = ω₁F + ω₁R
        Gᴸ = @. A + ω₁R * B

        # G^(2T)
        # A[l*N+i] = -v[i]/sqrt(3)*((l+1)/(2l+3)*((Gᴸ[(l+1)*N+i-1]+Gᴸ[(l+1)*N+i])+(l+2)*(Gᵀ[(l+1)*N+i-1]+Gᵀ[(l+1)*N+i]))/2 + l/(2l-1)*((Gᴸ[(l-1)*N+i-1]+Gᴸ[(l-1)*N+i])-(l-1)*(Gᵀ[(l-1)*N+i-1]+Gᵀ[(l-1)*N+i]))/2)
        # A[N] = 0
        ω₂F = Δv * sum(vb[i] / 3sqrt(3) * (Gᴸ[N+i] + 2Gᵀ[N+i]) for i = 1:N)
        for l = 0
            for i = 1
                A[l*N+i] = ω₂F * G[l*N+i] + ω₁ / sqrt(3) * Gᴸ[l*N+i] - v[i] / sqrt(3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i]) + (l + 2) * (Gᵀ[(l+1)*N+i])) / 2)
            end
            @turbo for i = 2:N-1
                A[l*N+i] = ω₂F * G[l*N+i] + ω₁ / sqrt(3) * Gᴸ[l*N+i] - v[i] / sqrt(3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i-1]) + (l + 2) * (Gᵀ[(l+1)*N+i-1])) / 2) - v[i] / sqrt(3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i]) + (l + 2) * (Gᵀ[(l+1)*N+i])) / 2)
            end
            A[N] = 0
        end
        for l = 2:2:lmax-1
            for i = 1
                A[l*N+i] = ω₂F * G[l*N+i] + ω₁ / sqrt(3) * Gᴸ[l*N+i] - v[i] / sqrt(3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i]) + (l + 2) * (Gᵀ[(l+1)*N+i])) / 2) - v[i] / sqrt(3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i]) - (l - 1) * (Gᵀ[(l-1)*N+i])) / 2)
            end
            @turbo for i = 2:N
                A[l*N+i] = ω₂F * G[l*N+i] + ω₁ / sqrt(3) * Gᴸ[l*N+i] - v[i] / sqrt(3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i-1]) + (l + 2) * (Gᵀ[(l+1)*N+i-1])) / 2) - v[i] / sqrt(3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i]) + (l + 2) * (Gᵀ[(l+1)*N+i])) / 2) - v[i] / sqrt(3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i-1]) - (l - 1) * (Gᵀ[(l-1)*N+i-1])) / 2) - v[i] / sqrt(3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i]) - (l - 1) * (Gᵀ[(l-1)*N+i])) / 2)
            end
        end
        for l = 1:2:lmax-2
            @turbo for i = 1:N-1
                A[l*N+i] = ω₂F * G[l*N+i] + ω₁ / sqrt(3) * Gᴸ[l*N+i] - vb[i] / sqrt(3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i]) + (l + 2) * (Gᵀ[(l+1)*N+i])) / 2) - vb[i] / sqrt(3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i+1]) + (l + 2) * (Gᵀ[(l+1)*N+i+1])) / 2) - vb[i] / sqrt(3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i]) - (l - 1) * (Gᵀ[(l-1)*N+i])) / 2) - vb[i] / sqrt(3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i+1]) - (l - 1) * (Gᵀ[(l-1)*N+i+1])) / 2)
            end
            for i = N
                A[l*N+i] = ω₂F * G[l*N+i] + ω₁ / sqrt(3) * Gᴸ[l*N+i] - vb[i] / sqrt(3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i]) + (l + 2) * (Gᵀ[(l+1)*N+i])) / 2) - vb[i] / sqrt(3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i]) - (l - 1) * (Gᵀ[(l-1)*N+i])) / 2)
            end
        end
        for l = lmax
            @turbo for i = 1:N-1
                A[l*N+i] = ω₂F * G[l*N+i] + ω₁ / sqrt(3) * Gᴸ[l*N+i] - vb[i] / sqrt(3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i]) - (l - 1) * (Gᵀ[(l-1)*N+i])) / 2) - vb[i] / sqrt(3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i+1]) - (l - 1) * (Gᵀ[(l-1)*N+i+1])) / 2)
            end
            for i = N
                A[l*N+i] = ω₂F * G[l*N+i] + ω₁ / sqrt(3) * Gᴸ[l*N+i] - vb[i] / sqrt(3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i]) - (l - 1) * (Gᵀ[(l-1)*N+i])) / 2)
            end
        end
        ldiv!(Fᴴᴼᴴ, A)
        ω₂R = sum(νₙₑₜ[i] * A[i] for i = 1:N) / (1 / Δv - sum(νₙₑₜ[i] * B[i] for i = 1:N))
        G²ᵀ = @. A + ω₂R * B
        ω₂ = ω₂F + ω₂R

        # G^(2L)
        # A[l*N+i] = v[i]*sqrt(2/3)*((l+1)/(2l+3)*((Gᴸ[(l+1)*N+i-1]+Gᴸ[(l+1)*N+i])-(l+2)/2*(Gᵀ[(l+1)*N+i-1]+Gᵀ[(l+1)*N+i]))/2 + l/(2l-1)*((Gᴸ[(l-1)*N+i-1]+Gᴸ[(l-1)*N+i])+(l-1)/2*(Gᵀ[(l-1)*N+i-1]+Gᵀ[(l-1)*N+i]))/2)
        # A[N] = 0
        ω̄₂F = -Δv * sum(vb[i] / 3 * sqrt(2 / 3) * (Gᴸ[N+i] - Gᵀ[N+i]) for i = 1:N)
        for l = 0
            for i = 1
                A[l*N+i] = ω̄₂F * G[l*N+i] - ω₁ * sqrt(2 / 3) * Gᴸ[l*N+i] + v[i] * sqrt(2 / 3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i]) - (l + 2) / 2 * (Gᵀ[(l+1)*N+i])) / 2)
            end
            @turbo for i = 2:N-1
                A[l*N+i] = ω̄₂F * G[l*N+i] - ω₁ * sqrt(2 / 3) * Gᴸ[l*N+i] + v[i] * sqrt(2 / 3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i-1]) - (l + 2) / 2 * (Gᵀ[(l+1)*N+i-1])) / 2) + v[i] * sqrt(2 / 3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i]) - (l + 2) / 2 * (Gᵀ[(l+1)*N+i])) / 2)
            end
            A[N] = 0
        end
        for l = 2:2:lmax-1
            for i = 1
                A[l*N+i] = ω̄₂F * G[l*N+i] - ω₁ * sqrt(2 / 3) * Gᴸ[l*N+i] + v[i] * sqrt(2 / 3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i]) - (l + 2) / 2 * (Gᵀ[(l+1)*N+i])) / 2) + v[i] * sqrt(2 / 3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i]) + (l - 1) / 2 * (Gᵀ[(l-1)*N+i])) / 2)
            end
            @turbo for i = 2:N
                A[l*N+i] = ω̄₂F * G[l*N+i] - ω₁ * sqrt(2 / 3) * Gᴸ[l*N+i] + v[i] * sqrt(2 / 3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i-1]) - (l + 2) / 2 * (Gᵀ[(l+1)*N+i-1])) / 2) + v[i] * sqrt(2 / 3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i]) - (l + 2) / 2 * (Gᵀ[(l+1)*N+i])) / 2) + v[i] * sqrt(2 / 3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i-1]) + (l - 1) / 2 * (Gᵀ[(l-1)*N+i-1])) / 2) + v[i] * sqrt(2 / 3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i]) + (l - 1) / 2 * (Gᵀ[(l-1)*N+i])) / 2)
            end
        end
        for l = 1:2:lmax-2
            @turbo for i = 1:N-1
                A[l*N+i] = ω̄₂F * G[l*N+i] - ω₁ * sqrt(2 / 3) * Gᴸ[l*N+i] + vb[i] * sqrt(2 / 3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i]) - (l + 2) / 2 * (Gᵀ[(l+1)*N+i])) / 2) + vb[i] * sqrt(2 / 3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i+1]) - (l + 2) / 2 * (Gᵀ[(l+1)*N+i+1])) / 2) + vb[i] * sqrt(2 / 3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i]) + (l - 1) / 2 * (Gᵀ[(l-1)*N+i])) / 2) + vb[i] * sqrt(2 / 3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i+1]) + (l - 1) / 2 * (Gᵀ[(l-1)*N+i+1])) / 2)
            end
            for i = N
                A[l*N+i] = ω̄₂F * G[l*N+i] - ω₁ * sqrt(2 / 3) * Gᴸ[l*N+i] + vb[i] * sqrt(2 / 3) * ((l + 1) / (2l + 3) * ((Gᴸ[(l+1)*N+i]) - (l + 2) / 2 * (Gᵀ[(l+1)*N+i])) / 2) + vb[i] * sqrt(2 / 3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i]) + (l - 1) / 2 * (Gᵀ[(l-1)*N+i])) / 2)
            end
        end
        for l = lmax
            @turbo for i = 1:N-1
                A[l*N+i] = ω̄₂F * G[l*N+i] - ω₁ * sqrt(2 / 3) * Gᴸ[l*N+i] + vb[i] * sqrt(2 / 3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i]) + (l - 1) / 2 * (Gᵀ[(l-1)*N+i])) / 2) + vb[i] * sqrt(2 / 3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i+1]) + (l - 1) / 2 * (Gᵀ[(l-1)*N+i+1])) / 2)
            end
            for i = N
                A[l*N+i] = ω̄₂F * G[l*N+i] - ω₁ * sqrt(2 / 3) * Gᴸ[l*N+i] + vb[i] * sqrt(2 / 3) * (l / (2l - 1) * ((Gᴸ[(l-1)*N+i]) + (l - 1) / 2 * (Gᵀ[(l-1)*N+i])) / 2)
            end
        end
        ldiv!(Fᴴᴼᴴ, A)
        ω̄₂R = sum(νₙₑₜ[i] * A[i] for i = 1:N) / (1 / Δv - sum(νₙₑₜ[i] * B[i] for i = 1:N))
        G²ᴸ = @. A + ω̄₂R * B
        ω̄₂ = ω̄₂F + ω̄₂R

        W = ω₁ * γ

        DTF = γ / σ₀ * Δv * sum(vb[i] / 3 * Gᵀ[N+i] for i = 1:N)
        DLF = γ / σ₀ * Δv * sum(vb[i] / 3 * Gᴸ[N+i] for i = 1:N)
        DT = γ / σ₀ * (ω₂ / sqrt(3) + ω̄₂ / sqrt(6))
        DL = γ / σ₀ * (ω₂ / sqrt(3) - ω̄₂ * sqrt(2 / 3))

        Rnet = γ * σ₀ * ω₀
        αη = -Rnet / W
        αηF = -Rnet / WF

        WBrambring = let Δ = 1 + 4(DL / W) * αη
            Δ < 0 ? NaN : W * (1 + sqrt(Δ)) / 2
        end
        αηBrambring = -Rnet / WBrambring
        DLBrambring = DL
        DLKondo = DL
        WKondo = W + 2 * αη * DL

        kel = γ * σ₀ * Δv * sum(νₘ(gas, v[i]) * G[i] for i = 1:N)
        if superelastic
            kex = γ * σ₀ * Δv * sum(νₑₓ(gas, v[i], T_og, supertype) * G[i] for i = 1:N)
        else
            kex = γ * σ₀ * Δv * sum(νₑₓ(gas, v[i]) * G[i] for i = 1:N)
        end

        kio = γ * σ₀ * Δv * sum(νᵢₒ(gas, v[i]) * G[i] for i = 1:N)
        kat = γ * σ₀ * Δv * sum(νₐₜ(gas, v[i]) * G[i] for i = 1:N)

        α = -kio / W
        η = -kat / W

        G₀ = @view G[1:N]
        G₁ = @view G[N+1:2N]

        μ = abs(W) * 1E21 / ETd
        μF = abs(WF) * 1E21 / ETd
        μBrambring = abs(WBrambring) * 1E21 / ETd
        μKondo = abs(WKondo) * 1E21 / ETd
        converged = !(attempts > 100 || !converged)


        #return W, DL, kio, kat, α, η 
        return (; v,
            vb,
            Δv,
            G,
            G₀,
            G₁,
            Gᵀ,
            Gᴸ,
            G²ᵀ,
            G²ᴸ,
            ω₀,
            ω₁,
            ω₂,
            ω̄₂,
            ε̄,
            WF,
            W,
            DTF,
            DLF,
            DT,
            DL,
            εmax,
            vmax,
            Rnet,
            αη,
            αηF,
            α,
            η,
            WBrambring,
            αηBrambring,
            DLBrambring,
            μ,
            μF,
            μBrambring,
            WKondo,
            μKondo,
            DLKondo,
            kel,
            kex,
            kio,
            kat,
            iterations,
            converged,)::NamedTuple{(:v, :vb, :Δv, :G, :G₀, :G₁, :Gᵀ, :Gᴸ, :G²ᵀ, :G²ᴸ, :ω₀, :ω₁, :ω₂, :ω̄₂, :ε̄, :WF, :W, :DTF, :DLF, :DT, :DL, :εmax, :vmax, :Rnet, :αη, :αηF, :α, :η, :WBrambring, :αηBrambring, :DLBrambring, :μ, :μF, :μBrambring, :WKondo, :μKondo, :DLKondo, :kel, :kex, :kio, :kat, :iterations, :converged),Tuple{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64},StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64},Float64,Vector{Float64},SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true},SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true},Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Int64,Bool}}
    end
end
function pts(gas, Ens; hard_error=false, kws...)
    stop_flag = Atomic{Bool}(false)
    sols = Vector{NamedTuple{(:v, :vb, :Δv, :G, :G₀, :G₁, :Gᵀ, :Gᴸ, :G²ᵀ, :G²ᴸ, :ω₀, :ω₁, :ω₂, :ω̄₂, :ε̄, :WF, :W, :DTF, :DLF, :DT, :DL, :εmax, :vmax, :Rnet, :αη, :αηF, :α, :η, :WBrambring, :αηBrambring, :DLBrambring, :μ, :μF, :μBrambring, :WKondo, :μKondo, :DLKondo, :kel, :kex, :kio, :kat, :iterations, :converged),Tuple{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64},StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64},Float64,Vector{Float64},SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true},SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true},Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Int64,Bool}}}(undef, length(Ens))
    Threads.@threads for i in eachindex(Ens)
        sols[i] = pt(gas; kws..., E=Ens[i])
        if hard_error
            if !sols[i].converged || sols[i].DL < 0 || sols[i].W > 0 || (sols[i].kio - sols[i].kat) > 1e21
                stop_flag[] = true
            end
            if stop_flag[]
                break
            end
        end
    end
    return stop_flag[], [Ens[i] for i in eachindex(sols) if sols[i].converged], [s.ε̄ for s in sols if s.converged], [s.W for s in sols if s.converged], [s.WF for s in sols if s.converged], [s.DL for s in sols if s.converged], [s.DLF for s in sols if s.converged], [s.DT for s in sols if s.converged], [s.DTF for s in sols if s.converged], [s.kel for s in sols if s.converged], [s.kex for s in sols if s.converged], [s.kio for s in sols if s.converged], [s.kat for s in sols if s.converged], [s.α for s in sols if s.converged], [s.η for s in sols if s.converged]
end
end
