using Combinatorics

function kSAT(n, k)
    all_combs = zeros(Int, 2^n, n)
    kk = 2 ## We start at 2 because the first is the all-0 configuration
    for i in combinations(1:n)
        for j in i
            all_combs[kk, j] = 1
        end
        kk += 1
    end

    inst = zeros(Int, 0, k)

    unique_sat = 0
    i = 1
    while unique_sat == 0
        newinst_spins = rand(1:n, 1, k) ## Para que me salga como vector columna
        newset_inst = vcat(inst, newinst_spins)
        m, k = size(newset_inst)
        
        good_combs = copy(all_combs)
        number_good_combs = length(good_combs)
        for ii in 1:m
            instance = newset_inst[ii,:]
            # println(instance, " done")

            mm, nn = size(good_combs)
            index2remove = []
            for index_c in 1:mm
                c = good_combs[index_c, :]
                # println(c, " ", instance)
                if mod(sum(c[instance]), 2) == 0
                    append!(index2remove, index_c) 
                end
            end

            for rmv in index2remove
                good_combs = good_combs[1:end .!= rmv, 1:end] #remove from the list of combinations that are valid
            end

            number_good_combs = size(good_combs)[1]
            # println("number of good combs left ", number_good_combs)
            if number_good_combs == 0 #break the loop and pick a new instance (i.e., don't update i) if this one has run out if chances
                break
            end
        end

        if number_good_combs != 0 # add a new constraint to the list
            inst = newset_inst
            if number_good_combs == 1 #exit loop
                unique_sat = 1
                # println("insts ", inst)
                # println("winner ", good_combs)
                return inst, good_combs     ### We need to return both variables out of the while loop
            end
        end
        # println("inst", inst)
    end

    return inst, good_combs
end



function ensure_different_problem(is1, sol1, is2, sol2) ##returns 0 for different problem, 1 for equals
    if sol1 != sol2 ## See if they have the same solution
        return 0 
    else
        m1, n1 = size(is1)
        m2, n2 = size(is2)
        
        if m1 != m2 ## See if they have the same number of instances
            return 0
        else
            equal_inst = 0
            for i in 1:m1
                inst1 = sort(is1[i, :])
                for j in 1:m2
                    inst2 = sort(is2[i, :])
                    if inst1 == inst2
                        equal_inst += 1
                    end
                end
                if equal_inst == 0 # If there was a distinct instance, return 0
                    return 0
                end
            end

            if equal_inst == m1
                return 1
            end
        end
    end
end

n = 10 ## n = 12 takes about 4 seconds to generate 3SAT
inst1, good_combs1 = kSAT(n, 3)
println("inst1 ", inst1)
inst2, good_combs2 = kSAT(n, 3)
println("inst2 ", inst2)
ensure_different_problem(inst1, good_combs1, inst2, good_combs2)