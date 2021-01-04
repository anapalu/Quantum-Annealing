using Combinatorics
using Random


function select_different_random(n, k)
    rand_pool = Array(1:n)
    instance = rand(rand_pool, k)
    unique_instance = unique(instance)
    n_different = length(unique_instance)
    while n_different < k
        filter!(x->!(x in unique_instance), rand_pool)
        append2inst = rand(rand_pool, k-n_different)
        instance = vcat(unique_instance, append2inst)
        unique_instance = unique(instance)
        n_different = length(unique_instance)
    end
    return sort(instance)
end

function factorial(a)
    if a == 0
        fact = 1
    else
        fact = a 
        for i in 1:a-1
            fact *= i
        end
    end
    return fact
end


function kSAT(n, k, seed = 1234)
    maxcount = 0 ## control to avoid getting stuck when simply changing the last proposed instance won't do

    ## Generate all possible combinations
    all_combs = zeros(Int, 2^n, n)
    kk = 2 ## we start at 2 because the first is the all-0 configuration
    for i in combinations(1:n)
        for j in i
            all_combs[kk, j] = 1
        end
        kk += 1
    end
    good_combs = copy(all_combs) ## initially, all of them are good combs

    combinatorial_num_n_over_k = factorial(n)/factorial(k)/factorial(n-k) ##number of possible instances

    inst = zeros(Int, 0, k) ## initialise list of instances

    Random.seed!(seed) ## set seed

    unique_sat = 0 ### control parameter to see whether we have successfully built a k-SAT
    while unique_sat == 0
        newinst_spins = select_different_random(n, k) 
        
        sort!(newinst_spins) ## We want to sort them in order to quickly identify repeated instances
        newinst_spins = reshape(newinst_spins, (1, k))  ## We want a column vector

        counter = 0 ## control to stope when we have gone over all the possible choices of select_different_random (aprox)
        while any([(newinst_spins == reshape(inst[ii,:], (1, k))) for ii=1:size(inst)[1]]) && counter < combinatorial_num_n_over_k - size(inst)[1]
            counter += 1
            newinst_spins = select_different_random(n, k)
            sort!(newinst_spins) ## We want to sort them in order to quickly identify repeated instances
            newinst_spins = reshape(newinst_spins, (1, k))  ## We want a column vector
        end

        newset_inst = vcat(inst, newinst_spins) ## current test list of instances
        
        m, _ = size(newset_inst)
        num_goodcombs, _ = size(good_combs)

        for ii in 1:m ## run through all current test instances
            
            curr_inst = newset_inst[ii, :] 
            jj = 1 ## index of the goodcomb being examined
            while jj <= num_goodcombs
                
                if sum(good_combs[jj, curr_inst]) !== 1
                    good_combs = good_combs[setdiff(1:end, jj), :] ## remove bad comb
                    
                else
                    jj += 1 ## we only increase the index when we have not removed the instance
                end
                num_goodcombs, _ = size(good_combs)  # update number of good combs
                
                if num_goodcombs == 0 ## exit loop sooner if we run out of good combs
                    break
                end
            end

            if num_goodcombs == 0 ## break the loop and pick a new instance (i.e., don't update i) if this one has run out if chances
                
                if maxcount >= Int(combinatorial_num_n_over_k) // 2 ## if we have found no solution at this point, start all over
                                                                        ## THIS BOUND IS AD HOC, IT MAY BE OPTIMISED
                    inst = reshape(select_different_random(n, k), (1,k))
                    good_combs = copy(all_combs)
                    maxcount = 0
                end
                break
            end
        end
        
        if num_goodcombs != 0 ## add the new constraint to the list
            inst = newset_inst

            if size(inst)[1] == 0 ## set newset_inst as inst in the first round
                inst = newset_inst
            end
            
            if num_goodcombs == 1 ## exit loop
                unique_sat = 1
                return inst, good_combs ## we actually exit the function here so we need not update unique_sat, but oh well
                                            ## I'm not so sure this would be the case in Python, for example, so it is not so bad
                                                ## to keep it in mind for the sake of translation
            end
        end
        maxcount += 1
        
    end

    # return inst, good_combs (see previous comment)
end



function ensure_different_problem(is1, sol1, is2, sol2) ##returns 0 for different problem, 1 for equals
    if sol1 != sol2 ## See if they have the same solution
        return 0 
    else
        m1, n1 = size(is1)
        m2, n2 = size(is2)
        
        if m1 != m2 ## See if they have the same number of instances
            return 0
        else ## check if all instances are equal
            equal_inst = 0
            for i in 1:m1
                inst1 = is1[i, :] 
                for j in 1:m2
                    inst2 = is2[j, :]
                    if inst1 == inst2 ## (each instance was already sorted)
                        equal_inst += 1
                    end
                end
                if equal_inst == 0 # If there was a different instance, return 0
                    return 0
                end
            end
            if equal_inst == m1
                return 1
            end
        end
    end
end

n = 12 ## n = 12 takes about 4 seconds to generate 3SAT
seed0 = 99790
inst1, good_combs1 = kSAT(n, 3, seed0)
println("inst1 ", inst1, "sol1", good_combs1)
inst2, good_combs2 = kSAT(n, 3, seed0 + 1)
println("inst2 ", inst2, "sol2", good_combs2)
ensure_different_problem(inst1, good_combs1, inst2, good_combs2)
