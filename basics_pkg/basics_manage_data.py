import numpy as np



def retrieve_instances(filename, k, select_lines=False):
    f = open(filename, 'r')

    contents = f.readlines()
    contents = contents[1:]
    if select_lines != False:
        contents = contents[:select_lines]

    insts = []
    sols = []
    for line in contents:
        inst = []
        sol = []
        instance, solution = line.split('#')

        for x in instance.split('\t')[:-1]:
            inst.append(int(x))
        for x in solution.split('\t')[:-1]:
            sol.append(int(x))

        num_clauses = len(inst)//k
        inst = np.asarray(inst).reshape(num_clauses, k)
        insts.append(inst)
        sols.append(np.asarray(sol))
    f.close()
    return np.asarray(insts, dtype = 'object'), np.asarray(sols)
