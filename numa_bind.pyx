from libc.stdlib cimport malloc, free
from libc.string cimport memset

cdef extern from "numa.h":
    cdef struct bitmask:
        unsigned long size
        unsigned long *maskp

    int numa_available()
    void numa_set_bind_policy(int strict)
    int numa_max_node()
    void numa_set_membind(bitmask* nodemask)
    bitmask *numa_allocate_nodemask()
    void numa_bitmask_setbit(bitmask *mask, unsigned int node)
    void numa_bitmask_clearall(bitmask *mask)

def set_membind(int node_id):
    if numa_available() == -1:
        raise RuntimeError("NUMA not available")
    cdef bitmask *mask = numa_allocate_nodemask()
    numa_bitmask_clearall(mask)
    numa_bitmask_setbit(mask, node_id)
    numa_set_bind_policy(1)  # strict binding
    numa_set_membind(mask)
