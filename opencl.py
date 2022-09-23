#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import time

NBODIES   = 5000
ITERS     = 100000
PLOTITERS = 10 # Plot every Nth

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

np.random.seed(0)

host_masses = 1e10 * np.random.rand(NBODIES).astype(np.float32)
host_xpos   = 1e3 * (np.random.rand(NBODIES).astype(np.float32) - 0.5)
host_ypos   = 1e3 * (np.random.rand(NBODIES).astype(np.float32) - 0.5)
host_xvel   = 10 * (np.random.rand(NBODIES).astype(np.float32) - 0.5)
host_yvel   = 10 * (np.random.rand(NBODIES).astype(np.float32) - 0.5)
host_kinet  = np.zeros(host_masses.shape).astype(np.float32)

mf = cl.mem_flags
masses = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_masses)
xpos   = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_xpos)
ypos   = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_ypos)
xvel   = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_xvel)
yvel   = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_yvel)
kinet  = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_kinet)

def read_to_string(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r") as f:
            return str(f.read())

prg = cl.Program(ctx, read_to_string("nbody.cl")).build()
nbody            = prg.nbody
kinetic_energies = prg.kinetic_energies

fig = plt.figure()
plt.grid(True)

for i in range(0, ITERS):
    # process
    nbody(queue, host_masses.shape, None, masses, xpos, ypos, xvel, yvel)

    if (i % PLOTITERS == 0):
        # kinetic_energies(queue, host_masses.shape, None, masses, xvel, yvel, kinet)
        # read out results
        cl.enqueue_copy(queue, dest=host_xpos, src=xpos)
        cl.enqueue_copy(queue, dest=host_ypos, src=ypos)
        cl.enqueue_copy(queue, dest=host_yvel, src=yvel)
        cl.enqueue_copy(queue, dest=host_xvel, src=xvel)
        # cl.enqueue_copy(queue, dest=host_kinet, src=kinet)
        queue.finish()

        plt.clf()
        plt.scatter(host_xpos, host_ypos)
        # plt.quiver(host_xpos, host_ypos, host_xvel, host_yvel)
        plt.draw()
        plt.pause(0.01)

