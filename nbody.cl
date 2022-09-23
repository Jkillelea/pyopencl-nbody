
// Compute 0.5 * m * v^2 for each particle
__kernel void kinetic_energies(__global float *masses, __global float *xvel, __global float *yvel, __global float *ke) {
    unsigned gid = get_global_id(0);
    ke[gid] = 0.5 * masses[gid] * (xvel[gid] * xvel[gid] + yvel[gid] * yvel[gid]);
}

// List of bodies in space, and does an euler approximation on them
__kernel void nbody(__global float *masses, __global float *xpos, __global float *ypos, __global float *xvel, __global float *yvel)
{
    int gid     = get_global_id(0);
    int nbodies = get_global_size(0);

    // compute acceleration caused by all other bodies
    const float G = 6.67e-11;
    const float dt = 1.0;

    float xacc = 0.0;
    float yacc = 0.0;

    // For every other solar system body
    for (int i = 0; i < nbodies; i++)
    {
        if (i != gid)
        {
            float dx = xpos[gid] - xpos[i];
            float dy = ypos[gid] - ypos[i];

            if (fabs(dx) > 1.0)
            {
                xacc += -sign(dx) * (G * masses[i]) / (dx * dx);
            }

            if (fabs(dy) > 1.0)
            {
                yacc += -sign(dy) * (G * masses[i]) / (dy * dy);
            }
        }
    }

    // Propagate own position based on velocity
    xpos[gid] += dt * xvel[gid];
    ypos[gid] += dt * yvel[gid];

    // update own velocity based on acceleration
    xvel[gid] += dt * xacc;
    yvel[gid] += dt * yacc;
}
