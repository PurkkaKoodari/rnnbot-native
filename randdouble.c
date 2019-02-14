#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdbool.h>

#include "randdouble.h"

double randdouble() {
    return rand() * (1.0 / RAND_MAX);
}

double randnormal() {
    static const double two_pi = 2 * M_PI;

    static double z1;
    static bool just_generated = true;

    just_generated = !just_generated;
    if (just_generated) return z1;

    double z0, u0, u1;

    do {
        u0 = randdouble();
        u1 = randdouble();
    } while (u0 <= DBL_MIN);

    z0 = sqrt(-2.0 * log(u0)) * cos(two_pi * u1);
    z1 = sqrt(-2.0 * log(u0)) * sin(two_pi * u1);

    return z0;
}
