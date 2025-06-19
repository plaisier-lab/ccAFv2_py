##########################################################
## OncoMerge:  ccAFv2.pdx                                ##
##  ______     ______     __  __                        ##
## /\  __ \   /\  ___\   /\ \/\ \                       ##
## \ \  __ \  \ \___  \  \ \ \_\ \                      ##
##  \ \_\ \_\  \/\_____\  \ \_____\                     ##
##   \/_/\/_/   \/_____/   \/_____/                     ##
## @Developed by: Plaisier Lab                          ##
##   (https://plaisierlab.engineering.asu.edu/)         ##
##   Arizona State University                           ##
##   242 ISTB1, 550 E Orange St                         ##
##   Tempe, AZ  85281                                   ##
## @Author:  Chris Plaisier, Samantha O'Connor,         ##
#            Thurston Herricks                          ##
## @License:  GNU GPLv3                                 ##
##                                                      ##
## If this program is used in your analysis please      ##
## mention who built it. Thanks. :-)                    ##
##########################################################

# ccAFv2.pyx
# distutils: language = c

cdef extern from "C_ccAFv2.h":
    void entry(const float[1][861] inp_tensor, float[1][7] oup_tensor)

def run_model(double[:, ::1] input):
    cdef float[1][861] inp_vec
    cdef float[1][7] oup_vec

    for i in range(861):
        inp_vec[0][i] = <float>input[0][i]

    entry(inp_vec, oup_vec)

    return [oup_vec[0][i] for i in range(7)]


