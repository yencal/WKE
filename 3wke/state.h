#ifndef STATE_H
#define STATE_H  


#include "matar.h"

using namespace mtr;

// node_state
struct node_t {

    DCArrayKokkos <double> coords;

    DCArrayKokkos <double> energy;

    void initialize(size_t num_rk, size_t num_nodes)
    {
        this->coords = DCArrayKokkos <double> (num_nodes);
        this->energy   = DCArrayKokkos <double> (num_rk, num_nodes);
    };

};// end node_t


struct elem_t {

    DCArrayKokkos <double> coords;

    DCArrayKokkos <double> energy;

    DCArrayKokkos <double> length;

    void initialize(size_t num_rk, size_t num_cells)
    {
        this->coords = DCArrayKokkos <double> (num_cells);
        this->energy   = DCArrayKokkos <double> (num_rk, num_cells);
        this->length   = DCArrayKokkos <double> (num_cells);
        
    };


};// end elem_t



#endif 
