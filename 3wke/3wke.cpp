#include "matar.h"
#include "state.h"
#include <stdio.h>
#include <math.h>  // c math lib
#include <chrono>
using namespace mtr;

// -----------------------------------------------------------------------------
//    Global variables
// -----------------------------------------------------------------------------
const double fuzz = 1.0E-15;
const double huge = 1.0E15;


// -----------------------------------------------------------------------------
//    wave number domain
// -----------------------------------------------------------------------------
struct region_t{
   double p_min;
   double p_max;
};


// -----------------------------------------------------------------------------
//    Functions
// -----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
int get_corners_in_cell(int,int);

KOKKOS_INLINE_FUNCTION
int get_corners_in_node(int,int);


// -----------------------------------------------------------------------------
//    The Main function
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]){
    
    // -------------------------------
    //    User settable variables
    // -------------------------------

    // time step settings
    const double time_max = 20.0;
    double       dt       = 0.01;
    const double dt_max   = 100;
    const double dt_cfl   = 0.3;
    const int    num_rk_stages = 2;
    const int    max_cycles = 2000000;

    // mesh information
    const double  p_min = 0.0;
    const double  p_max = 100.0;
    const int num_cells = 1000;


    // -------------------------------
    
    printf("\nstarting FE code\n");
    
    FILE * myfile;
    
    // This is the meat in the code, it must be inside Kokkos scope
    Kokkos::initialize(argc, argv);
    {
        
        // 1D linear element int( -Grad(phi) dot J^{-1}j )
        //const double integral_grad_basis[2] = {1.0, -1.0};
        
        // calculate mesh information based on inputs
        const int num_nodes = num_cells+1;
        double dp = (p_max-p_min)/num_cells;
        
        // initialize the time to zero
        double time = 0.0;
    
        // --- setup variables based on user inputs ---
        
        // nodal variables
        DCArrayKokkos <double> node_energy(num_nodes);    // energy = \omega(p)f(t,p) = g(t,p)
        DCArrayKokkos <double> node_energy_n(num_nodes);  // energy at  t_n
        DCArrayKokkos <double> node_residual(num_nodes);          

	// mesh variables
        DCArrayKokkos <double> cell_coords(num_cells);   // coordinates of cell
        DCArrayKokkos <double> cell_length(num_cells);   // length of the cell
        
        DCArrayKokkos <double> node_coords(num_nodes);   // coordinates of nodes
        DCArrayKokkos <double> node_coords_n(num_nodes); // coordinates at t_n
        
        // --- build the mesh ---
        
        // calculate nodal coordinates of the mesh
        FOR_ALL (node_id, 0, num_nodes, {
           node_coords(node_id) = double(node_id) * dp;
        }); // end parallel for on device

        
        
        // calculate cell center coordinates of the mesh
        FOR_ALL (cell_id, 0, num_cells, {
            cell_coords(cell_id) =
                           0.5*(node_coords(cell_id) + node_coords(cell_id+1));
            
            cell_length(cell_id)  = node_coords(cell_id+1) - node_coords(cell_id);
        }); // end parallel for on device

        
        // intialize the nodal state that is internal to the mesh
        FOR_ALL (node_id, 1, num_nodes-1, {
            node_energy(node_id) = 0.0;// initialize energy at n+1
	    node_energy_n(node_id) = 0.0;// initialize energy at n

	    // set IC //
	    if ( p_min <= node_coords(node_id) and node_coords(node_id) <= 100){
              node_energy_n(node_id) = node_coords(node_id)*(33.33-node_coords(node_id))*(33.33-node_coords(node_id))*(100-node_coords(node_id));// a little energy at lower modes and more energy at higher modes  
	      node_energy_n(node_id) = node_energy_n(node_id)/129629629.629; // normalize
	    }
	    else if ( 100 < node_coords(node_id) ){
	      node_energy_n(node_id) = 0.0;
	    }
        }); // end parallel for on device

        
        RUN ({
            
            node_energy(0) = 0.0;
            node_energy(num_nodes-1) = 0.0;
            node_energy_n(0) = 0.0;
            node_energy_n(num_nodes-1) = 0.0;

        }); // end run once on the device
        
        
        
        // -------------------------------
        //    Print initial state to file
        // -------------------------------
        
        // update the host side to print (i.e., copy from device to host)
        node_energy_n.update_host();
        
        // write out the intial conditions to a file on the host
        myfile=fopen("time0.txt","w");
        fprintf(myfile,"# x energy \n");
        
        // write data on the host side
        for (int cell_id=0; cell_id<num_cells; cell_id++){
        fprintf( myfile,"%f\t%f\t%f\t%f\t%f\n",
                 cell_coords.host(cell_id),
                 node_energy_n.host(cell_id) );
        }
        fclose(myfile);
        
        auto time_1 = std::chrono::high_resolution_clock::now();
        
        // -------------------------------------
        // Solve equations until time=time_max
        // -------------------------------------

	for (int cycle = 0; cycle<max_cycles; cycle++){
           
            
            // get the new time step
            double dt_ceiling = dt*1.1;
            
            // parallel reduction with min
            double dt_lcl;
            double min_dt_calc;
            REDUCE_MIN(cell_id, 0, num_cells, dt_lcl, {
                // mesh size
                double dx_lcl = node_coords(cell_id+1) - node_coords(cell_id);
                
                // local dt calc
                double dt_lcl_ = dt_cfl;//*dx_lcl/(cell_sspd(cell_id) + fuzz);
                
                // make dt be in bounds
                dt_lcl_ = fmin(dt_lcl_, dt_max);
                dt_lcl_ = fmin(dt_lcl_, time_max-time);
                dt_lcl_ = fmin(dt_lcl_, dt_ceiling);
        
                if (dt_lcl_ < dt_lcl) dt_lcl = dt;//_lcl_;
                        
            }, min_dt_calc); // end parallel reduction on min
            Kokkos::fence();
            
            // save the min dt
            if(min_dt_calc < dt) dt = min_dt_calc;
            
            
            //printf("time = %f, dt = %f \n", time, dt);
            if (dt<=fuzz) break;
            
            
            
            // --- integrate forward in time ---
            
            // Runge-Kutta loop
            for (int rk_stage=0; rk_stage<num_rk_stages; rk_stage++ ){
               
                // rk coefficient on dt
                double rk_alpha = 1.0/(double(num_rk_stages) - double(rk_stage));
            
                
                // save the state at t_n
                if (rk_stage==0){
                    
                    // nodal state
                    FOR_ALL (node_id, 0, num_nodes, {
                          node_energy_n(node_id)    = node_energy(node_id);
                    }); // end parallel for on device
                    
                    
                } // end if
                
                
                
                // --- RHS --- //
                


                
                // --- Calculate energy update  ---
                
                
                FOR_ALL (node_id, 0, num_nodes-1, {
                    
                    // update velocity
                    node_energy(node_id) = node_energy_n(node_id) -
                                rk_alpha*dt;//node_mass(node_id)*sum_node_res(node_id);
                    
                }); // end parallel for on device

                
            } // end rk loop

            
            
            // update the time
            time += dt;
            if (abs(time-time_max)<=fuzz) time=time_max;
            
        } // end for cycles in calculation
        //------------- Done with calculation ------------------
        
        auto time_2 = std::chrono::high_resolution_clock::now();
        
        auto calc_time = std::chrono::duration_cast
                           <std::chrono::nanoseconds>(time_2 - time_1).count();
        printf("\nCalculation time in seconds: %f \n", calc_time * 1e-9);
        
        // -------------------------------
        //    Print final state to a file
        // -------------------------------
        
        // update the host side to print (i.e., copy from device to host)
        cell_coords.update_host();
        node_energy.update_host();
        
        // write out the intial conditions to a file on the host
        myfile=fopen("timeEnd.txt","w");
        fprintf(myfile,"# x  den  pres  sie vel \n");
        
        // write data on the host side
        for (int cell_id=0; cell_id<num_cells; cell_id++){
           fprintf( myfile,"%f\t%f\t%f\t%f\t%f\n",
                    cell_coords.host(cell_id),
                    node_energy.host(cell_id) );
        }
        fclose(myfile);
        
        
        Kokkos::fence();
        
        // ======== Done using Kokkos ============
        
    } // end of kokkos scope
    Kokkos::finalize();
    
    
    printf("\nfinished\n\n");
    return 0;
  
} // end main function



KOKKOS_INLINE_FUNCTION
int get_corners_in_cell(int cell_gid, int corner_lid){
    // corner_lid is 0 to 1
    return (2*cell_gid + corner_lid);
}

KOKKOS_INLINE_FUNCTION
int get_corners_in_node(int node_gid, int corner_lid){
    // corner_lid is 0 to 1
    return (2*node_gid - 1 + corner_lid);
}





