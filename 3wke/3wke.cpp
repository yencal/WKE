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
    
    printf("\nstarting code\n");
    
    FILE * myfile;
    
    Kokkos::initialize(argc, argv);
    {
        
        // calculate mesh information based on inputs
        const int num_nodes = num_cells+1;
        double dp = (p_max-p_min)/num_cells;
        
	node_t node;
	elem_t elem;

	// initialize the time to zero
        double time = 0.0;
    
        // calculate nodal coordinates of the mesh
        FOR_ALL (node_id, 0, num_nodes, {
           node.coords(node_id) = double(node_id) * dp;
        }); // end parallel for on device

        
        
        // calculate cell center coordinates of the mesh
        FOR_ALL (cell_id, 0, num_cells, {
            elem.coords(cell_id) =
                           0.5*(node.coords(cell_id) + node.coords(cell_id+1));
            
            elem.length(cell_id)  = node.coords(cell_id+1) - node.coords(cell_id);
        }); // end parallel for on device

        
        // intialize the nodal state that is internal to the mesh
        FOR_ALL (node_id, 1, num_nodes-1, {
            node.energy(1,node_id) = 0.0;// initialize energy at n+1
	    node.energy(0,node_id) = 0.0;// initialize energy at n

	    if ( p_min <= node.coords(node_id) and node.coords(node_id) <= 100.0){
              node.energy(0,node_id) = node.coords(node_id)*(33.33-node.coords(node_id))*(33.33-node.coords(node_id))*(100-node.coords(node_id));// a little energy at lower modes and more energy at higher modes  
	      node.energy(0,node_id) = node.energy(0,node_id)/129629629.629; // normalize
	    }
	    else if ( 100.0 < node.coords(node_id) ){
	      node.energy(0,node_id) = 0.0;
	    }
        }); // end parallel for on device

        
        RUN ({
            
            node.energy(1,0) = 0.0;
            node.energy(1,num_nodes-1) = 0.0;
            node.energy(0,0) = 0.0;
            node.energy(0,num_nodes-1) = 0.0;

        }); // end run once on the device
        
        
        
        // -------------------------------
        //    Print initial state to file
        // -------------------------------
        
        // update the host side to print (i.e., copy from device to host)
        node.energy.update_host();
        
        // write out the intial conditions to a file on the host
        myfile=fopen("time0.txt","w");
        fprintf(myfile,"# x energy \n");
        
        // write data on the host side
        for (int cell_id=0; cell_id<num_cells; cell_id++){
        fprintf( myfile,"%f\t%f\t\n",
                 elem.coords.host(cell_id),
                 node.energy.host(0,cell_id) );
        }
        fclose(myfile);
        
        auto time_1 = std::chrono::high_resolution_clock::now();
        
        // -------------------------------------
        // Solve equations until time=time_max
        // -------------------------------------

	for (int cycle = 0; cycle<max_cycles; cycle++){
           
           /* 
            // get the new time step
            double dt_ceiling = dt*1.1;
            
            // parallel reduction with min
            double dt_lcl;
            double min_dt_calc;
            REDUCE_MIN(cell_id, 0, num_cells, dt_lcl, {
                // mesh size
                double dx_lcl = node.coords(cell_id+1) - node.coords(cell_id);
                
                // local dt calc
                double dt_lcl_ = dt_cfl*dx_lcl/(cell_sspd(cell_id) + fuzz);
                
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
            */
            
            
            // --- integrate forward in time ---
            
            // Runge-Kutta loop
            for (int rk_stage=0; rk_stage<num_rk_stages; rk_stage++ ){
               
                // rk coefficient on dt
                double rk_alpha = 1.0/(double(num_rk_stages) - double(rk_stage));
            
                
                // save the state at t_n
                if (rk_stage==0){
                    
                    // nodal state
                    FOR_ALL (node_id, 0, num_nodes, {
                          node.energy(0, node_id)  = node.energy(1, node_id);
                    }); // end parallel for on device
                    
                    
                } // end if
                
                
                
                // --- RHS --- //
                


                
                // --- Calculate energy update  ---
                
                
                FOR_ALL (node_id, 0, num_nodes, {
                    
                    // update velocity
                    node.energy(1,node_id) = node.energy(0,node_id) -
                                0.0*rk_alpha*dt;
                    
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
        elem.coords.update_host();
        node.energy.update_host();
        
        // write out the intial conditions to a file on the host
        myfile=fopen("timeEnd.txt","w");
        fprintf(myfile,"# p  energy \n");
        
        // write data on the host side
        for (int cell_id=0; cell_id<num_cells; cell_id++){
           fprintf( myfile,"%f\t%f\t\n",
                    elem.coords.host(cell_id),
                    node.energy.host(1,cell_id) );
        }
        fclose(myfile);
        
        Kokkos::fence();
        
        // ======== Done using Kokkos ============
        
    } // end of kokkos scope
    Kokkos::finalize();
    
    
    printf("\nfinished\n\n");
    return 0;
  
} // end main function





