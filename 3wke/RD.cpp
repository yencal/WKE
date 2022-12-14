//      ./wke --kokkos-num-threads=4

#include "matar.h"
#include <stdio.h>
#include <math.h>  
#include <chrono>
using namespace mtr;

const double fuzz = 1.0E-15;

struct region_t{
   double p_min;
   double p_max;
};

double get_collision_term(real_t Q, DCArrayKokkos <real_t> g_n);


int main(int argc, char* argv[]){
    
    const double time_max = 1.0;
    double       dt       = 0.1;
    const int    num_rk_stages = 2;
    const int    max_cycles = 2000000;

    const double  p_min = 0.0;
    const double  p_max = 100.0;
    const int num_cells = 1000;

    printf("\nstarting code\n");
    
    FILE * myfile;
    
    Kokkos::initialize(argc, argv);
    {
        
        const double integral_grad_basis[2] = {1.0, -1.0};
        
        const int num_nodes = num_cells+1;
        double dp = (p_max-p_min)/num_cells;
        
        double time = 0.0;
    
        DCArrayKokkos <double> node_g(num_nodes);  
        DCArrayKokkos <double> node_g_n(num_nodes);  
        
        DCArrayKokkos <double> cell_coords(num_cells);  
        DCArrayKokkos <double> cell_length(num_cells);   
        
        DCArrayKokkos <double> node_coords(num_nodes);   
        
        
        
        FOR_ALL (node_id, 0, num_nodes, {
           node_coords(node_id) = double(node_id) * dp;
        }); // end parallel for on device

        
        
        FOR_ALL (cell_id, 0, num_cells, {
            cell_coords(cell_id) =
                           0.5*(node_coords(cell_id) + node_coords(cell_id+1));
            
            cell_length(cell_id)  = node_coords(cell_id+1) - node_coords(cell_id);
        }); // end parallel for on device

        
        // set g_0 //
        FOR_ALL (node_id, 0, num_nodes, {
            node_g(node_id) = 0.0;
            node_g_n(node_id) = 0.0;
            if ( p_min <= node_coords(node_id) and node_coords(node_id) <= 100.0){
              node_g_n(node_id) = node_coords(node_id)*(33.33-node_coords(node_id))*(33.33-node_coords(node_id))*(100-node_coords(node_id));// a little energy at lower WNs and more energy at higher WNs  
              node_g_n(node_id) = node_g_n(node_id)/129629629.629; // normalize
            }
            else if ( 100.0 < node_coords(node_id) ){
              node_g_n(node_id) = 0.0;
            }
        }); // end parallel for on device
        
        
        // update the host side to print (i.e., copy from device to host)
        cell_coords.update_host();
        node_g_n.update_host();
        
        // write out the intial conditions to a file on the host
        myfile=fopen("outputs/t0.txt","w");
        fprintf(myfile,"# x  g \n");
        
        // write data on the host side
        for (int cell_id=0; cell_id<num_cells; cell_id++){
          fprintf( myfile,"%f\t%f\n",
                   cell_coords.host(cell_id),
                   0.5*(node_g_n.host(cell_id)+node_g_n.host(cell_id+1)) );
        }
          
        fclose(myfile);
        
        auto time_1 = std::chrono::high_resolution_clock::now();
        
        for (int cycle = 0; cycle<max_cycles; cycle++){
           
            
            for (int rk_stage=0; rk_stage<num_rk_stages; rk_stage++ ){
               
                // rk coefficient on dt
                double rk_alpha = 1.0/
                                     (double(num_rk_stages) - double(rk_stage));


                FOR_ALL (node_id, 0, num_nodes, {
                    double Q = 0.0;
                    get_collision_term(Q, node_g_n);               
                    node_g(node_id) = node_g_n(node_id) + rk_alpha*dt*Q;
                    node_g_n(node_id) = node_g(node_id); 
                }); // end parallel for on device

                Kokkos::fence();
                
                if ((cycle%10000)==0 and cycle!=0){
                  // update the host side to print (i.e., copy from device to host)
                  cell_coords.update_host();
                  node_g.update_host();
        
                  // write out the intial conditions to a file on the host
                  char filename[64];
                  sprintf(filename, "outputs/t%d.txt",cycle);
                  myfile=fopen(filename,"w");
                  fprintf(myfile,"# x  g \n");
        
                  // write data on the host side
                  for (int cell_id=0; cell_id<num_cells; cell_id++){
                    fprintf( myfile,"%f\t%f\n",
                    cell_coords.host(cell_id),
                    0.5*(node_g.host(cell_id) + node_g.host(cell_id+1)) );
                  }
                  fclose(myfile);
                }// end if
            } // end rk loop

            
            // update the host side to print (i.e., copy from device to host)
            cell_coords.update_host();
            node_g.update_host();
        
            myfile=fopen("outputs/tF.txt","a");

            fprintf(myfile,"# x  g \n");
        
            // write data on the host side
            for (int cell_id=0; cell_id<num_cells; cell_id++){
               fprintf( myfile,"%f\t%f\n",
                      cell_coords.host(cell_id),
                      0.5*(node_g.host(cell_id) + node_g.host(cell_id+1)) );
            }
            fclose(myfile);
            
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
        
    } // end of kokkos scope
    Kokkos::finalize();
    
    
    printf("\nfinished\n\n");
    return 0;
  
} // end main function


double get_collision_term(double Q, DCArrayKokkos <real_t> g_n){
    Q = 0.0;
    return Q;
}// end get_collision_term
