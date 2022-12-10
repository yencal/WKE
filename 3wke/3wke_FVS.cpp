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

double get_collision_term(DCArrayKokkos <double> g_n, const int cell_id, DCArrayKokkos <double> cell_length, DCArrayKokkos <double> p, const int num_cells);


int main(int argc, char* argv[]){
    
    Kokkos::initialize(argc, argv);
    {
        const double time_max = 50.0;
        double       dt       = 0.05;
        const int    num_rk_stages = 2;
        const int    max_cycles = 2000000;

        const double  p_min = 0.0;
        const double  p_max = 100.0;
        const int num_cells = 10*p_max;

        printf("\nstarting code\n");
    
        FILE * myfile;
    
        const int num_nodes = num_cells+1;
        double dp = (p_max-p_min)/num_cells;
        
        double time = 0.0;
    
        DCArrayKokkos <double> node_g(num_nodes);  
        DCArrayKokkos <double> node_g_n(num_nodes);  
        DCArrayKokkos <double> cell_g(num_cells);
        DCArrayKokkos <double> cell_g_n(num_cells); 
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
            node_g_n(node_id) = 0.0;
            
            if ( p_min <= node_coords(node_id) and node_coords(node_id) <= 100.0){
              node_g_n(node_id) = node_coords(node_id)*(33.33-node_coords(node_id))*(33.33-node_coords(node_id))*(100-node_coords(node_id));  
              node_g_n(node_id) = node_g_n(node_id)/129629629.629; 
            }
            else if ( 100.0 < node_coords(node_id) ){
              node_g_n(node_id) = 0.0;
            }
            
            //node_g_n(node_id) = 1.26157*exp(-50.0*(node_coords(node_id)-1.5)*(node_coords(node_id)-1.5) );  
        }); // end parallel for on device
       
        // Trapz approximation of cell average // 
        FOR_ALL (cell_id, 0, num_cells, {
            cell_g(cell_id) = 0.0;
            cell_g_n(cell_id) = 0.0;
            
            cell_g(cell_id) = 0.5*(node_g_n(cell_id) + node_g_n(cell_id+1));
            
            cell_g_n(cell_id) = 0.5*(node_g_n(cell_id) + node_g_n(cell_id+1));
        });
        
        // update the host side to print (i.e., copy from device to host)
        cell_coords.update_host();
        cell_g_n.update_host();
        
        // write out the intial conditions to a file on the host
        myfile=fopen("outputs/t0.txt","w");
        fprintf(myfile,"# x  g \n");
        
        // write data on the host side
        for (int cell_id=0; cell_id<num_cells; cell_id++){
          fprintf( myfile,"%f\t%f\n",
                   cell_coords.host(cell_id),
                   cell_g_n.host(cell_id) );
        }
          
        fclose(myfile);
        
        auto time_1 = std::chrono::high_resolution_clock::now();
        
        for (int cycle = 0; cycle<max_cycles; cycle++){
            
            std::cout << "cycle " << cycle << std::endl; 
            std::cout << " t = " << time << std::endl;  
            for (int rk_stage=0; rk_stage<num_rk_stages; rk_stage++ ){
                 
		 if (rk_stage==0){

                   FOR_ALL (cell_id, 0, num_cells, {
                     cell_g_n(cell_id) = cell_g(cell_id); 
		   }); // end parallel for on device
                 
                   Kokkos::fence();
		 };              
                // rk coefficient on dt
                double rk_alpha = 1.0/(double(num_rk_stages) - double(rk_stage));


                FOR_ALL (cell_id, 0, num_cells, {
                //for (int cell_id = 0; cell_id < num_cells; cell_id++){
                    double dQdp = 0.0;
                    dQdp = get_collision_term( cell_g_n, cell_id, cell_length, cell_coords, num_cells);               
                    
                    //Kokkos::fence();
                    cell_g(cell_id) = cell_g_n(cell_id) + cell_coords(cell_id)*dt/dp*rk_alpha*dQdp;
                    
                //}// end loop over cell_id
                }); // end parallel for on device


                Kokkos::fence();
                

                //if ((cycle%100)==0 and cycle!=0){
                  // update the host side to print (i.e., copy from device to host)
                  cell_coords.update_host();
                  cell_g.update_host();
        
                  // write out the intial conditions to a file on the host
                  char filename[64];
                  sprintf(filename, "outputs/t%d.txt",cycle);
                  myfile=fopen(filename,"w");
                  fprintf(myfile,"# x  g \n");
        
                  // write data on the host side
                  for (int cell_id=0; cell_id<num_cells; cell_id++){
                    fprintf( myfile,"%f\t%f\n",
                    cell_coords.host(cell_id),
                    cell_g.host(cell_id) );
                  }
                  fclose(myfile);
                //}// end if

            } // end rk loop

            
            // update the host side to print (i.e., copy from device to host)
            cell_coords.update_host();
            cell_g.update_host();
        
            // update the time
            time += dt;
            if (time>=time_max) break;
            
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
    printf("\nfinished\n\n");
    Kokkos::finalize();
    return 0;
  
} // end main function


double get_collision_term( DCArrayKokkos <double> g_n, int cell_id, DCArrayKokkos <double> cell_length, DCArrayKokkos <double> p, int num_cells){
    
    double gamma = 2.0;
    
    double Q1_right = 0.0;       
    double Q2_right = 0.0;
    double Q1_left = 0.0;       
    double Q2_left = 0.0;
    
    
    //FOR_ALL (index, 0, cell_id+1, {
    for (int index = 0; index < cell_id+1; index++){
      
      int lower_index = cell_id+1 - index;
      double Q11_right = 0.0;
      double Q21_right = 0.0;
      for (int k = lower_index; k < cell_id; k++){
        Q11_right += cell_length(k)*(g_n(k)/p(k))*std::pow(p(k)*p(index),gamma*0.5) ;
      }// end loop over k 

      for (int k = lower_index; k < num_cells; k++){
        Q21_right += cell_length(k)*(g_n(k)/p(k))*std::pow(p(k)*p(index),gamma*0.5) ;
      }// end loop over k 
      
      Q1_right += cell_length(index)*(g_n(index)/p(index)) * Q11_right; 
      Q2_right += cell_length(index)*(g_n(index)/p(index)) * Q21_right; 
         
    //}); // end parallel for on device
    }// end loop over index 
    //Kokkos::fence();
    
    //FOR_ALL (index, 0, cell_id, {
    for (int index = 0; index < cell_id; index++){
      
      int lower_index = cell_id - index;
      double Q11_left = 0.0;
      double Q21_left = 0.0;
      for (int k = lower_index; k < cell_id; k++){
        Q11_left += cell_length(k)*(g_n(k)/p(k))*std::pow(p(k)*p(index),gamma*0.5);
      }// end loop over k 

      for (int k = lower_index; k < num_cells; k++){
        Q21_left += cell_length(k)*(g_n(k)/p(k))*std::pow(p(k)*p(index),gamma*0.5);
      }// end loop over k 

      Q1_left += cell_length(index)*(g_n(index)/p(index)) * Q11_left; 
      Q2_left += cell_length(index)*(g_n(index)/p(index)) * Q21_left;
         
    //}); // end parallel for on device
    }// end loop over index 
    //Kokkos::fence();

    return Q2_right-2.0*Q1_right - Q2_left + 2.0*Q1_left;
}// end get_collision_term
