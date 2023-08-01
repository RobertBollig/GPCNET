/*
 * The entirety of this work is licensed under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 *
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <network_test.h>

#define VERSION 1.3

int RSEED=-10;

int fileSync=0;
int allTest=0;
int allCong=0;
int debugLock=0;
int NUM_CONGESTOR_TESTS=4;
int verbose=0;
int tblsize=80;

#define NUM_NETWORK_TESTS 3
/*#define NUM_CONGESTOR_TESTS 4 */

/* loop counts for the various tests */
#define NUM_LAT_TESTS 10000
#define NUM_LAT_RANDS 30
#define NUM_LAT_ITERS 200
#define NUM_BW_TESTS 10000
#define NUM_BW_RANDS 30
#define NUM_BW_ITERS 8
#define NUM_ALLREDUCE_TESTS 100000
#define NUM_ALLREDUCE_ITERS 200

/* test specific tuning */
#define BW_MSG_COUNT 16384
#define BW_OUTSTANDING 8

/* define where baseline sizes for congestor tests are */
#define A2A_BASE_NODES 32
#define INCAST_BASE_NODES 32
#define BCAST_BASE_NODES 32
#define ALLREDUCE_BASE_NODES 32

/* tuning for congestor tests */
#define A2A_MSG_COUNT 512
#define A2A_TESTS 256
#define A2A_BASELINE_ITERS 512
#define ALLREDUCE_MSG_COUNT 819200
#define ALLREDUCE_TESTS 256
#define ALLREDUCE_BASELINE_ITERS 512
#define INCAST_MSG_COUNT 512
#define INCAST_TESTS 256
#define INCAST_BASELINE_ITERS 512
#define BCAST_MSG_COUNT 512
#define BCAST_TESTS 256
#define BCAST_BASELINE_ITERS 512

#define CONGESTOR_NODE_FRAC 0.8

CommTest_t network_tests_list[6], congestor_tests_list[6];

typedef struct CongestorResults_st {
     double *baseline_perf_hires;
     double *perf_hires;
     double baseline_perf;
     double perf;
} CongestorResults_t;

int network_test_setup(CommTest_t req_test, int *ntests, int *nrands, int *niters, char *tname, char *tunits)
{
     int nl = 64;

     switch (req_test) {
     case P2P_LATENCY:
          *ntests = NUM_LAT_TESTS;
          *nrands = NUM_LAT_RANDS;
          *niters = NUM_LAT_ITERS;
          snprintf(tname, nl, "%s (8 B)", "RR Two-sided Lat");
          snprintf(tunits, nl, "%s", "usec");
          break;
     case RMA_LATENCY:
          *ntests = NUM_LAT_TESTS;
          *nrands = NUM_LAT_RANDS;
          *niters = NUM_LAT_ITERS;
          snprintf(tname, nl, "%s (8 B)", "RR Get Lat");
          snprintf(tunits, nl, "%s", "usec");
          break;
     case P2P_BANDWIDTH:
          *ntests = NUM_BW_TESTS;
          *nrands = NUM_BW_RANDS;
          *niters = NUM_BW_ITERS;
          snprintf(tname, nl, "%s (%4i B)", "RR Two-sided BW", (int)(sizeof(double)*BW_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case RMA_BANDWIDTH:
          *ntests = NUM_BW_TESTS;
          *nrands = NUM_BW_RANDS;
          *niters = NUM_BW_ITERS;
          snprintf(tname, nl, "%s (%4i B)", "RR Put BW", (int)(sizeof(double)*BW_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case P2P_NEIGHBORS:
          *ntests = NUM_BW_TESTS;
          *nrands = NUM_BW_RANDS;
          *niters = NUM_BW_ITERS;
          snprintf(tname, nl, "%s (%4i B)", "RR Two-sided BW+Sync", (int)(sizeof(double)*BW_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case ALLREDUCE_LATENCY:
          *ntests = NUM_ALLREDUCE_TESTS;
          *nrands = 1;
          *niters = NUM_ALLREDUCE_ITERS;
          snprintf(tname, nl, "%s (8 B)", "Multiple Allreduce");
          snprintf(tunits, nl, "%s", "usec");
          break;
     default:
          break;
     }

     return 0;
}

int run_network_tests(CommConfig_t *config, int ncong_tests, MPI_Comm test_comm,
                      MPI_Comm local_comm, MPI_Comm global_comm)
{
     int itest, ctest, myrank, niters, ntests, nrands;
     CommResults_t *results_base, *results, nullrslt;
     MPI_Request req;
     int nl=64;
     char tname[nl], tunits[nl];

     mpi_error(MPI_Comm_rank(local_comm, &myrank));

     results_base = malloc(sizeof(CommResults_t) * NUM_NETWORK_TESTS);
     results      = malloc(sizeof(CommResults_t) * NUM_NETWORK_TESTS);
     if (results_base == NULL || results == NULL) {
          die("Failed to allocate results in run_network_tests()\n");
     }

     /* gather the baseline performance */
     print_header(config, 0, 0);
     for (itest = 0; itest < NUM_NETWORK_TESTS; itest++) {

          results_base[itest].distribution = NULL;
          network_test_setup(network_tests_list[itest], &ntests, &nrands, &niters, tname, tunits);
          if (network_tests_list[itest] != ALLREDUCE_LATENCY) {
               random_ring(config, 0, ntests, nrands, niters, network_tests_list[itest],
                           TEST_NULL, test_comm, local_comm, &results_base[itest]);
          } else {
               allreduce_test(config, ntests, niters, test_comm, local_comm, &results_base[itest]);
          }

          int from_min = 0;
          if (network_tests_list[itest] == P2P_LATENCY || network_tests_list[itest] == RMA_LATENCY ||
              network_tests_list[itest] == ALLREDUCE_LATENCY) from_min = 1;
          print_results(config, myrank, 1, from_min, tname, tunits, &results_base[itest]);

          if (myrank == 0) {
               write_distribution(network_tests_list[itest], TEST_NULL, 1, &results_base[itest], tname, tunits);
          }

     }

     /* barrier to prevent congestors from communicating while we get baselines */
     if (myrank == 0 && fileSync) {createLock("congestorRelease");}
     mpi_error(MPI_Barrier(global_comm)); /* Barrier 1 */

     /* allow congestor tests to get their baselines */
     if (verbose) {
         print_header(config, 1, 0);
         for (ctest = 0; ctest < ncong_tests; ctest++) {
              print_results(config, myrank, 0, 0, " ", " ", &nullrslt);
         }
     }
     if (myrank == 0 && fileSync) {waitLock("testerRelease");}
     mpi_error(MPI_Barrier(global_comm)); /* Barrier 2 */

     /* now loop over tests where we run congestor tests at the same time */
     if (!verbose) {
         print_header(config, 2, network_tests_list[itest]);
     }
     for (itest = 0; itest < NUM_NETWORK_TESTS; itest++) {

          results[itest].distribution = NULL;
          if (verbose) {
              print_header(config, 2, network_tests_list[itest]);
          }

          /* allow the congestors to get started and then release us once they are loading the network */
          if (fileSync) {
              if (myrank == 0) {waitLock("warmupCompleteRelease");}
              mpi_error(MPI_Barrier(global_comm)); /* Barrier 3 (Warm-up finished) */
          } else { /* Why is this even non-blocking if you just call a MPI_WAIT directly after it? */
              mpi_error(MPI_Ibarrier(global_comm, &req)); /* Barrier 3 (Warm-up finished) */
              mpi_error(MPI_Wait(&req, MPI_STATUS_IGNORE));
          }

          network_test_setup(network_tests_list[itest], &ntests, &nrands, &niters, tname, tunits);
          if (network_tests_list[itest] != ALLREDUCE_LATENCY) {
               random_ring(config, 0, ntests, nrands, niters, network_tests_list[itest],
                           TEST_CONGESTORS, test_comm, local_comm, &results[itest]);
          } else {
               allreduce_test(config, ntests, niters, test_comm, local_comm, &results[itest]);
          }

          /* now let the congestor tests know we are done */
          if (fileSync) {
              if (myrank == 0) {createLock("testdoneRelease");}
              mpi_error(MPI_Barrier(global_comm)); /* Barrier 4 (Test finished) */
          } else { /* Why is this even non-blocking if you just call a MPI_WAIT directly after it? */
              mpi_error(MPI_Ibarrier(global_comm, &req)); /* Barrier 4 (Test finished) */
              mpi_error(MPI_Wait(&req, MPI_STATUS_IGNORE));
          }

          int from_min = 0;
          if (network_tests_list[itest] == P2P_LATENCY || network_tests_list[itest] == RMA_LATENCY ||
              network_tests_list[itest] == ALLREDUCE_LATENCY) from_min = 1;
          print_results(config, myrank, 1, from_min, tname, tunits, &results[itest]);

          if (myrank == 0) {
               write_distribution(network_tests_list[itest], TEST_CONGESTORS, 0, &results[itest], tname, tunits);
          }

          /* allow congestors to print out results */
          if (verbose) {
              for (ctest = 0; ctest < ncong_tests; ctest++) {
                   print_results(config, myrank, 0, 0, " ", " ", &nullrslt);
              }
          }

     }

     /* now print the final results table */
     print_header(config, 3, 0);
     for (itest = 0; itest < NUM_NETWORK_TESTS; itest++) {

          if (network_tests_list[itest] == P2P_LATENCY || network_tests_list[itest] == P2P_NEIGHBORS ||
              network_tests_list[itest] == ALLREDUCE_LATENCY) {
               network_test_setup(network_tests_list[itest], &ntests, &nrands, &niters, tname, tunits);

               if (network_tests_list[itest] == P2P_LATENCY || network_tests_list[itest] == ALLREDUCE_LATENCY) {
                    print_comparison_results(config, myrank, 1, 1, tname, &results_base[itest], &results[itest]);
               } else {
                    print_comparison_results(config, myrank, 1, 0, tname, &results_base[itest], &results[itest]);
               }
          }
     }

     return 0;
}

int congestor_test_setup(CommTest_t req_test, int a2aiters, int incastiters, int bcastiters, int allreduceiters,
                         int a2atests, int incasttests, int bcasttests, int allreducetests,
                         int *ntests, int *niters, char *tname, char *tunits)
{
     int nl=64;

     switch (req_test) {
     case A2A_CONGESTOR:
          *ntests = a2atests;
          *niters = a2aiters;
          snprintf(tname, nl, "%s (%i B)", "Alltoall", (int)(sizeof(double)*A2A_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case ALLREDUCE_CONGESTOR:
          *ntests = allreducetests;
          *niters = allreduceiters;
          snprintf(tname, nl, "%s (%i B)", "Allreduce", (int)(sizeof(double)*ALLREDUCE_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case P2P_INCAST_CONGESTOR:
          *ntests = incasttests;
          *niters = incastiters;
          snprintf(tname, nl, "%s (%i B)", "Two-sided Incast", (int)(sizeof(double)*INCAST_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case RMA_INCAST_CONGESTOR:
          *ntests = incasttests;
          *niters = incastiters;
          snprintf(tname, nl, "%s (%i B)", "Put Incast", (int)(sizeof(double)*INCAST_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case P2P_BCAST_CONGESTOR:
          *ntests = bcasttests;
          *niters = bcastiters;
          snprintf(tname, nl, "%s (%i B)", "Two-sided Bcast", (int)(sizeof(double)*BCAST_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case RMA_BCAST_CONGESTOR:
          *ntests = bcasttests;
          *niters = bcastiters;
          snprintf(tname, nl, "%s (%i B)", "Get Bcast", (int)(sizeof(double)*BCAST_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     default:
          break;
     }

     return 0;
}

int run_congestor_tests(CommConfig_t *config, int nntwk_tests, int congestor_set, MPI_Comm congestor_comm,
                        MPI_Comm test_comm, MPI_Comm local_comm, MPI_Comm global_comm)
{
     int i, iteration, itest, ibar_cmplt, test_cmplt, ctest, niters, ntests, real_ntests, measure_perf;
     int a2aiters, allreduce_iters, incast_iters, bcast_iters;
     int myrank, nranks, test_myrank, test_nranks, cong_myrank, cong_nranks;
     CongestorResults_t congestor_perf;
     CommResults_t results, nullrslt;
     MPI_Request req, areq;
     double *sbuf, *rbuf;
     int nl = 64;
     char tname[nl], tunits[nl];

     mpi_error(MPI_Comm_rank(congestor_comm, &cong_myrank));
     mpi_error(MPI_Comm_size(congestor_comm, &cong_nranks));
     mpi_error(MPI_Comm_rank(local_comm, &myrank));
     mpi_error(MPI_Comm_size(local_comm, &nranks));
     mpi_error(MPI_Comm_rank(test_comm, &test_myrank));
     mpi_error(MPI_Comm_size(test_comm, &test_nranks));

     /* determine the number of iterations per pass for a2a from a base setting */
     a2aiters = A2A_BASELINE_ITERS;
     if (cong_nranks > A2A_BASE_NODES) {
          a2aiters = (int)((float)a2aiters * (float)A2A_BASE_NODES / (float)cong_nranks);
          a2aiters = (a2aiters < 1) ? 1 : a2aiters;
     }

     /* we roughly assume the allreduce algorithm behaves like a ring or recursive halving /
        recursive doubling in that the amount of data any rank sends saturates at 2 the message
        length.  In reality, message size drops per op and there is more overhead with scale.
        As a crude model we'll put a weak reduction in iters to account for this. */
     allreduce_iters = ALLREDUCE_BASELINE_ITERS;
     if (cong_nranks > ALLREDUCE_BASE_NODES) {
          allreduce_iters = (int)((float)allreduce_iters / (1.0 + log((float)cong_nranks / (float)INCAST_BASE_NODES)));
          allreduce_iters = (allreduce_iters < 1) ? 1 : allreduce_iters;
     }

     /* reduce iters for incast linearly */
     incast_iters = INCAST_BASELINE_ITERS;
     if (cong_nranks > INCAST_BASE_NODES) {
          incast_iters = (int)((float)incast_iters * (float)INCAST_BASE_NODES / (float)cong_nranks);
          incast_iters = (incast_iters < 1) ? 1 : incast_iters;
     }

     /* reduce iters for bcast linearly */
     bcast_iters = BCAST_BASELINE_ITERS;
     if (cong_nranks > BCAST_BASE_NODES) {
          incast_iters = (int)((float)bcast_iters * (float)BCAST_BASE_NODES / (float)cong_nranks);
          incast_iters = (bcast_iters < 1) ? 1 : incast_iters;
     }

     /* allocate space to the various performance tracking arrays */
     congestor_test_setup(congestor_tests_list[congestor_set], a2aiters, incast_iters, bcast_iters,
                          allreduce_iters, A2A_TESTS, INCAST_TESTS, BCAST_TESTS, ALLREDUCE_TESTS, &ntests,
                          &niters, tname, tunits);
     congestor_perf.baseline_perf_hires = malloc(sizeof(double) * ntests * niters);
     congestor_perf.perf_hires          = malloc(sizeof(double) * ntests * niters);
     if (congestor_perf.baseline_perf_hires == NULL || congestor_perf.perf_hires == NULL) {
          die("Failed to allocate perf_hires in run_congestor_tests()\n");
     }

     /* allocate the a2a buffers as needed */
     if (congestor_tests_list[congestor_set] == A2A_CONGESTOR ||
         congestor_tests_list[congestor_set] == P2P_INCAST_CONGESTOR) {
          a2a_buffers(config, congestor_comm);
     } else if (congestor_tests_list[congestor_set] == ALLREDUCE_CONGESTOR) {
         allreduce_buffers(config, congestor_comm);
     } else if (congestor_tests_list[congestor_set] == RMA_INCAST_CONGESTOR) {
          init_rma_a2a(config, congestor_comm);
     }

     /* barrier to let regular network tests get their baselines */
     if (!allCong) { /* Only mixed Tester/Congestors print output */
         print_header(config, 0, 0);
         for (itest = 0; itest < nntwk_tests; itest++) {
              print_results(config, myrank, 0, 0, " ", " ", &nullrslt);
         }
     }
     if (myrank == 0 && fileSync) {waitLock("congestorRelease");}
     mpi_error(MPI_Barrier(global_comm)); /* Barrier 1 */

     /* loop over the congestor test and gather those baselines */
     results.distribution = NULL;
     if (verbose) {
         print_header(config, 1, 0);
         for (ctest = 0; ctest < NUM_CONGESTOR_TESTS; ctest++) {

              if (ctest == congestor_set) {

                   congestor_test_setup(congestor_tests_list[ctest], a2aiters, incast_iters, bcast_iters,
                                        allreduce_iters, A2A_TESTS, INCAST_TESTS, BCAST_TESTS, ALLREDUCE_TESTS,
                                        &ntests, &niters, tname, tunits);

                   congestor(config, ntests, niters, congestor_comm, congestor_tests_list[ctest], 1,
                             congestor_perf.baseline_perf_hires, &congestor_perf.baseline_perf, &real_ntests);

                   summarize_performance(config, congestor_perf.baseline_perf_hires,
                                         &congestor_perf.baseline_perf, real_ntests*niters, 1,
                                         0, test_comm, &results);

                   if (!allCong) { /* Only mixed Tester/Congestors print output */
                       print_results(config, test_myrank, 1, 0, tname, tunits, &results);
                       if (test_myrank == 0) {
                            write_distribution(congestor_tests_list[ctest], TEST_NULL, 1, &results, tname, tunits);
                       }
                   }
                   mpi_error(MPI_Barrier(local_comm));

              } else {

                   /* let the ranks in the given congestor test run in isolation */
                   if (!allCong) { /* Only mixed Tester/Congestors print output */
                      print_results(config, test_myrank, 0, 0, " ", " ", &nullrslt);
                   }
                   mpi_error(MPI_Barrier(local_comm));

              }

         }
     }

     /* sync with the network tests */
     if (myrank == 0 && fileSync) {createLock("testerRelease");}
     mpi_error(MPI_Barrier(global_comm)); /* Barrier 2 */

     /* loop over the combined tests */
     if (!verbose) {
         if (!allCong) { /* Only mixed Tester/Congestors print output */
            print_header(config, 2, network_tests_list[itest]);
         }
     }
     for (itest = 0; itest < nntwk_tests; itest++) {

          if (verbose) {
              print_header(config, 2, network_tests_list[itest]);
          }
          ctest = congestor_set;

          congestor_test_setup(congestor_tests_list[ctest], a2aiters, incast_iters, bcast_iters, allreduce_iters,
                               A2A_TESTS, INCAST_TESTS, BCAST_TESTS, ALLREDUCE_TESTS, &ntests,
                               &niters, tname, tunits);

          test_cmplt = 0;
          iteration  = 0;
          ibar_cmplt = 0;
          areq       = MPI_REQUEST_NULL;
          while (test_cmplt == 0) {

               /* only measure the performance on the first iteration */
               measure_perf = (iteration == 0) ? 1 : 0;

               /* if we have done a warmup iteration, let's complete the ibarrier
                  that releases the network tests to start */
               if (iteration == 1) {
                   if (!fileSync) mpi_error(MPI_Wait(&req, MPI_STATUS_IGNORE));
               }

               congestor(config, ntests, niters, congestor_comm, congestor_tests_list[ctest], measure_perf,
                         congestor_perf.perf_hires, &congestor_perf.perf, &real_ntests);

               /* we have done a non-warmup test so we can start our ibarrier */
               if (iteration == 1) {
                    if (!fileSync) mpi_error(MPI_Ibarrier(global_comm, &req)); /* Barrier 4 (Test finished) */
               }

               /* if we have started our ibarrier, use it to check if the network test
                  has finished yet.  must be consistent within our congestion test
                  communicator */
                if (fileSync && iteration >= 3) { /* Allow for more iterations before checking if test is complete to avoid filelocking overhead */
                    /* checkLock returns 1 if the file is found, 0 if file is not found */
                    if (myrank == 0) {ibar_cmplt=checkLock("testdoneRelease");}
                    mpi_error(MPI_Bcast(&ibar_cmplt,1,MPI_INT,0,local_comm)); /* Broadcast completion to all congestors */
                    if (ibar_cmplt) {
                        /* Wait until all congestor nodes agree that the testers are done */
                        if (myrank == 0) {waitLock("testdoneRelease");}
                        test_cmplt = 1;
                    }
                } else if (!fileSync && iteration >=1){
                    /* wait on any outstanding reduction */
                    if (areq != MPI_REQUEST_NULL) {
                         mpi_error(MPI_Wait(&areq, MPI_STATUS_IGNORE));

                         /* if the network test is done we can exit this loop */
                         if (ibar_cmplt) test_cmplt = 1;
                    }

                    /* the decision that the network tests are done must be consistent across our congestor comms */
                    if (! test_cmplt) {
                         if (req != MPI_REQUEST_NULL) {
                              mpi_error(MPI_Test(&req, &ibar_cmplt, MPI_STATUS_IGNORE));
                         } else {
                              ibar_cmplt = 1;
                         }
                         mpi_error(MPI_Iallreduce(MPI_IN_PLACE, &ibar_cmplt, 1, MPI_INT, MPI_MIN, test_comm, &areq));
                    }
                }

               /* we have done warmup comms so let's release the network tests */
               if (iteration == 0) {
                    if (fileSync) {
                        if (myrank == 0) {createLock("warmupCompleteRelease");}
                        mpi_error(MPI_Barrier(global_comm)); /* Barrier 3 (Warm-up finished) */
                    } else {
                        mpi_error(MPI_Ibarrier(global_comm, &req)); /* Barrier 3 (Warm-up finished) */
                    }
               }
               iteration++;
          }

          /* print out the network test results */
          if (!allCong) { /* Only mixed Tester/Congestors print output */
               print_results(config, myrank, 0, 0, " ", " ", &nullrslt);
          }

          /* we must delay performance summary until after the network tests is done otherwise
             we introduce a stall on congesting traffic from the reductions done here */
          summarize_performance(config, congestor_perf.perf_hires, &congestor_perf.perf, real_ntests*niters, 1,
                                0, test_comm, &results);
          if (test_myrank == 0) {
               write_distribution(congestor_tests_list[ctest], network_tests_list[itest], 0, &results, tname, tunits);
          }

          for (ctest = 0; ctest < NUM_CONGESTOR_TESTS; ctest++) {
               if (verbose) {
                    if (ctest == congestor_set) {
                         print_results(config, test_myrank, 1, 0, tname, tunits, &results);
                    } else {
                         print_results(config, test_myrank, 0, 0, " ", " ", &nullrslt);
                    }
               }
               mpi_error(MPI_Barrier(local_comm)); /* Local barrier of only congestor nodes */
          }
     }

     /* now print the final results table */
     if (!allCong) { /* Only mixed Tester/Congestors print output */
         print_header(config, 3, 0);
         for (itest = 0; itest < nntwk_tests; itest++) {

              if (network_tests_list[itest] == P2P_LATENCY || network_tests_list[itest] == ALLREDUCE_LATENCY) {
                   print_comparison_results(config, myrank, 0, 0, " ", &nullrslt, &nullrslt);
              } else if (network_tests_list[itest] == P2P_NEIGHBORS) {
                   print_comparison_results(config, myrank, 0, 0, " ", &nullrslt, &nullrslt);
              }
         }
     }

     free(congestor_perf.baseline_perf_hires);
     free(congestor_perf.perf_hires);

     return 0;
}

int build_subcomms(int ncongestor_tests, CommConfig_t *config, CommNodes_t *nodes, int *am_congestor, int *congestor_set,
                   MPI_Comm *congestor_comm, MPI_Comm *test_comm, MPI_Comm *local_comm, int *nt_nodes, int *nc_nodes)
{
     int i, j, ncongestor_nodes, ntest_nodes, gmyrank;
     int *allnodes, *congestor_nodes, *test_nodes;
     MPI_Comm null_comm, tmp_comm;
     FILE *fp;

     switch(*am_congestor) {
     case 1: /*All nodes are going to be congestors*/
         ncongestor_nodes = nodes->nnodes;
         break;
     case 0: /*All nodes are going to be testers*/
         ncongestor_nodes = 0;
         break;
     default: /*Default tester and congestor split*/
         /* default behavior */
         ncongestor_nodes = (int)floor((double)(CONGESTOR_NODE_FRAC * nodes->nnodes));
         /* Make sure that there at least two test nodes
          * Consistency with NUM_CONGESTOR_TESTS is checked later
          */
         if ((nodes->nnodes - ncongestor_nodes) < 2) {ncongestor_nodes = nodes->nnodes - 2;}
         break;
     }

     ntest_nodes      = nodes->nnodes - ncongestor_nodes;
     *nt_nodes        = ntest_nodes;
     *nc_nodes        = ncongestor_nodes;

     if (allTest) {
         if (ntest_nodes < 2) {
              if (config->myrank == 0) {
                   printf("ERROR: this tester partition must be run on at least %i nodes\n",2);
              }
              MPI_Finalize();
              exit(1);
         }
     } else if (allCong) {
         if (ncongestor_nodes < 2*NUM_CONGESTOR_TESTS) {
              if (config->myrank == 0) {
                   printf("ERROR: this Congestor partition must be run on at least %i nodes\n",(2*NUM_CONGESTOR_TESTS));
              }
              MPI_Finalize();
              exit(1);
         }
     } else {
         if (ntest_nodes < 2 || ncongestor_nodes < 2*NUM_CONGESTOR_TESTS) {
              if (config->myrank == 0) {
                   printf("ncongestor_nodes=%d\n",ncongestor_nodes);
                   printf("ntest_nodes=%d\n",ntest_nodes);
                   printf("ERROR: this application must be run on at least %i nodes\n",2+(2*NUM_CONGESTOR_TESTS));
              }
              MPI_Finalize();
              exit(1);
         }
     }

     allnodes        = malloc(sizeof(int) * nodes->nnodes);
     congestor_nodes = malloc(sizeof(int) * ncongestor_nodes);
     test_nodes      = malloc(sizeof(int) * ntest_nodes);
     if (allnodes == NULL || congestor_nodes == NULL || test_nodes == NULL) {
          die("Failed to allocate node lists in build_subcomms()\n");
     }

     /* generate a shuffled list of all nodes we will draw from.  the purpose is to prevent rank
        reordering from improving the test results and to mimic non-ideal node selection from the WLM
        observed on busy systems.  think of network tests and congestors all as different apps running
        at the same time in different allocations */
     for (i = 0; i < nodes->nnodes; i++) {
          allnodes[i] = i;
     }

     time_t now = time(NULL);
     MPI_Bcast(&now, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
     int seed = (int)now;

     /* to use a fixed seed for debug purposes, uncomment the following line */
     if (RSEED !=-10) {seed = RSEED;}
     shuffle(allnodes, nodes->nnodes, seed, 0);

     /* pull out the different node types */
     for (i = 0; i < ncongestor_nodes; i++) {
          congestor_nodes[i] = allnodes[i];
     }
     j = ncongestor_nodes;
     for (i = 0; i < ntest_nodes; i++) {
          test_nodes[i] = allnodes[i+j];
     }

     congestion_subcomms(config, nodes, congestor_nodes, ncongestor_nodes, am_congestor, local_comm);
     if (*am_congestor) {
          node_slice_subcomms(config, nodes, congestor_nodes, ncongestor_nodes, &tmp_comm);
     } else {
          node_slice_subcomms(config, nodes, congestor_nodes, ncongestor_nodes, &null_comm);
     }
     if (*am_congestor) {
          node_slice_subcomms(config, nodes, test_nodes, ntest_nodes, &null_comm);
     } else {
          node_slice_subcomms(config, nodes, test_nodes, ntest_nodes, test_comm);
     }

     /* congestors now further slice up their communicator into that for the different congestor tests */
     if (*am_congestor) {
          split_subcomms(ncongestor_tests, *local_comm, tmp_comm, congestor_set, test_comm, congestor_comm);
     }

     MPI_Comm_rank(MPI_COMM_WORLD, &gmyrank);
     if (gmyrank == 0) {
         if (!allTest && !allCong) {
              fp = fopen("allnodes_congestion.dat", "w+");
              for (i = 0; i < nodes->nnodes; i++) {
                   CommNode_t *tmp = nodes->nodes_head;
                   for (j = 0; j < nodes->nnodes; j++) {
                        if (tmp->node_id == allnodes[i]) {
                             fprintf(fp, "%-32s\n", tmp->host_name);
                        }
                        tmp = tmp->next;
                   }
              }
              fclose(fp);
         }
         if (!allTest) {
              fp = fopen("congestors_nodes_congestion.dat", "w+");
              for (i = 0; i < ncongestor_nodes; i++) {
                   CommNode_t *tmp = nodes->nodes_head;
                   for (j = 0; j < nodes->nnodes; j++) {
                        if (tmp->node_id == congestor_nodes[i]) {
                             fprintf(fp, "%-32s\n", tmp->host_name);
                        }
                        tmp = tmp->next;
                   }
              }
              fclose(fp);
         }
         if (!allCong) {
              fp = fopen("network_nodes_congestion.dat", "w+");
              for (i = 0; i < ntest_nodes; i++) {
                   CommNode_t *tmp = nodes->nodes_head;
                   for (j = 0; j < nodes->nnodes; j++) {
                        if (tmp->node_id == test_nodes[i]) {
                             fprintf(fp, "%-32s\n", tmp->host_name);
                        }
                        tmp = tmp->next;
                   }
              }
              fclose(fp);
         }
     }

     free(congestor_nodes);
     free(test_nodes);
     free(allnodes);

     return 0;
}

int main(int argc, char* argv[])
{
     CommConfig_t test_config;
     CommNodes_t nodes;
     MPI_Comm local_comm, test_comm, congestor_comm;
     int i, am_congestor=-10, congestor_set, nt_nodes, nc_nodes;

     init_mpi(&test_config, &nodes, &argc, &argv, BW_MSG_COUNT, BW_MSG_COUNT, A2A_MSG_COUNT,
                   INCAST_MSG_COUNT, BCAST_MSG_COUNT, ALLREDUCE_MSG_COUNT, BW_OUTSTANDING);

     /* Set up the random seed from environment variables */
     if (getenv("RSEED")) {
         RSEED=atoi(getenv("RSEED"));
         if (test_config.myrank == 0) {
             printf("RSEED is taken from environment.\n");
             printf("RSEED=%d\n\n", RSEED);
         }
     } else {
         if (test_config.myrank == 0) {printf("Random seed is truly random.\n\n");}
     }

     /* Set up whether filelocks print debug statements */
     if (getenv("DEBUGLOCK")) {
         debugLock=atoi(getenv("DEBUGLOCK"));
     }

     /* Set up the test setup from environment variables */
     if (getenv("CONGESTOR")) {
         am_congestor=atoi(getenv("CONGESTOR"));
         if (test_config.myrank == 0) {printf("Test partitioning is taken from environment\n");}
         switch(am_congestor) {
         case 1:
             if (test_config.myrank == 0) {
                 printf("All nodes are going to be congestors.\n");
                 printf("Test Syncing will be done by a lock file.\n\n");
             }
             fileSync=1; allCong=1;
             break;
         case 0:
             if (test_config.myrank == 0) {
                 printf("All nodes are going to be testers.\n");
                 printf("Test Syncing will be done by a lock file.\n\n");
             }
             fileSync=1; allTest=1;
             break;
         default:
             if (test_config.myrank == 0) {printf("Default test partitioning\n\n");}
             break;
         }
     } else {
         if (test_config.myrank == 0) {printf("Default test partitioning\n\n");}
     }

     /* Set up verbosity*/
     if (getenv("VERBOSE")) {
         verbose=atoi(getenv("VERBOSE"));
         if (verbose) {
             tblsize=VTBLSIZE;
             if (test_config.myrank == 0) {
                 printf("Verbose output is enabled.\n\n");
             }
         }
     }

     /* Set up the number of congestor tests from environment variables */
     if (getenv("NUM_CONGESTOR_TESTS")) {
         NUM_CONGESTOR_TESTS=atoi(getenv("NUM_CONGESTOR_TESTS"));
         if (test_config.myrank == 0) {
             printf("NUM_CONGESTOR_TESTS is taken from environment\n");
             if (NUM_CONGESTOR_TESTS > 6 || NUM_CONGESTOR_TESTS < 1) {
                 printf("NUM_CONGESTOR_TESTS has to be in the range of [1-6].\n");
                 MPI_Finalize();
                 exit(1);
             }
         }
     }
     if (test_config.myrank == 0) {printf("NUM_CONGESTOR_TESTS=%d\n", NUM_CONGESTOR_TESTS);}

     /* last we initialize the pretty table row separators */
     table_outerbar[tblsize] = '\0';
     table_innerbar[tblsize] = '\0';
     print_buffer[tblsize]   = '\0';
     for (i = 0; i < tblsize; i++) {
          table_outerbar[i] = '-';
          table_innerbar[i] = '-';
          print_buffer[i]   = '\0';
     }
     table_outerbar[0] = '+';
     table_outerbar[tblsize-1] = '+';
     table_innerbar[0] = '+';
     table_innerbar[tblsize-1] = '+';
     table_innerbar[34] = '+';
     for (i = 1; i < 3+verbose*4; i++) {
         table_innerbar[34+i*15] = '+';
     }

     build_subcomms(NUM_CONGESTOR_TESTS, &test_config, &nodes, &am_congestor, &congestor_set, &congestor_comm,
                    &test_comm, &local_comm, &nt_nodes, &nc_nodes);
     if (am_congestor) {
          init_rma(&test_config, congestor_comm);
     } else {
          init_rma(&test_config, test_comm);
     }

     /* define the sequence of each test type.  to increase the test count of either you must update
        the setting for NUM_NETWORK_TESTS and NUM_CONGESTOR_TESTS defined at the top */
     if (test_config.myrank == 0) printf("Chosen Tests\n");
     network_tests_list[0] = P2P_LATENCY; if (test_config.myrank == 0) printf("P2P_LATENCY\n");
     network_tests_list[1] = P2P_NEIGHBORS; if (test_config.myrank == 0) printf("P2P_NEIGHBORS\n");
     network_tests_list[2] = ALLREDUCE_LATENCY; if (test_config.myrank == 0) printf("ALLREDUCE_LATENCY\n");
     if (test_config.myrank == 0) printf("\n");

     if (test_config.myrank == 0) printf("Chosen Congestors\n");
     switch(NUM_CONGESTOR_TESTS) {
     case 6: congestor_tests_list[5] = ALLREDUCE_CONGESTOR; if (test_config.myrank == 0) printf("ALLREDUCE_CONGESTOR\n");
     case 5: congestor_tests_list[4] = P2P_BCAST_CONGESTOR; if (test_config.myrank == 0) printf("P2P_BCAST_CONGESTOR\n");
     case 4: congestor_tests_list[3] = RMA_BCAST_CONGESTOR; if (test_config.myrank == 0) printf("RMA_BCAST_CONGESTOR\n");
     case 3: congestor_tests_list[2] = RMA_INCAST_CONGESTOR; if (test_config.myrank == 0) printf("RMA_INCAST_CONGESTOR\n");
     case 2: congestor_tests_list[1] = P2P_INCAST_CONGESTOR; if (test_config.myrank == 0) printf("P2P_INCAST_CONGESTOR\n");
     case 1: congestor_tests_list[0] = A2A_CONGESTOR; if (test_config.myrank == 0) printf("A2A_CONGESTOR\n");
     default: break;
     }
     if (test_config.myrank == 0) printf("\n");

     if (test_config.myrank == 0) {
          printf("NetworkLoad Tests v%3.1f\n", VERSION);
          printf("  Test with %i MPI ranks (%i nodes)\n", test_config.nranks, nodes.nnodes);
          printf("  %i nodes running Network Tests\n", nt_nodes);
          printf("  %i nodes running Congestion Tests (min %i nodes per congestor)\n\n", nc_nodes, (int)floor((double)nc_nodes/NUM_CONGESTOR_TESTS));
          printf("  Legend\n   RR = random ring communication pattern\n   Lat = latency\n   BW = bandwidth\n   BW+Sync = bandwidth with barrier\n");
     }

     if (am_congestor) {

          run_congestor_tests(&test_config, NUM_NETWORK_TESTS, congestor_set, congestor_comm, test_comm, local_comm, MPI_COMM_WORLD);

     } else {

          run_network_tests(&test_config, NUM_CONGESTOR_TESTS, test_comm, local_comm, MPI_COMM_WORLD);

     }

     finalize_mpi(&test_config, &nodes);

     return 0;
}
