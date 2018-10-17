// Copyright 2017 ETH Zurich and University of Bologna.
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the “License”); you may not use this file except in
// compliance with the License.  You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "rt/rt_api.h"
#include "pulp.h"
#include "convolution.h"

#define STACK_SIZE      2048
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

// End of computation
int done = 0;


void check_Conv5x5_Scalar();
void check_Conv5x5_Vector();

RT_L2_DATA Pixel Out[IMG_DIM];

void main_fn()
{
#ifdef DOTP
  check_Conv5x5_Vector();
#else
  check_Conv5x5_Scalar();
#endif
}

void check_Conv5x5_Scalar() {

  // start benchmark
  Filtc Kernel5x5_Scalar[FILT_DIM];
  Pixel In[IMG_DIM];

  printf("[scalar] 2D Convolution WINDOW=%d, DATA_WIDTH=%d\n",FILT_WIN,DATA_WIDTH);
  InitZero(Out, IMG_DIM);

  reset_timer();
  start_timer();

  Conv5x5_Scalar(In_Img, Out, IMG_ROW, IMG_COL, Filter_Kern);

  stop_timer();

  printf("Number of cycles: %d\n ",get_time());

  checkresult(Out, Gold_Out_Img, IMG_DIM);

}

void check_Conv5x5_Vector() {

  if (get_core_id()==0) {
    printf("2D Convolution WINDOW=%d, DATA_WIDTH=%d\n",FILT_WIN,DATA_WIDTH);
    InitZero(Out, IMG_DIM);
  }

#ifdef PARALLEL
  synch_barrier();
#endif

  reset_timer();
  start_timer();

  Conv5x5_Vector(In_Img, Out, IMG_ROW, IMG_COL, Filter_Kern);

  stop_timer();
  if (rt_core_id()==0)  printf("Number of cycles: %d\n ",get_time());

#ifdef PARALLEL
  synch_barrier();
#endif

  checkresult(Out, Gold_Out_Img, IMG_DIM);

}

// load kernel
void __attribute__ ((noinline)) InitKernel(Filtc * __restrict__ Kernel, int size)
{
  int i;
  int n = size*size;
  for (i=0; i < n; i++) {
      Kernel[i] = Filter_Kern[i];
  }
}

// load input img
void __attribute__ ((noinline)) InitData(Pixel * __restrict__ Img, int size)
{
  int i;

  for (i=0; i < size; i++)
      Img[i] = In_Img[i];

}

// load initialize out to 0
void __attribute__ ((noinline)) InitZero(Pixel * __restrict__ Img, int size)
{
  int i;

  for (i=0; i < size; i++)
      Img[i] = 0;

}

#define CONV2D_DEBUG
int  __attribute__ ((noinline)) checkresult(Pixel * __restrict__ Out, Pixel * __restrict__ OutGold, int N)
{
  int i;
  int err = 0;

  for (i = 0; i<N; i++) {
    if (Out[i]!=OutGold[i]) {
#ifdef CONV2D_DEBUG
      printf("At index %d: Actual value: %x: Expected: %x\n", i, Out[i],  OutGold[i]);
#endif
      err++;
    }
  }
  return err;
}

#ifndef FABRIC
static void cluster_entry(void *arg)
{

  rt_team_fork(NUM_CORES, main_fn, (void *)0x0);

}
#endif

static void end_of_call(void *arg)
{
  done = 1;
  //printf("[clusterID: 0x%x] Hello from core %d\n", rt_cluster_id(), rt_core_id());
}

int main()
{
  //printf("Entering main controller\n");

#ifdef FABRIC
  main_fn();
#else
  rt_event_sched_t * psched = rt_event_internal_sched();
  if (rt_event_alloc(psched, NUM_CORES)) return -1;

  rt_cluster_mount(MOUNT, CID, 0, NULL);

  void *stacks = rt_alloc(RT_ALLOC_CL_DATA, STACK_SIZE*rt_nb_pe());
  if (stacks == NULL) return -1;

  rt_cluster_call(NULL, CID, cluster_entry, NULL, stacks, STACK_SIZE, STACK_SIZE, NUM_CORES, rt_event_get(psched, end_of_call, (void *) CID));

  while(!done)
    rt_event_execute(psched, 1);

  rt_cluster_mount(UNMOUNT, CID, 0, NULL);
#endif

  return 0;
}
