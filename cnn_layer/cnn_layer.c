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
#include "cnn_layer.h"

#define STACK_SIZE      2048
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

// End of computation
int done = 0;

static Pixel PULP_L2_DATA  Out[OUT_DIM];

void check_CNN_5x5_Scalar          ();
void check_CNN_5x5_Vector          ();

void main_fn()
{

#ifdef DOTP
  check_CNN_5x5_Vector();
#else
  check_CNN_5x5_Scalar();
#endif

}

void check_CNN_5x5_Scalar() {

  // start benchmark
  printf("CNN WINDOW=%d, DATA_WIDTH=%d\n",FILT_WIN,DATA_WIDTH);
  InitZero(Out, OUT_DIM);

  reset_timer();
  start_timer();

  CNN_layer_Scalar(In_Img, Out, IMG_ROW, IMG_COL, Filter_Kern);

  stop_timer();

  printf("Number of cycles: %d\n ",get_time());

}

void check_CNN_5x5_Vector() {

  printf("CNN WINDOW=%d, DATA_WIDTH=%d\n",FILT_WIN,DATA_WIDTH);

  InitZero(Out, OUT_DIM);

  reset_timer();
  start_timer();

  CNN_layer_Vector(In_Img, Out, IMG_ROW, IMG_COL, Filter_Kern);

  stop_timer();
  printf("Number of cycles: %d\n ",get_time());

}

// load initialize out to 0
void __attribute__ ((noinline)) InitZero(Pixel * __restrict__ Img, int size)
{
  int i;

  for (i=0; i < size; i++)
    Img[i] = 0;

}

#define CNN_DEBUG
int  __attribute__ ((noinline)) checkresult(Pixel * __restrict__ Out, Pixel * __restrict__ OutGold, int N)
{
  int i;
  int err = 0;

  for (i = 0; i<N; i++) {
    if (Out[i]!=OutGold[i]) {
#ifdef CNN_DEBUG
      printf("At index %d: Actual value: %d: Expected: %d\n", i, Out[i],  OutGold[i]);
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
