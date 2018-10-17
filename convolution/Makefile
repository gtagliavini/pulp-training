PULP_APP = convolution

ifdef FABRIC
PULP_APP_FC_SRCS  = convolution.c conv_kernels.c
else
PULP_APP_FC_SRCS  = convolution.c
PULP_APP_SRCS  = conv_kernels.c
endif

PULP_CFLAGS = -O3 -g3 #-DDOTP

ifdef FABRIC
PULP_CFLAGS += -DFABRIC
endif

ifeq (${OPT}, 0)
PULP_CL_ARCH_CFLAGS = -march=rv32imc  -D__riscv__
PULP_ARCH_LDFLAGS = -march=rv32imc -D__riscv__
PULP_CFLAGS += -DNUM_CORES=1
endif

ifeq (${OPT}, 1)
PULP_CFLAGS += -DNUM_CORES=1
endif

ifeq (${OPT}, 2)
PULP_CFLAGS += -DNUM_CORES=1 -DDOTP
endif


ifdef CORES
PULP_CFLAGS += -DNUM_CORES=${CORES} -DDOTP -DPARALLEL
endif

#PULP_FC_ARCH_CFLAGS = -march=rv32imcxpulpslim -DRV_ISA_RV32=1 -D__riscv__
#PULP_FC_ARCH_CFLAGS = -march=rv32imcxpulpv2 -D__riscv__
#zero-riscy:   -march=rv32imcxpulpslim -DRV_ISA_RV32=1 -D__riscv__
#riscy:        -march=rv32imcxpulpv2 -D__riscv__

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
