#!/bin/bash

export UCX_LOG_LEVEL=info
export UCX_MODULE_LOG_LEVEL=info
export UCX_PROTO_INFO=y

date > log_file.txt

ucx_info -d >> log_file.txt

unset UCX_TLS
unset UCX_RNDV_SCHEME
unset UCX_RNDV_FRAG_MEM_TYPES
unset UCX_RNDV_FRAG_MEM_TYPE

mpiexec -n 2 ./build/bw_and_leak_check H >> log_file.txt
mpiexec -n 2 ./build/bw_and_leak_check D >> log_file.txt

export UCX_TLS=cuda,sm
export UCX_RNDV_SCHEME=put_ppln
export UCX_RNDV_FRAG_MEM_TYPES=cuda
export UCX_RNDV_FRAG_MEM_TYPE=cuda

mpiexec -n 2 ./build/bw_and_leak_check H >> log_file.txt
mpiexec -n 2 ./build/bw_and_leak_check D >> log_file.txt
