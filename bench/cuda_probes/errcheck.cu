#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
int main() {
    printf("cudaErrorDeviceUninitialized=%d str=\"%s\"\n", (int)cudaErrorDeviceUninitialized, cudaGetErrorString(cudaErrorDeviceUninitialized));
    cuInit(0); cudaSetDevice(0);
    void* p = nullptr; cudaMalloc(&p, 1 << 20);
    CUmemGenericAllocationHandle h;
    printf("cudaMalloc buffer: cuMemRetainAllocationHandle -> %d (expect non-zero)\n", (int)cuMemRetainAllocationHandle(&h, p));
    CUmemAllocationProp prop = {}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
    size_t gran = 0; cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    CUmemGenericAllocationHandle a; cuMemCreate(&a, gran, &prop, 0);
    CUdeviceptr va; cuMemAddressReserve(&va, gran, 0, 0, 0); cuMemMap(va, gran, 0, a, 0);
    CUmemAccessDesc acc = {}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; cuMemSetAccess(va, gran, &acc, 1);
    CUresult r = cuMemRetainAllocationHandle(&h, (void*)va);
    printf("cuMemCreate buffer: cuMemRetainAllocationHandle -> %d (expect 0)\n", (int)r);
    if (r == CUDA_SUCCESS) cuMemRelease(h);
    cudaIpcMemHandle_t ih; printf("cuMemCreate buffer: cudaIpcGetMemHandle -> %s\n", cudaGetErrorString(cudaIpcGetMemHandle(&ih, (void*)va)));
    return 0;
}
