#include <cstdio>
#include <cuda.h>
int main() {
    cuInit(0); CUdevice d; cuDeviceGet(&d, 0); CUcontext c; cuDevicePrimaryCtxRetain(&c, d); cuCtxSetCurrent(c);
    CUmemAllocationProp prop = {}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t g = 0; cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    size_t chunk = 2 * g;  // two physical allocations of 2 granules each, mapped back to back in one 6-granule reservation
    CUmemGenericAllocationHandle a, b; cuMemCreate(&a, chunk, &prop, 0); cuMemCreate(&b, chunk, &prop, 0);
    CUdeviceptr va; cuMemAddressReserve(&va, 3 * chunk, 0, 0, 0);
    cuMemMap(va, chunk, 0, a, 0); cuMemMap(va + chunk, chunk, 0, b, 0);   // third chunk left unmapped
    CUmemAccessDesc acc = {}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; cuMemSetAccess(va, 2 * chunk, &acc, 1);
    printf("granularity=%zu chunk=%zu\n", g, chunk);
    for (size_t off : {(size_t)0, g, chunk, chunk + g, 2 * chunk}) {
        CUdeviceptr base = 0; size_t sz = 0; CUresult r = cuMemGetAddressRange(&base, &sz, va + off);
        CUmemGenericAllocationHandle h = 0; CUresult rh = cuMemRetainAllocationHandle(&h, (void*)(va + off));
        CUmemAllocationProp p2 = {}; if (rh == CUDA_SUCCESS) cuMemGetAllocationPropertiesFromHandle(&p2, h);
        printf("off=%zu getAddressRange rc=%d base_off=%lld size=%zu | retain rc=%d handle=%llu same_as_a=%d same_as_b=%d reqHandleTypes=%d\n",
               off, (int)r, r == CUDA_SUCCESS ? (long long)(base - va) : -1LL, sz, (int)rh, (unsigned long long)h, h == a, h == b, (int)p2.requestedHandleTypes);
        if (rh == CUDA_SUCCESS) cuMemRelease(h);
    }
    int fd = -1; printf("export POSIX_FD rc=%d fd=%d\n", (int)cuMemExportToShareableHandle(&fd, a, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), fd);
}
