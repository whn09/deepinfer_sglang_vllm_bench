#include <cstdio>
#include <cuda.h>
int main() {
    cuInit(0); CUdevice d; cuDeviceGet(&d, 0); int fab = -1, posix = -1, vmm = -1;
    cuDeviceGetAttribute(&fab, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, d);
    cuDeviceGetAttribute(&posix, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, d);
    cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, d);
    CUmemAllocationProp prop = {}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC; size_t g = 0; cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    CUmemGenericAllocationHandle h; CUresult r = cuMemCreate(&h, g, &prop, 0);
    CUmemFabricHandle fh; CUresult e = r == CUDA_SUCCESS ? cuMemExportToShareableHandle(&fh, h, CU_MEM_HANDLE_TYPE_FABRIC, 0) : r;
    printf("VMM=%d POSIX_FD=%d FABRIC_attr=%d cuMemCreate(FABRIC)=%d export(FABRIC)=%d\n", vmm, posix, fab, (int)r, (int)e);
}
