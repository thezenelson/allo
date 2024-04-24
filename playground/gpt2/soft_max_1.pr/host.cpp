//=============================================================================
// Auto generated by Allo
//=============================================================================

// OpenCL utility layer include
#include "xcl2.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Program program;
    cl::Kernel krnl_top;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
    // hood user ptr is used if it is properly aligned. when not aligned, runtime had no choice
    // but to create its own host side buffer. So it is recommended to use this allocator if
    // user wish to create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
    // boundary. It will ensure that user buffer is used when user create Buffer/Mem object with
    // CL_MEM_USE_HOST_PTR
    size_t size_bytes_in0 = sizeof(float) * 50257;
    std::vector<float, aligned_allocator<float> > source_in0(50257);
    size_t size_bytes_out0 = sizeof(float) * 50257;
    std::vector<float, aligned_allocator<float> > source_out0(50257);

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_top = cl::Kernel(program, "top", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_in0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_bytes_in0, source_in0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_out0(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, size_bytes_out0, source_out0.data(), &err));

    OCL_CHECK(err, err = krnl_top.setArg(0, buffer_in0));
    OCL_CHECK(err, err = krnl_top.setArg(1, buffer_out0));
    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in0}, 0 /* 0 means from host*/));

    cl::Event event;
    uint64_t nstimestart, nstimeend;
    std::cout << "|-------------------------+-------------------------|\n"
              << "| Kernel                  |    Wall-Clock Time (ns) |\n"
              << "|-------------------------+-------------------------|\n";

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_top, nullptr, &event));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out0}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    // OpenCL Host Code Ends

    // Get the execution time
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
    auto exe_time = nstimeend - nstimestart;
    std::cout << "| " << std::left << std::setw(24) << "top "
              << "|" << std::right << std::setw(24) << exe_time << " |\n";
    std::cout << "|-------------------------+-------------------------|\n";
    std::cout << "Note: Wall Clock Time is meaningful for real hardware execution "
              << "only, not for emulation.\n";
    std::cout << "Please refer to profile summary for kernel execution time for "
              << "hardware emulation.\n";
    std::cout << "TEST PASSED\n\n";
    return EXIT_SUCCESS;
}
