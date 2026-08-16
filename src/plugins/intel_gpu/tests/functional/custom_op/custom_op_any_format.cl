// Writes a value derived from its own dispatch coordinates and the declared output
// width, rather than copying elementwise. A flat copy cannot detect a transposed
// output layout -- element i maps to element i either way -- so this kernel makes the
// axis extents observable in the result.
__kernel void custom_kernel_axis_probe(__global const INPUT0_TYPE* input,
                                       __global OUTPUT0_TYPE* output) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = OUTPUT0_DIMS[3];

    output[y * w + x] = (OUTPUT0_TYPE)(y * 1000 + x);
}
