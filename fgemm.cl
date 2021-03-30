
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
//#pragma OPENCL EXTENSION cl_arm_printf: enable

// ternary acceleration code

#ifdef CHW

// CHW memory arch, // K = Cin * ks * ks / 4
__kernel void gemm2_2x1(__global short *dst, __global uchar *src, __global uchar *weight, int M, int N, int K)
{
  int x = get_global_id(0) << 1; // M with length width*height
  int y = get_global_id(1); // N with length cout
  x = min(x, M-2);
  y = min(y, N-1);

  short2 result = 0;
  uchar2 buffer, masked, filter;
  short i;
  for(i=0; i<K; i++)
  {
    filter = (uchar2)weight[i*N + y];
    masked = ((filter >> 1) & 0x55) | ((filter << 1) & 0xaa);
    masked = masked ^ filter;
    buffer = vload2(0, src + i*M + x);
    buffer = ~(buffer ^ filter);
    buffer = (masked | 0x55) | (~masked | buffer);
    result += convert_short2(popcount(buffer));
  }
  result = result - (short2)(4*K);
  vstore2(result, 0, dst + y*M + x);
  return;
}

__kernel void gemm2_2x2(__global short *dst, __global uchar *src, __global uchar *weight, int M, int N, int K)
{
  int x = get_global_id(0) << 1; // M with length width*height
  int y = get_global_id(1) << 1; // N with length cout
  x = min(x, M-2);
  y = min(y, N-2);

  short4 result = 0;
  uchar2 buffer, masked, filter;
  short i;
  for(i=0; i<K; i++)
  {
    filter = vload2(0, weight + i*N + y);
    masked = ((filter >> 1) & 0x55) | ((filter << 1) & 0xaa);
    masked = masked ^ filter;

    buffer = vload2(0, src + i*M + x);
    buffer = ~(buffer ^ (uchar2)filter.s0);
    buffer = (uchar2)(masked.s0 | 0x55) | (buffer | ~masked.s0);
    result.lo += convert_short2(popcount(buffer));

    buffer = ~(buffer ^ (uchar2)filter.s1);
    buffer = (uchar2)(masked.s1 | 0x55) | (buffer | ~masked.s1);
    result.hi += convert_short2(popcount(buffer));
  }
  result = result - (short4)(4*K);
  vstore2(result.lo, 0, dst + (y+0)*M + x);
  vstore2(result.hi, 0, dst + (y+1)*M + x);
  return;
}

__kernel void gemm2_2x4(__global short *dst, __global uchar *src, __global uchar *weight, int M, int N, int K)
{
  int x = get_global_id(0) << 1; // M with length width*height
  int y = get_global_id(1) << 2; // N with length cout
  x = min(x, M-2);
  y = min(y, N-4);

  short8 result = 0;
  uchar2 buffer;
  uchar4 filter, masked;
  short i;
  for(i=0; i<K; i++)
  {
    filter = vload4(0, weight + i*N + y);
    masked = ((filter >> 1) & 0x55) | ((filter << 1) & 0xaa);
    masked = masked ^ filter;

    buffer = vload2(0, src + i*M + x);
    buffer = ~(buffer ^ (uchar2)filter.s0);
    buffer = (uchar2)(masked.s0 | 0x55) | (buffer | ~masked.s0);
    result.lo.lo += convert_short2(popcount(buffer));

    buffer = ~(buffer ^ (uchar2)filter.s1);
    buffer = (uchar2)(masked.s1 | 0x55) | (buffer | ~masked.s1);
    result.lo.hi += convert_short2(popcount(buffer));

    buffer = ~(buffer ^ (uchar2)filter.s2);
    buffer = (uchar2)(masked.s2 | 0x55) | (buffer | ~masked.s2);
    result.hi.lo += convert_short2(popcount(buffer));

    buffer = ~(buffer ^ (uchar2)filter.s3);
    buffer = (uchar2)(masked.s3 | 0x55) | (buffer | ~masked.s3);
    result.hi.hi += convert_short2(popcount(buffer));
  }
  result = result - (short8)(4*K);
  vstore2(result.lo.lo, 0, dst + (y+0)*M + x);
  vstore2(result.lo.hi, 0, dst + (y+1)*M + x);
  vstore2(result.hi.lo, 0, dst + (y+2)*M + x);
  vstore2(result.hi.hi, 0, dst + (y+3)*M + x);
  return;
}

__kernel void gemm2_4x1(__global short *dst, __global uchar *src, __global uchar *weight, int M, int N, int K)
{
  int x = get_global_id(0) << 2; // M with length width*height
  int y = get_global_id(1); // N with length cout
  x = min(x, M-4);
  y = min(y, N-1);

  short4 result = 0;
  uchar4 buffer;
  uchar filter, masked;
  short i;
  for(i=0; i<K; i++)
  {
    filter = weight[i*N + y];
    masked = ((filter >> 1) & 0x55) | ((filter << 1) & 0xaa);
    masked = masked ^ filter;

    buffer = vload4(0, src + i*M + x);
    buffer = ~(buffer ^ (uchar4)filter);
    buffer = (uchar4)(masked | 0x55) | (buffer | ~masked);
    result += convert_short4(popcount(buffer));
  }
  result = result - (short4)(4*K);
  vstore4(result, 0, dst + y*M + x);
  return;
}

__kernel void gemm2_4x2(__global short *dst, __global uchar *src, __global uchar *weight, int M, int N, int K)
{
  int x = get_global_id(0) << 2; // M with length width*height
  int y = get_global_id(1) << 1; // N with length cout
  x = min(x, M-4);
  y = min(y, N-2);

  short8 result = 0;
  uchar4 buffer;
  uchar2 filter, masked;
  short i;
  for(i=0; i<K; i++)
  {
    filter = vload2(0, weight + i*N + y);
    masked = ((filter >> 1) & 0x55) | ((filter << 1) & 0xaa);
    masked = masked ^ filter;

    buffer = vload4(0, src + i*M + x);
    buffer = ~(buffer ^ (uchar4)filter.s0);
    buffer = (uchar4)(masked.s0 | 0x55) | (buffer | ~masked.s0);
    result.lo += convert_short4(popcount(buffer));

    buffer = ~(buffer ^ (uchar4)filter.s1);
    buffer = (uchar4)(masked.s1 | 0x55) | (buffer | ~masked.s1);
    result.hi += convert_short4(popcount(buffer));
  }
  result = result - (short8)(4*K);
  vstore4(result.lo, 0, dst + (y+0)*M + x);
  vstore4(result.hi, 0, dst + (y+1)*M + x);
  return;
}

__kernel void gemm2_4x4(__global short *dst, __global uchar *src, __global uchar *weight, int M, int N, int K)
{
  int x = get_global_id(0) << 2; // M with length width*height
  int y = get_global_id(1) << 2; // N with length cout
  x = min(x, M-4);
  y = min(y, N-4);

  short16 result = 0;
  uchar4 buffer;
  uchar4 filter, masked;
  short i;
  for(i=0; i<K; i++)
  {
    filter = vload4(0, weight + i*N + y);
    masked = ((filter >> 1) & 0x55) | ((filter << 1) & 0xaa);
    masked = masked ^ filter;

    buffer = vload4(0, src + i*M + x);
    buffer = ~(buffer ^ (uchar4)filter.s0);
    buffer = (uchar4)(masked.s0 | 0x55) | (buffer | ~masked.s0);
    result.lo.lo += convert_short4(popcount(buffer));

    buffer = ~(buffer ^ (uchar4)filter.s1);
    buffer = (uchar4)(masked.s1 | 0x55) | (buffer | ~masked.s1);
    result.lo.hi += convert_short4(popcount(buffer));

    buffer = ~(buffer ^ (uchar4)filter.s2);
    buffer = (uchar4)(masked.s2 | 0x55) | (buffer | ~masked.s2);
    result.hi.lo += convert_short4(popcount(buffer));

    buffer = ~(buffer ^ (uchar4)filter.s3);
    buffer = (uchar4)(masked.s3 | 0x55) | (buffer | ~masked.s3);
    result.hi.hi += convert_short4(popcount(buffer));
  }
  result = result - (short16)(4*K);
  vstore4(result.lo.lo, 0, dst + (y+0)*M + x);
  vstore4(result.lo.hi, 0, dst + (y+1)*M + x);
  vstore4(result.hi.lo, 0, dst + (y+2)*M + x);
  vstore4(result.hi.hi, 0, dst + (y+3)*M + x);
  return;
}

__kernel void gemm2_8x1(__global short *dst, __global uchar *src, __global uchar *weight, int M, int N, int K)
{
  int x = get_global_id(0) << 3; // M with length width*height
  int y = get_global_id(1); // N with length cout
  x = min(x, M-8);
  y = min(y, N-1);

  short8 result = 0;
  uchar8 buffer;
  uchar filter, masked;
  short i;
  for(i=0; i<K; i++)
  {
    filter = weight[i*N + y];
    masked = ((filter >> 1) & 0x55) | ((filter << 1) & 0xaa);
    masked = masked ^ filter;

    buffer = vload8(0, src + i*M + x);
    buffer = ~(buffer ^ (uchar8)filter);
    buffer = (uchar8)(masked | 0x55) | (buffer | ~masked);
    result += convert_short8(popcount(buffer));
  }
  result = result - (short8)(4*K);
  vstore8(result, 0, dst + y*M + x);
  return;
}

#endif

#pragma OPENCL EXTENSION cl_khr_fp16 : disable


