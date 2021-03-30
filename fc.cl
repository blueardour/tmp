
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef CHW

__kernel void fc_1(
  __global half *src, __global half *weight, __global half *dst, 
  int cin, int cout)
{
  int x = get_global_id(0); // out
  x = x >= cout ? cout - 1: x;
  int i;
  half buffer;
  buffer = 0;
  for(i=0; i<cin; i++) {
    buffer += src[i] * weight[i*cout + x];
  }
  dst[x] = buffer;
}

__kernel void fc_2(
  __global half *src, __global half *weight, __global half *dst, 
  int cin, int cout)
{
  int x = get_global_id(0) << 1; // out
  x = x >= (cout-2) ? cout - 2: x;
  int i;
  half2 buffer;
  buffer = 0;
  for(i=0; i<cin; i++) {
    buffer += (half2)src[i] * vload2(0, weight + i*cout + x);
  }
  vstore2(buffer, 0, dst + x);
}

__kernel void fc_4(
  __global half *src, __global half *weight, __global half *dst, 
  int cin, int cout)
{
  int x = get_global_id(0) << 2; // out
  x = x >= (cout-4) ? cout - 4: x;
  int i;
  half4 buffer;
  buffer = 0;
  for(i=0; i<cin; i++) {
    buffer += (half4)src[i] * vload4(0, weight + i*cout + x);
  }
  vstore4(buffer, 0, dst + x);
}

__kernel void fc_8(
  __global half *src, __global half *weight, __global half *dst, 
  int cin, int cout)
{
  int x = get_global_id(0) << 3; // out
  x = x >= (cout-8) ? cout - 8: x;
  int i;
  half8 buffer;
  buffer = 0;
  for(i=0; i<cin; i++) {
    buffer += (half8)src[i] * vload8(0, weight + i*cout + x);
  }
  vstore8(buffer, 0, dst + x);
}

#endif

#pragma OPENCL EXTENSION cl_khr_fp16 : disable
