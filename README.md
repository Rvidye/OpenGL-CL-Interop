# OpenGLES - OpenCL Interop Android

OpenCL is an open standard for parallel computing that enables performance portability across diverse computing platforms. In this video, I have demonstrated OpenCL and OpenGL ES 3.2 Interoperability on ARM Mali G-76 GPU with Android 12.0.

We render a 256 x 256 grid of sin wave in the demo which the CPU handles with ease. As we increase the Grid size from 256 x 256 to 1024 x 1024 we see the CPU struggle to keep up with GPU rendering. By pressing the volume key, we transfer sin wave computation from the CPU to OpenCL Kernel.

This demo is a modified version of the sample present in the Book "OpenCL Programming Guide" by Aaftab Munshi, Benedict Gaster, and Timothy G. Mattson.

Device Details :
Xiaomi Redmi Note 10 S
GPU: ARM Mali-G76
OpenCL Version: 1.2 Full Profile

References : 
"The OpenCL Programming Guide" by Aaftab Munshi, Benedict Gaster, and Timothy G. Mattson.
OpenGL ES 3.2 Android Documentation.]
