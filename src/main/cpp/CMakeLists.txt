#
# Copyright (C) The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.4.1)

set(DIST_DIR ${CMAKE_SOURCE_DIR}/../../../../distribution)

# build native_app_glue as a static lib
set(${CMAKE_C_FLAGS}, "${CMAKE_C_FLAGS}")
add_library(native_app_glue STATIC
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

# now build app's shared lib
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -Wall -Werror")

# Export ANativeActivity_onCreate(),
# Refer to: https://github.com/android-ndk/ndk/issues/381.
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

add_library(native-activity SHARED main.cpp)

target_include_directories(
        native-activity PRIVATE
    ${ANDROID_NDK}/sources/android/native_app_glue
        ${DIST_DIR}/OpenCL/include
        )

add_library(cpp SHARED IMPORTED)
set_target_properties(cpp PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libc++.so)

add_library(hidlbase SHARED IMPORTED)
set_target_properties(hidlbase PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libhidlbase.so)

add_library(processgroup SHARED IMPORTED)
set_target_properties(processgroup PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libprocessgroup.so)

add_library(utils SHARED IMPORTED)
set_target_properties(utils PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libutils.so)

add_library(base SHARED IMPORTED)
set_target_properties(base PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libbase.so)

add_library(cutils SHARED IMPORTED)
set_target_properties(cutils PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libcutils.so)

add_library(dl_android SHARED IMPORTED)
set_target_properties(dl_android PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libdl_android.so)

add_library(cgrouprc SHARED IMPORTED)
set_target_properties(cgrouprc PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libcgrouprc.so)

add_library(vndksupport SHARED IMPORTED)
set_target_properties(vndksupport PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libvndksupport.so)

add_library(ld SHARED IMPORTED)
set_target_properties(ld PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/ld-android.so)

add_library(ladder SHARED IMPORTED)
set_target_properties(ladder PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libladder.so)

add_library(foo SHARED IMPORTED)
set_target_properties(foo PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libfoo.so)

add_library(hardware SHARED IMPORTED)
set_target_properties(hardware PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libhardware.so)

add_library(gpuaux SHARED IMPORTED)
set_target_properties(gpuaux PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libgpu_aux.so)

add_library(dpframework SHARED IMPORTED)
set_target_properties(dpframework PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libdpframework.so)

add_library(gpud SHARED IMPORTED)
set_target_properties(gpud PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libgpud.so)

add_library(gralloc SHARED IMPORTED)
set_target_properties(gralloc PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libgralloc_extra.so)

add_library(gralloctypes SHARED IMPORTED)
set_target_properties(gralloctypes PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libgralloctypes.so)

add_library(ahgc SHARED IMPORTED)
set_target_properties(ahgc PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libahgc.so)

add_library(ahgc1 SHARED IMPORTED)
set_target_properties(ahgc1 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libahgc1.1.so)

add_library(ahgc12 SHARED IMPORTED)
set_target_properties(ahgc12 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libahgc1.2.so)

add_library(ahgm SHARED IMPORTED)
set_target_properties(ahgm PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libahgm.so)

add_library(ahgm2 SHARED IMPORTED)
set_target_properties(ahgm2 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libahgm2.so)

add_library(ahgm2.1 SHARED IMPORTED)
set_target_properties(ahgm2.1 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libahgm2.1.so)

add_library(ahgm3 SHARED IMPORTED)
set_target_properties(ahgm3 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libahgm3.so)

add_library(ahgm4 SHARED IMPORTED)
set_target_properties(ahgm4 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libahgm4.so)

add_library(vmhm1.0 SHARED IMPORTED)
set_target_properties(vmhm1.0 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libvmhm1.0.so)

add_library(vmhm1.1 SHARED IMPORTED)
set_target_properties(vmhm1.1 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libvmhm1.1.so)

add_library(vmhm1.2 SHARED IMPORTED)
set_target_properties(vmhm1.2 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libvmhm1.2.so)

add_library(vmhm1.3 SHARED IMPORTED)
set_target_properties(vmhm1.3 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libvmhm1.3.so)

add_library(vmhm1.4 SHARED IMPORTED)
set_target_properties(vmhm1.4 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libvmhm1.4.so)

add_library(vmhm1.5 SHARED IMPORTED)
set_target_properties(vmhm1.5 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libvmhm1.5.so)

add_library(ion_ulit SHARED IMPORTED)
set_target_properties(ion_ulit PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libion_ulit.so)

add_library(ion_mtk SHARED IMPORTED)
set_target_properties(ion_mtk PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libion_mtk.so)

add_library(ion SHARED IMPORTED)
set_target_properties(ion PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libion.so)

add_library(dmapbufmap SHARED IMPORTED)
set_target_properties(dmapbufmap PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libdmabufheap.so)

add_library(utilscallstack SHARED IMPORTED)
set_target_properties(utilscallstack PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libutilscallstack.so)

add_library(ahcnp SHARED IMPORTED)
set_target_properties(ahcnp PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libahgcv2.so)

add_library(ahgcv2 SHARED IMPORTED)
set_target_properties(ahgcv2 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/android.hardware.common-V2-ndk_platform.so)

add_library(ahm1.0 SHARED IMPORTED)
set_target_properties(ahm1.0 PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/android.hidl.memory@1.0.so)

add_library(hidlmemory SHARED IMPORTED)
set_target_properties(hidlmemory PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libhidlmemory.so)

add_library(vmhp SHARED IMPORTED)
set_target_properties(vmhp PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/vendor.mediatek.hardware.pq@2.0.so)

add_library(q_prot SHARED IMPORTED)
set_target_properties(q_prot PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libpq_prot.so)

add_library(ahmt SHARED IMPORTED)
set_target_properties(ahmt PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/android.hidl.memory.token@1.0.so)

add_library(backtrace SHARED IMPORTED)
set_target_properties(backtrace PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libbacktrace.so)

add_library(unwindstack SHARED IMPORTED)
set_target_properties(unwindstack PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libunwindstack.so)

add_library(lzma SHARED IMPORTED)
set_target_properties(lzma PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/liblzma.so)

add_library(lib_openCL SHARED IMPORTED)
set_target_properties(lib_openCL PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libOpenCL.so)

add_library(GLES_mali SHARED IMPORTED)
set_target_properties(GLES_mali PROPERTIES IMPORTED_LOCATION ${DIST_DIR}/OpenCL/lib/${ANDROID_ABI}/libGLES_mali.so)


# add lib dependencies
target_link_libraries(native-activity
    android
    native_app_glue
        ahgm
        ahgm
        gralloc
        utilscallstack
        backtrace
        unwindstack
        lzma
        dmapbufmap
        gralloctypes
        gpud
        gpuaux
        vmhm1.5
        vmhm1.4
        vmhm1.3
        vmhm1.2
        vmhm1.1
        vmhm1.0
        ahgm4
        ahgm3
        ahgm2.1
        ahgm2
        hardware
        dpframework
        ion_mtk
        ion_ulit
        ion
        ladder
        q_prot
        vmhp
        hidlmemory
        ahmt
        ahm1.0
        foo
        ahgcv2
        ahcnp
        hidlbase
        ahgc12
        ahgc1
        ahgc
        cutils
        base
        utils
        processgroup
        cgrouprc
        vndksupport
        dl_android
        ld
        cpp
        log
        EGL
        GLES_mali
        GLESv3
        )
