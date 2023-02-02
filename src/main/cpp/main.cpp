/*
 * Copyright (C) 2010 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

//BEGIN_INCLUDE(all)
#include <initializer_list>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <jni.h>
#include <cerrno>
#include <cassert>

#include <GLES3/gl32.h>
#include <GLES3/gl3ext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>


#define CL_TARGET_OPENCL_VERSION 210
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_egl.h>

#include "vmath.h"
#include <android/log.h>
#include <android_native_app_glue.h>

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))

/**
 * Our saved state data.
 */
struct saved_state {
    float angle;
    int32_t x;
    int32_t y;
};

/**
 * Shared state for our app.
 */
struct engine {
    struct android_app* app;

    int animating;
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;
    int32_t width;
    int32_t height;
    struct saved_state state;
};

// Global Data
bool onGPU = false;

unsigned long mesh_width = 256;
unsigned long mesh_height = 256;

// OpenGL Varriables
GLuint program;
GLuint vao;
GLuint vbo_cpu,vbo_gpu;

// uniforms
GLuint mvp,color;
vmath::mat4 perspectiveProjection;

GLfloat *buffer;
GLfloat *clBuffer;
GLfloat mytime = 0.0f;

// opencl variables

cl_platform_id cpPlatform;
cl_context cxGPUcontext;
cl_device_id* cdDevices;
cl_uint uiDevCount;
cl_command_queue cqCommandQueue;
cl_kernel ckKernel;
cl_mem vbo_cl;
cl_program cpProgram;
cl_int ciErrNum;
size_t szGlobalWorkSize[] = {mesh_width,mesh_height};

// Functions

void runCPU();
void runGPU();
void CreateVBO();


/**
 * Initialize an EGL context for the current display.
 */
static int engine_init_display(struct engine* engine) {
    // initialize OpenGL ES and EGL



    //gl3stubInit();
    /*
     * Here specify the attributes of the desired configuration.
     * Below, we select an EGLConfig with at least 8 bits per color
     * component compatible with on-screen windows
     */

    const EGLint attribs[] = {
            EGL_SURFACE_TYPE,
            EGL_WINDOW_BIT,
            EGL_BLUE_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_RED_SIZE, 8,
            EGL_DEPTH_SIZE,8,
            EGL_NONE
    };
    EGLint w, h, format;
    EGLint numConfigs;
    EGLConfig config = nullptr;
    EGLSurface surface;
    EGLContext context;



    eglBindAPI(EGL_OPENGL_ES_API);
    LOGI("GLERROR : %d",eglGetError());
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if(display == EGL_NO_DISPLAY)
        LOGI("GLERROR : %d",eglGetError());
    if(!eglInitialize(display, nullptr, nullptr))
       LOGI("Wrong Display");
    LOGI("GLERROR : %d",eglGetError());
    /* Here, the application chooses the configuration it desires.
     * find the best match if possible, otherwise use the very first one
     */
    eglChooseConfig(display, attribs, nullptr,0, &numConfigs);
    LOGI("GLERROR : %d",eglGetError());
    std::unique_ptr<EGLConfig[]> supportedConfigs(new EGLConfig[numConfigs]);
    assert(supportedConfigs);
    eglChooseConfig(display, attribs, supportedConfigs.get(), numConfigs, &numConfigs);
    LOGI("GLERROR : %d",eglGetError());
    assert(numConfigs);
    auto i = 0;
    for (; i < numConfigs; i++) {
        auto& cfg = supportedConfigs[i];
        EGLint r, g, b, d;
        if (eglGetConfigAttrib(display, cfg, EGL_RED_SIZE, &r)   &&
            eglGetConfigAttrib(display, cfg, EGL_GREEN_SIZE, &g) &&
            eglGetConfigAttrib(display, cfg, EGL_BLUE_SIZE, &b)  &&
            eglGetConfigAttrib(display, cfg, EGL_DEPTH_SIZE, &d) &&
            r == 8 && g == 8 && b == 8 && d == 8 ) {

            config = supportedConfigs[i];
            break;
        }
    }
    if (i == numConfigs) {
        config = supportedConfigs[0];
    }

    if (config == nullptr) {
        LOGW("Unable to initialize EGLConfig");
        return -1;
    }

    /* EGL_NATIVE_VISUAL_ID is an attribute of the EGLConfig that is
     * guaranteed to be accepted by ANativeWindow_setBuffersGeometry().
     * As soon as we picked a EGLConfig, we can safely reconfigure the
     * ANativeWindow buffers to match, using EGL_NATIVE_VISUAL_ID. */
    EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE};
    if(eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format) == GL_FALSE)
        LOGI("GLERROR : %d",eglGetError());
    //ANativeWindow_setBuffersGeometry(engine->app->window,engine->width,engine->height,format);
    surface = eglCreateWindowSurface(display, config, engine->app->window, nullptr);
    LOGI("GLERROR : %d",eglGetError());
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
    if(context == EGL_NO_CONTEXT)
        LOGI("context is 0");
    LOGI("GLERROR CONTEXT: %d",eglGetError());


    if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
        LOGW("Unable to eglMakeCurrent");
        return -1;
    }

    eglQuerySurface(display, surface, EGL_WIDTH, &w);
    eglQuerySurface(display, surface, EGL_HEIGHT, &h);

    engine->display = display;
    engine->context = context;
    engine->surface = surface;
    engine->width = w;
    engine->height = h;
    engine->state.angle = 0;

    // Check openGL on the system
    auto opengl_info = {GL_VENDOR, GL_RENDERER, GL_VERSION, GL_EXTENSIONS};
    for (auto name : opengl_info) {
        auto info = glGetString(name);
        LOGI("OpenGL Info: %s", info);
    }

    if(engine->context == EGL_NO_CONTEXT || engine->surface == EGL_NO_SURFACE || engine->display == EGL_NO_DISPLAY)
        LOGI("Error HEre");
    // Initialize GL state.

    const GLchar * vsrc =
        "#version 320 es " \
        "\n" \
        "in vec4 a_position;\n" \
        "out vec4 out_color;\n" \
        "\n" \
        "uniform mat4 u_mvpMatrix;\n" \
        "\n" \
        "void main(void)\n" \
        "{\n"  \
        "   //gl_PointSize = 2.0f;\n" \
        "   gl_Position = u_mvpMatrix * a_position;\n" \
        "}\n";

    GLuint vsobj = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vsobj,1,(const GLchar**)&vsrc,NULL);

    glCompileShader(vsobj);
    GLint status = 0;
    GLint infoLogLength;
    char * log = NULL;

    glGetShaderiv(vsobj, GL_COMPILE_STATUS, &status);
    if(status == GL_FALSE)
    {
        glGetShaderiv(vsobj,GL_INFO_LOG_LENGTH,&infoLogLength);
        if(infoLogLength > 0)
        {
            log = (char*)malloc(infoLogLength);
            if(log != NULL)
            {
                GLsizei written;
                glGetShaderInfoLog(vsobj,infoLogLength,&written,log);
                LOGW("Vertex Shader Error : %s",log);
                free(log);
            }
        }
    }


    const GLchar * fsrc =
    "#version 320 es" \
    "\n" \
    "precision mediump float;\n" \
    "uniform vec4 out_color;\n" \
    "out vec4 FragColor;\n"\
    "\n"\
    "void main(void)\n" \
    "{\n"\
    "   FragColor = out_color;\n"\
    "}\n";

    GLuint fsobj = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fsobj,1,(const GLchar**)&fsrc,NULL);

    glCompileShader(fsobj);
    status = 0;
    infoLogLength = 0;
    log = NULL;

    glGetShaderiv(fsobj, GL_COMPILE_STATUS, &status);
    if(status == GL_FALSE)
    {
        glGetShaderiv(fsobj,GL_INFO_LOG_LENGTH,&infoLogLength);
        if(infoLogLength > 0)
        {
            log = (char*)malloc(infoLogLength);
            if(log != NULL)
            {
                GLsizei written;
                glGetShaderInfoLog(fsobj,infoLogLength,&written,log);
                LOGW("Fragment Shader Error : %s",log);
                free(log);
            }
        }
    }

    program = glCreateProgram();
    glAttachShader(program,vsobj);
    glAttachShader(program,fsobj);
    glBindAttribLocation(program,0,"a_position");
    glLinkProgram(program);

    status = 0;
    infoLogLength = 0;
    log = NULL;

    glGetProgramiv(program,GL_LINK_STATUS,&status);

    if(status == GL_FALSE)
    {
        glGetProgramiv(program,GL_INFO_LOG_LENGTH,&infoLogLength);
        if(infoLogLength > 0)
        {
            log = (char*)malloc(infoLogLength);
            if(log != NULL)
            {
                GLsizei written;
                glGetProgramInfoLog(program,infoLogLength,&written,log);
                LOGW("Program Linking Error : %s",log);
                free(log);
            }
        }
    }

    mvp = glGetUniformLocation(program,"u_mvpMatrix");
    color = glGetUniformLocation(program,"out_color");

    // Initialize OpenCL
    char chBuffer[1024];

    cl_uint num_platforms;
    cl_platform_id* clPlatformIDs;

    ciErrNum = clGetPlatformIDs(0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS || num_platforms <= 0)
    {
        LOGI("Failed to Find OpenCL Platforms\n");
        //return -5;
    }
    else
    {
        LOGI("OpenCL platforms : %d\n",num_platforms);
        if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
        {

            LOGI("Failed to allocate memory for cl_platform ID's!\n\n");
            return -8;
        }

        ciErrNum = clGetPlatformIDs(num_platforms, clPlatformIDs, NULL);
        for (cl_uint j = 0; j < num_platforms; ++j)
        {
            ciErrNum = clGetPlatformInfo(clPlatformIDs[j], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
            LOGI("OpenCL platforms names : %s\n",chBuffer);
            if (ciErrNum == CL_SUCCESS)
            {
                if (strstr(chBuffer, "ARM") != NULL)
                {
                    cpPlatform = clPlatformIDs[j];
                    break;
                }
            }
        }

        if (cpPlatform == NULL)
        {
            LOGI("Pltform Not Founc\n\n");
            cpPlatform = clPlatformIDs[0];
        }
        free(clPlatformIDs);
    }

    // Get The Number Of GPU devices Available to the platform

    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiDevCount);
    LOGI("OpenCL platforms : %d\n",uiDevCount);
    if (ciErrNum != CL_SUCCESS)
    {
        LOGI("clGetDeviceIDs failed\n");
        return -7;
    }

    cdDevices = new cl_device_id[uiDevCount];
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, cdDevices, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        LOGI("clGetDeviceIDs failed\n");
        return -7;
    }

    char bufferExt[728];
    size_t size_ext;
    clGetDeviceInfo(cdDevices[0], CL_DEVICE_EXTENSIONS, 728, &bufferExt, &size_ext);
    LOGI("Device Size = %zu\n", size_ext);
    LOGI("Device Extensions = %s\n", bufferExt);

    cl_context_properties props[] =
            {
                    CL_CONTEXT_PLATFORM,(cl_context_properties)cpPlatform,
                    0
            };
    // cxGPUcontext = clCreateContext(props, 1, &cdDevices[0], NULL, NULL, &ciErrNum);
     cxGPUcontext = clCreateContextFromType(props,CL_DEVICE_TYPE_GPU, nullptr, nullptr,&ciErrNum);
     //cxGPUcontext = clCreateContext(0,1,&cdDevices[0],NULL,NULL,&ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        LOGW("clCreateContext failed : %d\n",ciErrNum);
        //exit(1);
        //return -7;
    }
    else
        LOGW("clCreate Contextr Success \n\n");

    clGetDeviceInfo(cdDevices[0], CL_DEVICE_NAME, sizeof(chBuffer), &chBuffer, NULL);
    LOGI("Selected GPU = %s\n\n", chBuffer);


    cqCommandQueue = clCreateCommandQueueWithProperties(cxGPUcontext,cdDevices[0],0,&ciErrNum);
    if(ciErrNum != CL_SUCCESS)
    {
        LOGI("clCommandQueue Failed %d\n",ciErrNum);
    }

    // OpenCL Program Setup

    const char *sinwave =
        "__kernel void sine_wave(__global float4* pos, unsigned long width, unsigned long height, float time)\n" \
        "{\n" \
        "unsigned int x = get_global_id(0);\n" \
        "unsigned int y = get_global_id(1);\n" \
        "float u = x / (float) width;\n" \
        "float v = y / (float) height;\n" \
        "u = u*2.0f - 1.0f;\n" \
        "v = v*3.0f - 1.0f;\n" \
        "float freq = 4.0f;\n" \
        "float w = sin(u*freq + time) * cos(v*freq + time) * 0.5f;\n" \
        "pos[y*width+x] = (float4)(u, w, v, 1.0f);\n" \
        "}\n";

    cpProgram = clCreateProgramWithSource(cxGPUcontext,1,(const char **)&sinwave,NULL,&ciErrNum);
    if(ciErrNum != CL_SUCCESS)
    {
        LOGI("clCreateProgram With Source Failed\n");
    }

    ciErrNum = clBuildProgram(cpProgram,0, nullptr,"-cl-fast-relaxed-math", nullptr, nullptr);
    if(ciErrNum != CL_SUCCESS)
    {
        char buildLog[2048];
        clGetProgramBuildInfo(cpProgram,cdDevices[0],CL_PROGRAM_BUILD_LOG,sizeof(buildLog),buildLog,nullptr);
        LOGI("Error In OpenCL Kernel : %s",buildLog);
        clReleaseProgram(cpProgram);
    }

    // Create OpenCL Kernel

    ckKernel = clCreateKernel(cpProgram,"sine_wave",&ciErrNum);
    if(ciErrNum != CL_SUCCESS)
    {
        LOGI("clCreateKernel Failed %d \n",ciErrNum);
    }
    // create vao & vbo

    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo_cpu);
    glGenBuffers(1,&vbo_gpu);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo_cpu);
    buffer = (GLfloat*) malloc(mesh_width * mesh_height * 4 * sizeof(GLfloat));
    glBufferData(GL_ARRAY_BUFFER,mesh_width * mesh_height * 4 * sizeof(GLfloat),NULL,GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);

    runCPU();

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo_gpu);
    //buffer = (GLfloat*) malloc(mesh_width * mesh_height * 4 * sizeof(GLfloat));
    clBuffer = (GLfloat*) malloc(mesh_width * mesh_height * 4 * sizeof(GLfloat));
    glBufferData(GL_ARRAY_BUFFER,mesh_width * mesh_height * 4 * sizeof(GLfloat),NULL,GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);

    vbo_cl = clCreateBuffer(cxGPUcontext,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(GLfloat) * mesh_width * mesh_height * 4,clBuffer,&ciErrNum);
    //svbo_cl = clCreateFromGLBuffer(cxGPUcontext,CL_MEM_WRITE_ONLY,vbo_gpu,&ciErrNum);
    if(ciErrNum != CL_SUCCESS)
    {
        LOGI("Error Creating CL Buffer from GL Error %d",ciErrNum);
    }
    LOGI("No OpenCL error so Far !!!");
    runGPU();

    glViewport(0,0,engine->width,engine->height);
    perspectiveProjection = vmath::perspective(45.0f,(GLfloat)engine->width/(GLfloat)engine->height,0.1f,100.0f);
    glClearColor(0.0f,0.0f,0.0f, 1);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    //LOGI("context %p, surface = %p, display = %p\n\n",engine->context,engine->surface,engine->display);

    if(engine->context == EGL_NO_CONTEXT || engine->surface == EGL_NO_SURFACE || engine->display == EGL_NO_DISPLAY)
        LOGI("Error HEre");

    return 0;
}

void runCPU()
{
    float freq = 4.0f;
    for(int y = 0; y < mesh_height; y++)
    {
        for(int x = 0; x < mesh_width; x++)
        {
            int offset = x + y * mesh_width;
            GLfloat u = (GLfloat) x / (GLfloat) mesh_width;
            GLfloat v = (GLfloat) y / (GLfloat) mesh_height;
            GLfloat w = sinf(u * freq + mytime) * cosf(v * freq + mytime) * 1.0f;

            u = u * 2.0f - 1.0f;
            v = v * 3.0f - 1.0f;

            buffer[offset * 4 + 0] = (GLfloat)u;
            buffer[offset * 4 + 1] = (GLfloat)w;
            buffer[offset * 4 + 2] = (GLfloat)v;
            buffer[offset * 4 + 3] = (GLfloat)1.0f;
        }
    }

    size_t size = mesh_width * mesh_height * 4 * sizeof(GLfloat);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo_cpu);
    glBufferData(GL_ARRAY_BUFFER,size,buffer,GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);
}

void runGPU()
{
    ciErrNum = CL_SUCCESS;
    clFinish(cqCommandQueue);

    ciErrNum = clSetKernelArg(ckKernel,0,sizeof(cl_mem),(void *)&vbo_cl);
    if(ciErrNum != CL_SUCCESS)
    {
        LOGI("1 Set Kernel Args Or ND Kernel Range Failed %d\n",ciErrNum);
    }
    ciErrNum = clSetKernelArg(ckKernel,1,sizeof(unsigned long),&mesh_width);
    if(ciErrNum != CL_SUCCESS)
    {
        LOGI("2 Set Kernel Args Or ND Kernel Range Failed %d\n",ciErrNum);
    }
    ciErrNum = clSetKernelArg(ckKernel,2,sizeof(unsigned  long),&mesh_height);
    if(ciErrNum != CL_SUCCESS)
    {
        LOGI("3 Set Kernel Args Or ND Kernel Range Failed %d\n",ciErrNum);
    }
    ciErrNum = clSetKernelArg(ckKernel,3,sizeof(float),&mytime);
    if(ciErrNum != CL_SUCCESS)
    {
        LOGI("4 Set Kernel Args Or ND Kernel Range Failed %d\n",ciErrNum);
    }

    ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,vbo_cl,CL_TRUE,0,mesh_width*mesh_height*4*sizeof(float),clBuffer,0,0,0);

    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue,ckKernel,2, NULL,szGlobalWorkSize, nullptr,0,NULL,NULL);
    if(ciErrNum != CL_SUCCESS)
    {
        LOGI("5 Set Kernel Args Or ND Kernel Range Failed %d\n",ciErrNum);
    }

    ciErrNum = clEnqueueReadBuffer(cqCommandQueue,vbo_cl,CL_TRUE,0,mesh_width*mesh_height*4*sizeof(float),clBuffer,0,NULL,NULL);

    if(ciErrNum != CL_SUCCESS)
    {
        LOGI("Error Reading Result Buffer %d",ciErrNum);
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo_gpu);
    glBufferData(GL_ARRAY_BUFFER,mesh_width * mesh_height * 4 * sizeof(GLfloat),clBuffer,GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);
}

/**
 * Just the current frame in the display.
 */
static void engine_draw_frame(struct engine* engine) {
    if (engine->display == nullptr) {
        // No display.
        return;
    }

    // Just fill the screen with a color.

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if(mytime >= 360.0f)
        mytime = 0.0f;

    mytime += 0.01f;

    if(onGPU)
        runGPU();
    else
        runCPU();

    vmath::mat4 model = vmath::mat4::identity();
    vmath::mat4 mvpMatrix = vmath::mat4::identity();

    model = vmath::translate(0.0f,0.0f,-2.0f);
    mvpMatrix = perspectiveProjection * model;

    glUseProgram(program);
    glUniformMatrix4fv(mvp,1,GL_FALSE,mvpMatrix);

    glBindVertexArray(vao);
    if(onGPU)
    {
        glUniform4f(color, 1.0f, 0.0, 0.0f, 1.0f);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu);
    }
    else
    {
        glUniform4f(color,0.0f,1.0,0.0f,1.0f);
        glBindBuffer(GL_ARRAY_BUFFER,vbo_cpu);
    }

    glDrawArrays(GL_POINTS,0,mesh_width * mesh_height);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);
    glUseProgram(0);

    //if(engine->context != EGL_NO_CONTEXT && engine->surface == EGL_BAD_SURFACE)
   // LOGI("context %p, surface = %p, display = %p\n\n",engine->context,engine->surface,engine->display);
}

/**
 * Tear down the EGL context currently associated with the display.
 */
static void engine_term_display(struct engine* engine) {
    if (engine->display != EGL_NO_DISPLAY) {
        eglMakeCurrent(engine->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (engine->context != EGL_NO_CONTEXT) {
            eglDestroyContext(engine->display, engine->context);
        }
        if (engine->surface != EGL_NO_SURFACE) {
            eglDestroySurface(engine->display, engine->surface);
            engine->surface = EGL_NO_SURFACE;
        }
        eglTerminate(engine->display);
    }

    if(vbo_cpu) glDeleteBuffers(1,&vbo_cpu);
    if(vao) glDeleteVertexArrays(1,&vao);

    if(program)
    {
        glUseProgram(program);

        GLsizei num_attached_shaders;
        glGetProgramiv(program,GL_ATTACHED_SHADERS,&num_attached_shaders);
        GLuint * shader_objects = NULL;

        shader_objects = (GLuint*)malloc(num_attached_shaders);

        glGetAttachedShaders(program,num_attached_shaders,&num_attached_shaders,shader_objects);

        for(GLsizei i = 0; i < num_attached_shaders; i++)
        {
            glDetachShader(program,shader_objects[i]);
            glDeleteShader(shader_objects[i]);
            shader_objects[i] = 0;
        }
        free(shader_objects);
        glUseProgram(0);
        glDeleteProgram(program);
    }

    engine->animating = 0;
    engine->display = EGL_NO_DISPLAY;
    engine->context = EGL_NO_CONTEXT;
    engine->surface = EGL_NO_SURFACE;
}

/**
 * Process the next input event.
 */
static int32_t engine_handle_input(struct android_app* app, AInputEvent* event) {
    //auto* engine = (struct engine*)app->userData;
    switch (AInputEvent_getType(event)) 
    {
        //engine->animating = 1;
        case AINPUT_EVENT_TYPE_MOTION:
        switch (AInputEvent_getSource(event))
        {
            case AINPUT_SOURCE_TOUCHSCREEN:
            int action = AKeyEvent_getAction(event) & AMOTION_EVENT_ACTION_MASK;
            switch(action)
            {
                case AMOTION_EVENT_ACTION_DOWN:

                    mesh_width = mesh_width * 2;
                    mesh_height = mesh_height * 2;

                    szGlobalWorkSize[0] = mesh_width;
                    szGlobalWorkSize[1] = mesh_height;

                    buffer = (GLfloat*) realloc(buffer,mesh_width*mesh_height*4*sizeof(GLfloat));
                    clBuffer = (GLfloat*) realloc(clBuffer,mesh_width*mesh_height*4*sizeof(GLfloat));
                    LOGI("width %lu height %lu\n",mesh_width,mesh_height);

                    vbo_cl = clCreateBuffer(cxGPUcontext,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(GLfloat) * mesh_width * mesh_height * 4,clBuffer,&ciErrNum);
                    if(ciErrNum != CL_SUCCESS)
                    {
                        LOGI("Error Creating CL Buffer from GL Error %d",ciErrNum);
                    }


                break;
                case AMOTION_EVENT_ACTION_MOVE:

                break;
            }
            break;
        }
        return 1;
        break;
        case AINPUT_EVENT_TYPE_KEY:
            int action = AKeyEvent_getAction(event) & AMOTION_EVENT_ACTION_MASK;
            LOGI("Volume Keypressed\n");
            if(action == 0)
            {
                onGPU = !onGPU;
                LOGI("%d",onGPU);
            }
            break;
    }

    return 0;
}

/**
 * Process the next main command.
 */
static void engine_handle_cmd(struct android_app* app, int32_t cmd) {
    auto* engine = (struct engine*)app->userData;
    switch (cmd) {
        case APP_CMD_SAVE_STATE:
            // The system has asked us to save our current state.  Do so.
            engine->app->savedState = malloc(sizeof(struct saved_state));
            *((struct saved_state*)engine->app->savedState) = engine->state;
            engine->app->savedStateSize = sizeof(struct saved_state);
            break;
        case APP_CMD_INIT_WINDOW:
            // The window is being shown, get it ready.
            if (engine->app->window != nullptr) {
                engine_init_display(engine);
                engine_draw_frame(engine);
            }
            break;
        case APP_CMD_TERM_WINDOW:
            // The window is being hidden or closed, clean it up.
            engine_term_display(engine);
            break;
        case APP_CMD_WINDOW_REDRAW_NEEDED:
            //engine_draw_frame(engine);
            break;
        case APP_CMD_GAINED_FOCUS:
            // When our app gains focus, we start monitoring the accelerometer.

            break;
        case APP_CMD_LOST_FOCUS:
            // When our app loses focus, we stop monitoring the accelerometer.
            // This is to avoid consuming battery while not being used.

            // Also stop animating.
            engine->animating = 0;
            engine_draw_frame(engine);
            break;
        default:
            break;
    }
}

/**
 * This is the main entry point of a native application that is using
 * android_native_app_glue.  It runs in its own thread, with its own
 * event loop for receiving input events and doing other things.
 */
void android_main(struct android_app* state) {
    struct engine engine{};

    //dlopen("liblzma.so",RTLD_LAZY);
    //dlopen("libahgm4.so",RTLD_LAZY);
    //android_dlopen_ext();

    memset(&engine, 0, sizeof(engine));
    state->userData = &engine;
    state->onAppCmd = engine_handle_cmd;
    state->onInputEvent = engine_handle_input;
    engine.app = state;

    // Prepare to monitor accelerometer

    if (state->savedState != nullptr) {
        // We are starting with a previous saved state; restore from it.
        engine.state = *(struct saved_state*)state->savedState;
    }

    // loop waiting for stuff to do.

    while (true) {
        // Read all pending events.
        int ident;
        int events;
        struct android_poll_source* source;

        // If not animating, we will block forever waiting for events.
        // If animating, we loop until all events are read, then continue
        // to draw the next frame of animation.

        while ((ident = ALooper_pollAll(0 , nullptr, &events,(void**)&source)) >= 0)
        {

            // Process this event.
            if (source != nullptr) {
                source->process(state, source);
            }

            // If a sensor has data, process it now.

            // Check if we are exiting.
            if (state->destroyRequested != 0) {
                engine_term_display(&engine);
                return;
            }
            // Drawing is throttled to the screen update rate, so there
            // is no need to do timing here.
        }
        engine_draw_frame(&engine);
        glFlush();
        eglSwapBuffers(eglGetCurrentDisplay(), eglGetCurrentSurface(EGL_DRAW));
        //LOGI("GLERROR : %x",eglGetError());
    }
}

//END_INCLUDE(all)
