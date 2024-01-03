#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>

#include <fcntl.h>
#include <unistd.h>

#include <sys/mman.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dlsym_wrapper.h"

static pthread_mutex_t global_mutex = PTHREAD_MUTEX_INITIALIZER;

static volatile DLSYM_PROC_T real_dlsym;
static void * (* volatile real_glXGetProcAddressARB)(const char *);
static void (* volatile real_glXSwapBuffers)(void *, void *);
static void (* volatile real_glReadPixels)(int, int, int, int, int, int, void *);

static volatile int *ctl_buf;
static char *img_buf;

static void init_shm() {
    int ctl = shm_open("ktane_ctl", O_RDWR|O_CREAT|O_TRUNC, 0666);
    int img = shm_open("ktane_img", O_RDWR|O_CREAT|O_TRUNC, 0666);

    ftruncate(ctl, 4);
    ftruncate(img, 1920*1080*3);

    ctl_buf = mmap(NULL, 4, PROT_READ|PROT_WRITE, MAP_SHARED, ctl, 0);
    img_buf = mmap(NULL, 1920*1080*3, PROT_READ|PROT_WRITE, MAP_SHARED, img, 0);
}

static DLSYM_PROC_T get_dlsym() {
    return (DLSYM_PROC_T)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
}

extern void glXSwapBuffers(void *dpy, void *drawable) {
    if (!real_glXSwapBuffers) {
        pthread_mutex_lock(&global_mutex);
        if (!real_glXSwapBuffers) {
            real_glXSwapBuffers = dlsym(RTLD_NEXT, "glXSwapBuffers");
            real_glReadPixels = dlsym(RTLD_NEXT, "glReadPixels");

            init_shm();
        }
        pthread_mutex_unlock(&global_mutex);
    }

    if (*ctl_buf == 1) {
        real_glReadPixels(0, 0, 1920, 1080, 0x80E0, 0x1401, img_buf);
        *ctl_buf = 0;
    }
    
    return real_glXSwapBuffers(dpy, drawable);
}

extern void *glXGetProcAddressARB(const char *name) {
    if (!real_glXGetProcAddressARB) {
        pthread_mutex_lock(&global_mutex);
        if (!real_glXGetProcAddressARB)
            real_glXGetProcAddressARB = real_dlsym(RTLD_NEXT, "glXGetProcAddressARB");
        pthread_mutex_unlock(&global_mutex);
    }

    if (!strcmp(name, "glXSwapBuffers"))
        return glXSwapBuffers;

    return real_glXGetProcAddressARB(name);
}

extern void *dlsym(void *handle, const char *name) {
    if (!real_dlsym) {
        pthread_mutex_lock(&global_mutex);
        if (!real_dlsym)
            real_dlsym = get_dlsym();
        pthread_mutex_unlock(&global_mutex);
    }

    if (!strcmp(name, "glXGetProcAddressARB"))
        return glXGetProcAddressARB;
    
    void *r = real_dlsym(handle, name);
    return r;
}
