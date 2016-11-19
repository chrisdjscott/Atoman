#include "osx_helper.h"

extern "C"
int osx_retina_hack(long winId)
{
    disableGLHiDPI(winId);

    return 0;
}

