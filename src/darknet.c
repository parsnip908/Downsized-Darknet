#include "darknet.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include "parser.h"
#include "utils.h"
#include "blas.h"
#include "connected_layer.h"


int main(int argc, char **argv)
{
#ifdef _DEBUG
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    printf(" _DEBUG is used \n");
#endif

#ifdef DEBUG
    printf(" DEBUG=1 \n");
#endif

    int i;
    for (i = 0; i < argc; ++i) {
        if (!argv[i]) continue;
        strip_args(argv[i]);
    }

    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }

    gpu_index = find_int_arg(argc, argv, "-i", 0);

    gpu_index = -1;
    printf(" GPU isn't used \n");
    init_cpu();

    show_opencv_info();

    float thresh = find_float_arg(argc, argv, "-thresh", .24);
    int ext_output = find_arg(argc, argv, "-ext_output");
    char *filename = (argc > 4) ? argv[4]: 0;
    test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, 0.5, 0, ext_output, 0, NULL, 0, 0);
    return 0;
}
