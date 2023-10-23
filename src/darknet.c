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

    init_cpu();

    run_detector(argc, argv);
    return 0;
}
