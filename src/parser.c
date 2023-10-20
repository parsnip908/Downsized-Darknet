#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "activations.h"
#include "assert.h"
#include "blas.h"
#include "convolutional_layer.h"
#include "list.h"
#include "option_list.h"
#include "parser.h"
#include "utils.h"
#include "version.h"

typedef struct{
    char *type;
    list *options;
}section;

list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, "[scale_channels]") == 0) return SCALE_CHANNELS;
    if (strcmp(type, "[sam]") == 0) return SAM;
    if (strcmp(type, "[crop]")==0) return CROP;
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[detection]")==0) return DETECTION;
    if (strcmp(type, "[region]")==0) return REGION;
    if (strcmp(type, "[yolo]") == 0) return YOLO;
    if (strcmp(type, "[Gaussian_yolo]") == 0) return GAUSSIAN_YOLO;
    if (strcmp(type, "[local]")==0) return LOCAL;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[crnn]")==0) return CRNN;
    if (strcmp(type, "[gru]")==0) return GRU;
    if (strcmp(type, "[lstm]")==0) return LSTM;
    if (strcmp(type, "[conv_lstm]") == 0) return CONV_LSTM;
    if (strcmp(type, "[history]") == 0) return HISTORY;
    if (strcmp(type, "[rnn]")==0) return RNN;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return CONNECTED;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[local_avg]") == 0
        || strcmp(type, "[local_avgpool]") == 0) return LOCAL_AVGPOOL;
    if (strcmp(type, "[reorg3d]")==0) return REORG;
    if (strcmp(type, "[reorg]") == 0) return REORG_OLD;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[lrn]")==0
            || strcmp(type, "[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[contrastive]") == 0) return CONTRASTIVE;
    if (strcmp(type, "[route]")==0) return ROUTE;
    if (strcmp(type, "[upsample]") == 0) return UPSAMPLE;
    if (strcmp(type, "[empty]") == 0
        || strcmp(type, "[silence]") == 0) return EMPTY;
    if (strcmp(type, "[implicit]") == 0) return IMPLICIT;
    return BLANK;
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void parse_data(char *data, float *a, int n)
{
    int i;
    if(!data) return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for(i = 0; i < n && !done; ++i){
        while(*++next !='\0' && *next != ',');
        if(*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    int train;
    network net;
} size_params;

convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int groups = option_find_int_quiet(options, "groups", 1);
    int size = option_find_int(options, "size",1);
    int stride = -1;
    //int stride = option_find_int(options, "stride",1);
    int stride_x = option_find_int_quiet(options, "stride_x", -1);
    int stride_y = option_find_int_quiet(options, "stride_y", -1);
    if (stride_x < 1 || stride_y < 1) {
        stride = option_find_int(options, "stride", 1);
        if (stride_x < 1) stride_x = stride;
        if (stride_y < 1) stride_y = stride;
    }
    else {
        stride = option_find_int_quiet(options, "stride", 1);
    }
    int dilation = option_find_int_quiet(options, "dilation", 1);
    int antialiasing = option_find_int_quiet(options, "antialiasing", 0);
    if (size == 1) dilation = 1;
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int assisted_excitation = option_find_float_quiet(options, "assisted_excitation", 0);

    int share_index = option_find_int_quiet(options, "share_index", -1000000000);
    convolutional_layer *share_layer = NULL;
    if(share_index >= 0) share_layer = &params.net.layers[share_index];
    else if(share_index != -1000000000) share_layer = &params.net.layers[params.index + share_index];

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.", DARKNET_LOC);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int cbn = option_find_int_quiet(options, "cbn", 0);
    if (cbn) batch_normalize = 2;
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);
    int use_bin_output = option_find_int_quiet(options, "bin_output", 0);
    int sway = option_find_int_quiet(options, "sway", 0);
    int rotate = option_find_int_quiet(options, "rotate", 0);
    int stretch = option_find_int_quiet(options, "stretch", 0);
    int stretch_sway = option_find_int_quiet(options, "stretch_sway", 0);
    if ((sway + rotate + stretch + stretch_sway) > 1) {
        error("Error: should be used only 1 param: sway=1, rotate=1 or stretch=1 in the [convolutional] layer", DARKNET_LOC);
    }
    int deform = sway || rotate || stretch || stretch_sway;
    if (deform && size == 1) {
        error("Error: params (sway=1, rotate=1 or stretch=1) should be used only with size >=3 in the [convolutional] layer", DARKNET_LOC);
    }

    convolutional_layer layer = make_convolutional_layer(batch,1,h,w,c,n,groups,size,stride_x,stride_y,dilation,padding,activation, batch_normalize, binary, xnor, params.net.adam, use_bin_output, params.index, antialiasing, share_layer, assisted_excitation, deform, params.train);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);
    layer.sway = sway;
    layer.rotate = rotate;
    layer.stretch = stretch;
    layer.stretch_sway = stretch_sway;
    layer.angle = option_find_float_quiet(options, "angle", 15);
    layer.grad_centr = option_find_int_quiet(options, "grad_centr", 0);
    layer.reverse = option_find_float_quiet(options, "reverse", 0);
    layer.coordconv = option_find_int_quiet(options, "coordconv", 0);

    layer.stream = option_find_int_quiet(options, "stream", -1);
    layer.wait_stream_id = option_find_int_quiet(options, "wait_stream", -1);

    if(params.net.adam){
        layer.B1 = params.net.B1;
        layer.B2 = params.net.B2;
        layer.eps = params.net.eps;
    }

    return layer;
}


int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == '#') break;
            if (a[i] == ',') ++n;
        }
        mask = (int*)xcalloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
        }
        *num = n;
    }
    return mask;
}

float *get_classes_multipliers(char *cpc, const int classes, const float max_delta)
{
    float *classes_multipliers = NULL;
    if (cpc) {
        int classes_counters = classes;
        int *counters_per_class = parse_yolo_mask(cpc, &classes_counters);
        if (classes_counters != classes) {
            printf(" number of values in counters_per_class = %d doesn't match with classes = %d \n", classes_counters, classes);
            error("Error!", DARKNET_LOC);
        }
        float max_counter = 0;
        int i;
        for (i = 0; i < classes_counters; ++i) {
            if (counters_per_class[i] < 1) counters_per_class[i] = 1;
            if (max_counter < counters_per_class[i]) max_counter = counters_per_class[i];
        }
        classes_multipliers = (float *)calloc(classes_counters, sizeof(float));
        for (i = 0; i < classes_counters; ++i) {
            classes_multipliers[i] = max_counter / counters_per_class[i];
            if(classes_multipliers[i] > max_delta) classes_multipliers[i] = max_delta;
        }
        free(counters_per_class);
        printf(" classes_multipliers: ");
        for (i = 0; i < classes_counters; ++i) printf("%.1f, ", classes_multipliers[i]);
        printf("\n");
    }
    return classes_multipliers;
}


int *parse_gaussian_yolo_mask(char *a, int *num) // Gaussian_YOLOv3
{
    int *mask = 0;
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == '#') break;
            if (a[i] == ',') ++n;
        }
        mask = (int *)calloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
        }
        *num = n;
    }
    return mask;
}


void parse_net_options(list *options, network *net)
{
    net->max_batches = option_find_int(options, "max_batches", 0);
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->learning_rate_min = option_find_float_quiet(options, "learning_rate_min", .00001);
    net->batches_per_cycle = option_find_int_quiet(options, "sgdr_cycle", net->max_batches);
    net->batches_cycle_mult = option_find_int_quiet(options, "sgdr_mult", 2);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->track = option_find_int_quiet(options, "track", 0);
    net->augment_speed = option_find_int_quiet(options, "augment_speed", 2);
    net->init_sequential_subdivisions = net->sequential_subdivisions = option_find_int_quiet(options, "sequential_subdivisions", subdivs);
    if (net->sequential_subdivisions > subdivs) net->init_sequential_subdivisions = net->sequential_subdivisions = subdivs;
    net->try_fix_nan = option_find_int_quiet(options, "try_fix_nan", 0);
    net->batch /= subdivs;          // mini_batch
    const int mini_batch = net->batch;
    net->batch *= net->time_steps;  // mini_batch * time_steps
    net->subdivisions = subdivs;    // number of mini_batches

    net->weights_reject_freq = option_find_int_quiet(options, "weights_reject_freq", 0);
    net->equidistant_point = option_find_int_quiet(options, "equidistant_point", 0);
    net->badlabels_rejection_percentage = option_find_float_quiet(options, "badlabels_rejection_percentage", 0);
    net->num_sigmas_reject_badlabels = option_find_float_quiet(options, "num_sigmas_reject_badlabels", 0);
    net->ema_alpha = option_find_float_quiet(options, "ema_alpha", 0);
    *net->badlabels_reject_threshold = 0;
    *net->delta_rolling_max = 0;
    *net->delta_rolling_avg = 0;
    *net->delta_rolling_std = 0;
    *net->seen = 0;
    *net->cur_iteration = 0;
    *net->cuda_graph_ready = 0;
    net->use_cuda_graph = option_find_int_quiet(options, "use_cuda_graph", 0);
    net->loss_scale = option_find_float_quiet(options, "loss_scale", 1);
    net->dynamic_minibatch = option_find_int_quiet(options, "dynamic_minibatch", 0);
    net->optimized_memory = option_find_int_quiet(options, "optimized_memory", 0);
    net->workspace_size_limit = (size_t)1024*1024 * option_find_float_quiet(options, "workspace_size_limit_MB", 1024);  // 1024 MB by default


    net->adam = option_find_int_quiet(options, "adam", 0);
    if(net->adam){
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .000001);
    }

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, "min_crop",net->w);
    net->flip = option_find_int_quiet(options, "flip", 1);
    net->blur = option_find_int_quiet(options, "blur", 0);
    net->gaussian_noise = option_find_int_quiet(options, "gaussian_noise", 0);
    net->mixup = option_find_int_quiet(options, "mixup", 0);
    int cutmix = option_find_int_quiet(options, "cutmix", 0);
    int mosaic = option_find_int_quiet(options, "mosaic", 0);
    if (mosaic && cutmix) net->mixup = 4;
    else if (cutmix) net->mixup = 2;
    else if (mosaic) net->mixup = 3;
    net->letter_box = option_find_int_quiet(options, "letter_box", 0);
    net->mosaic_bound = option_find_int_quiet(options, "mosaic_bound", 0);
    net->contrastive = option_find_int_quiet(options, "contrastive", 0);
    net->contrastive_jit_flip = option_find_int_quiet(options, "contrastive_jit_flip", 0);
    net->contrastive_color = option_find_int_quiet(options, "contrastive_color", 0);
    net->unsupervised = option_find_int_quiet(options, "unsupervised", 0);
    if (net->contrastive && mini_batch < 2) {
        error("Error: mini_batch size (batch/subdivisions) should be higher than 1 for Contrastive loss!", DARKNET_LOC);
    }
    net->label_smooth_eps = option_find_float_quiet(options, "label_smooth_eps", 0.0f);
    net->resize_step = option_find_float_quiet(options, "resize_step", 32);
    net->attention = option_find_int_quiet(options, "attention", 0);
    net->adversarial_lr = option_find_float_quiet(options, "adversarial_lr", 0);
    net->max_chart_loss = option_find_float_quiet(options, "max_chart_loss", 20.0);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);
    net->power = option_find_float_quiet(options, "power", 4);

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied", DARKNET_LOC);

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
#ifdef GPU
    if (net->gpu_index >= 0) {
        char device_name[1024];
        int compute_capability = get_gpu_compute_capability(net->gpu_index, device_name);
#ifdef CUDNN_HALF
        if (compute_capability >= 700) net->cudnn_half = 1;
        else net->cudnn_half = 0;
#endif// CUDNN_HALF
        fprintf(stderr, " %d : compute_capability = %d, cudnn_half = %d, GPU: %s \n", net->gpu_index, compute_capability, net->cudnn_half, device_name);
    }
    else fprintf(stderr, " GPU isn't used \n");
#endif// GPU
    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    } else if (net->policy == STEPS || net->policy == SGDR){
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        char *s = option_find(options, "seq_scales");
        if(net->policy == STEPS && (!l || !p)) error("STEPS policy must have steps and scales in cfg file", DARKNET_LOC);

        if (l) {
            int len = strlen(l);
            int n = 1;
            int i;
            for (i = 0; i < len; ++i) {
                if (l[i] == '#') break;
                if (l[i] == ',') ++n;
            }
            int* steps = (int*)xcalloc(n, sizeof(int));
            float* scales = (float*)xcalloc(n, sizeof(float));
            float* seq_scales = (float*)xcalloc(n, sizeof(float));
            for (i = 0; i < n; ++i) {
                float scale = 1.0;
                if (p) {
                    scale = atof(p);
                    p = strchr(p, ',') + 1;
                }
                float sequence_scale = 1.0;
                if (s) {
                    sequence_scale = atof(s);
                    s = strchr(s, ',') + 1;
                }
                int step = atoi(l);
                l = strchr(l, ',') + 1;
                steps[i] = step;
                scales[i] = scale;
                seq_scales[i] = sequence_scale;
            }
            net->scales = scales;
            net->steps = steps;
            net->seq_scales = seq_scales;
            net->num_steps = n;
        }
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
        //net->power = option_find_float(options, "power", 1);
    }

}

int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}

void set_train_only_bn(network net)
{
    int train_only_bn = 0;
    int i;
    for (i = net.n - 1; i >= 0; --i) {
        if (net.layers[i].train_only_bn) train_only_bn = net.layers[i].train_only_bn;  // set l.train_only_bn for all previous layers
        if (train_only_bn) {
            net.layers[i].train_only_bn = train_only_bn;

            if (net.layers[i].type == CONV_LSTM) {
                net.layers[i].wf->train_only_bn = train_only_bn;
                net.layers[i].wi->train_only_bn = train_only_bn;
                net.layers[i].wg->train_only_bn = train_only_bn;
                net.layers[i].wo->train_only_bn = train_only_bn;
                net.layers[i].uf->train_only_bn = train_only_bn;
                net.layers[i].ui->train_only_bn = train_only_bn;
                net.layers[i].ug->train_only_bn = train_only_bn;
                net.layers[i].uo->train_only_bn = train_only_bn;
                if (net.layers[i].peephole) {
                    net.layers[i].vf->train_only_bn = train_only_bn;
                    net.layers[i].vi->train_only_bn = train_only_bn;
                    net.layers[i].vo->train_only_bn = train_only_bn;
                }
            }
            else if (net.layers[i].type == CRNN) {
                net.layers[i].input_layer->train_only_bn = train_only_bn;
                net.layers[i].self_layer->train_only_bn = train_only_bn;
                net.layers[i].output_layer->train_only_bn = train_only_bn;
            }
        }
    }
}

network parse_network_cfg(char *filename)
{
    return parse_network_cfg_custom(filename, 0, 0);
}

network parse_network_cfg_custom(char *filename, int batch, int time_steps)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections", DARKNET_LOC);
    network net = make_network(sections->size - 1);
    net.gpu_index = gpu_index;
    size_params params;

    if (batch > 0) params.train = 0;    // allocates memory for Inference only
    else params.train = 1;              // allocates memory for Inference & Training

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]", DARKNET_LOC);
    parse_net_options(options, &net);

#ifdef GPU
    printf("net.optimized_memory = %d \n", net.optimized_memory);
    if (net.optimized_memory >= 2 && params.train) {
        pre_allocate_pinned_memory((size_t)1024 * 1024 * 1024 * 8);   // pre-allocate 8 GB CPU-RAM for pinned memory
    }
#endif  // GPU

    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    if (batch > 0) net.batch = batch;
    if (time_steps > 0) net.time_steps = time_steps;
    if (net.batch < 1) net.batch = 1;
    if (net.time_steps < 1) net.time_steps = 1;
    if (net.batch < net.time_steps) net.batch = net.time_steps;
    params.batch = net.batch;
    params.time_steps = net.time_steps;
    params.net = net;
    printf("mini_batch = %d, batch = %d, time_steps = %d, train = %d \n", net.batch, net.batch * net.subdivisions, net.time_steps, params.train);

    int last_stop_backward = -1;
    int avg_outputs = 0;
    int avg_counter = 0;
    float bflops = 0;
    size_t workspace_size = 0;
    size_t max_inputs = 0;
    size_t max_outputs = 0;
    int receptive_w = 1, receptive_h = 1;
    int receptive_w_scale = 1, receptive_h_scale = 1;
    const int show_receptive_field = option_find_float_quiet(options, "show_receptive_field", 0);

    n = n->next;
    int count = 0;
    free_section(s);

    // find l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
    node *n_tmp = n;
    int count_tmp = 0;
    if (params.train == 1) {
        while (n_tmp) {
            s = (section *)n_tmp->val;
            options = s->options;
            int stopbackward = option_find_int_quiet(options, "stopbackward", 0);
            if (stopbackward == 1) {
                last_stop_backward = count_tmp;
                printf("last_stop_backward = %d \n", last_stop_backward);
            }
            n_tmp = n_tmp->next;
            ++count_tmp;
        }
    }

    int old_params_train = params.train;

    fprintf(stderr, "   layer   filters  size/strd(dil)      input                output\n");
    while(n){

        params.train = old_params_train;
        if (count < last_stop_backward) params.train = 0;

        params.index = count;
        fprintf(stderr, "%4d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = { (LAYER_TYPE)0 };
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params);
        }
        else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }

        // calculate receptive field
        if(show_receptive_field)
        {
            int dilation = max_val_cmp(1, l.dilation);
            int stride = max_val_cmp(1, l.stride);
            int size = max_val_cmp(1, l.size);

            if (l.type == UPSAMPLE || (l.type == REORG))
            {

                l.receptive_w = receptive_w;
                l.receptive_h = receptive_h;
                l.receptive_w_scale = receptive_w_scale = receptive_w_scale / stride;
                l.receptive_h_scale = receptive_h_scale = receptive_h_scale / stride;

            }
            else {
                if (l.type == ROUTE) {
                    receptive_w = receptive_h = receptive_w_scale = receptive_h_scale = 0;
                    int k;
                    for (k = 0; k < l.n; ++k) {
                        layer route_l = net.layers[l.input_layers[k]];
                        receptive_w = max_val_cmp(receptive_w, route_l.receptive_w);
                        receptive_h = max_val_cmp(receptive_h, route_l.receptive_h);
                        receptive_w_scale = max_val_cmp(receptive_w_scale, route_l.receptive_w_scale);
                        receptive_h_scale = max_val_cmp(receptive_h_scale, route_l.receptive_h_scale);
                    }
                }
                else
                {
                    int increase_receptive = size + (dilation - 1) * 2 - 1;// stride;
                    increase_receptive = max_val_cmp(0, increase_receptive);

                    receptive_w += increase_receptive * receptive_w_scale;
                    receptive_h += increase_receptive * receptive_h_scale;
                    receptive_w_scale *= stride;
                    receptive_h_scale *= stride;
                }

                l.receptive_w = receptive_w;
                l.receptive_h = receptive_h;
                l.receptive_w_scale = receptive_w_scale;
                l.receptive_h_scale = receptive_h_scale;
            }
            //printf(" size = %d, dilation = %d, stride = %d, receptive_w = %d, receptive_w_scale = %d - ", size, dilation, stride, receptive_w, receptive_w_scale);

            int cur_receptive_w = receptive_w;
            int cur_receptive_h = receptive_h;

            fprintf(stderr, "%4d - receptive field: %d x %d \n", count, cur_receptive_w, cur_receptive_h);
        }

#ifdef GPU
        // futher GPU-memory optimization: net.optimized_memory == 2
        l.optimized_memory = net.optimized_memory;
        if (net.optimized_memory == 1 && params.train && l.type != DROPOUT) {
            if (l.delta_gpu) {
                cuda_free(l.delta_gpu);
                l.delta_gpu = NULL;
            }
        } else if (net.optimized_memory >= 2 && params.train && l.type != DROPOUT)
        {
            if (l.output_gpu) {
                cuda_free(l.output_gpu);
                //l.output_gpu = cuda_make_array_pinned(l.output, l.batch*l.outputs); // l.steps
                l.output_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
            }
            if (l.activation_input_gpu) {
                cuda_free(l.activation_input_gpu);
                l.activation_input_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
            }

            if (l.x_gpu) {
                cuda_free(l.x_gpu);
                l.x_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
            }

            // maximum optimization
            if (net.optimized_memory >= 3 && l.type != DROPOUT) {
                if (l.delta_gpu) {
                    cuda_free(l.delta_gpu);
                    //l.delta_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
                    //printf("\n\n PINNED DELTA GPU = %d \n", l.batch*l.outputs);
                }
            }

            if (l.type == CONVOLUTIONAL) {
                set_specified_workspace_limit(&l, net.workspace_size_limit);   // workspace size limit 1 GB
            }
        }
#endif // GPU

        l.clip = option_find_float_quiet(options, "clip", 0);
        l.dynamic_minibatch = net.dynamic_minibatch;
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.dont_update = option_find_int_quiet(options, "dont_update", 0);
        l.burnin_update = option_find_int_quiet(options, "burnin_update", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.train_only_bn = option_find_int_quiet(options, "train_only_bn", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        option_unused(options);

        if (l.stopbackward == 1) printf(" ------- previous layers are frozen ------- \n");

        net.layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if (l.inputs > max_inputs) max_inputs = l.inputs;
        if (l.outputs > max_outputs) max_outputs = l.outputs;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            if (l.antialiasing) {
                params.h = l.input_layer->out_h;
                params.w = l.input_layer->out_w;
                params.c = l.input_layer->out_c;
                params.inputs = l.input_layer->outputs;
            }
            else {
                params.h = l.out_h;
                params.w = l.out_w;
                params.c = l.out_c;
                params.inputs = l.outputs;
            }
        }
        if (l.bflops > 0) bflops += l.bflops;

        if (l.w > 1 && l.h > 1) {
            avg_outputs += l.outputs;
            avg_counter++;
        }
    }

    if (last_stop_backward > -1) {
        int k;
        for (k = 0; k < last_stop_backward; ++k) {
            layer l = net.layers[k];
            if (l.keep_delta_gpu) {
                if (!l.delta) {
                    net.layers[k].delta = (float*)xcalloc(l.outputs*l.batch, sizeof(float));
                }
#ifdef GPU
                if (!l.delta_gpu) {
                    net.layers[k].delta_gpu = (float *)cuda_make_array(NULL, l.outputs*l.batch);
                }
#endif
            }

            net.layers[k].onlyforward = 1;
            net.layers[k].train = 0;
        }
    }

    free_list(sections);

#ifdef GPU
    if (net.optimized_memory && params.train)
    {
        int k;
        for (k = 0; k < net.n; ++k) {
            layer l = net.layers[k];
            // delta GPU-memory optimization: net.optimized_memory == 1
            if (!l.keep_delta_gpu) {
                const size_t delta_size = l.outputs*l.batch; // l.steps
                if (net.max_delta_gpu_size < delta_size) {
                    net.max_delta_gpu_size = delta_size;
                    if (net.global_delta_gpu) cuda_free(net.global_delta_gpu);
                    if (net.state_delta_gpu) cuda_free(net.state_delta_gpu);
                    assert(net.max_delta_gpu_size > 0);
                    net.global_delta_gpu = (float *)cuda_make_array(NULL, net.max_delta_gpu_size);
                    net.state_delta_gpu = (float *)cuda_make_array(NULL, net.max_delta_gpu_size);
                }
                if (l.delta_gpu) {
                    if (net.optimized_memory >= 3) {}
                    else cuda_free(l.delta_gpu);
                }
                l.delta_gpu = net.global_delta_gpu;
            }
            else {
                if (!l.delta_gpu) l.delta_gpu = (float *)cuda_make_array(NULL, l.outputs*l.batch);
            }

            // maximum optimization
            if (net.optimized_memory >= 3 && l.type != DROPOUT) {
                if (l.delta_gpu && l.keep_delta_gpu) {
                    //cuda_free(l.delta_gpu);   // already called above
                    l.delta_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
                    //printf("\n\n PINNED DELTA GPU = %d \n", l.batch*l.outputs);
                }
            }

            net.layers[k] = l;
        }
    }
#endif

    set_train_only_bn(net); // set l.train_only_bn for all required layers

    net.outputs = get_network_output_size(net);
    net.output = get_network_output(net);
    avg_outputs = avg_outputs / avg_counter;
    fprintf(stderr, "Total BFLOPS %5.3f \n", bflops);
    fprintf(stderr, "avg_outputs = %d \n", avg_outputs);
#ifdef GPU
    get_cuda_stream();
    //get_cuda_memcpy_stream();
    if (gpu_index >= 0)
    {
        int size = get_network_input_size(net) * net.batch;
        net.input_state_gpu = cuda_make_array(0, size);
        if (cudaSuccess == cudaHostAlloc(&net.input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped)) net.input_pinned_cpu_flag = 1;
        else {
            cudaGetLastError(); // reset CUDA-error
            net.input_pinned_cpu = (float*)xcalloc(size, sizeof(float));
        }

        // pre-allocate memory for inference on Tensor Cores (fp16)
        *net.max_input16_size = 0;
        *net.max_output16_size = 0;
        if (net.cudnn_half) {
            *net.max_input16_size = max_inputs;
            CHECK_CUDA(cudaMalloc((void **)net.input16_gpu, *net.max_input16_size * sizeof(short))); //sizeof(half)
            *net.max_output16_size = max_outputs;
            CHECK_CUDA(cudaMalloc((void **)net.output16_gpu, *net.max_output16_size * sizeof(short))); //sizeof(half)
        }
        if (workspace_size) {
            fprintf(stderr, " Allocate additional workspace_size = %1.2f MB \n", (float)workspace_size/1000000);
            net.workspace = cuda_make_array(0, workspace_size / sizeof(float) + 1);
        }
        else {
            net.workspace = (float*)xcalloc(1, workspace_size);
        }
    }
#else
        if (workspace_size) {
            net.workspace = (float*)xcalloc(1, workspace_size);
        }
#endif

    LAYER_TYPE lt = net.layers[net.n - 1].type;
    if ((net.w % 32 != 0 || net.h % 32 != 0) && (lt == YOLO || lt == REGION || lt == DETECTION)) {
        printf("\n Warning: width=%d and height=%d in cfg-file must be divisible by 32 for default networks Yolo v1/v2/v3!!! \n\n",
            net.w, net.h);
    }
    return net;
}



list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *sections = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = (section*)xmalloc(sizeof(section));
                list_insert(sections, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return sections;
}

void save_convolutional_weights_binary(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    int size = (l.c/l.groups)*l.size*l.size;
    binarize_weights(l.weights, l.n, size, l.binary_weights);
    int i, j, k;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    for(i = 0; i < l.n; ++i){
        float mean = l.binary_weights[i*size];
        if(mean < 0) mean = -mean;
        fwrite(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                if (l.binary_weights[index + k] > 0) c = (c | 1<<k);
            }
            fwrite(&c, sizeof(char), 1, fp);
        }
    }
}

void save_shortcut_weights(layer l, FILE *fp)
{
#ifdef GPU
    if (gpu_index >= 0) {
        pull_shortcut_layer(l);
        printf("\n pull_shortcut_layer \n");
    }
#endif
    int i;
    //if(l.weight_updates) for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weight_updates[i]);
    //printf(" l.nweights = %d - update \n", l.nweights);
    for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    printf(" l.nweights = %d \n\n", l.nweights);

    int num = l.nweights;
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_implicit_weights(layer l, FILE *fp)
{
#ifdef GPU
    if (gpu_index >= 0) {
        pull_implicit_layer(l);
        //printf("\n pull_implicit_layer \n");
    }
#endif
    int i;
    //if(l.weight_updates) for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weight_updates[i]);
    //printf(" l.nweights = %d - update \n", l.nweights);
    //for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    //printf(" l.nweights = %d \n\n", l.nweights);

    int num = l.nweights;
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    int num = l.nweights;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
    //if(l.adam){
    //    fwrite(l.m, sizeof(float), num, fp);
    //    fwrite(l.v, sizeof(float), num, fp);
    //}
}

void save_convolutional_weights_ema(layer l, FILE *fp)
{
    if (l.binary) {
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if (gpu_index >= 0) {
        pull_convolutional_layer(l);
    }
#endif
    int num = l.nweights;
    fwrite(l.biases_ema, sizeof(float), l.n, fp);
    if (l.batch_normalize) {
        fwrite(l.scales_ema, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights_ema, sizeof(float), num, fp);
    //if(l.adam){
    //    fwrite(l.m, sizeof(float), num, fp);
    //    fwrite(l.v, sizeof(float), num, fp);
    //}
}

void save_batchnorm_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_batchnorm_layer(l);
    }
#endif
    fwrite(l.biases, sizeof(float), l.c, fp);
    fwrite(l.scales, sizeof(float), l.c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_connected_layer(l);
    }
#endif
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}

void save_weights_upto(network net, char *filename, int cutoff, int save_ema)
{
#ifdef GPU
    if(net.gpu_index >= 0){
        cuda_set_device(net.gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int32_t major = MAJOR_VERSION;
    int32_t minor = MINOR_VERSION;
    int32_t revision = PATCH_VERSION;
    fwrite(&major, sizeof(int32_t), 1, fp);
    fwrite(&minor, sizeof(int32_t), 1, fp);
    fwrite(&revision, sizeof(int32_t), 1, fp);
    (*net.seen) = get_current_iteration(net) * net.batch * net.subdivisions; // remove this line, when you will save to weights-file both: seen & cur_iteration
    fwrite(net.seen, sizeof(uint64_t), 1, fp);

    int i;
    for(i = 0; i < net.n && i < cutoff; ++i){
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL && l.share_layer == NULL) {
            if (save_ema) {
                save_convolutional_weights_ema(l, fp);
            }
            else {
                save_convolutional_weights(l, fp);
            }
        } if (l.type == SHORTCUT && l.nweights > 0) {
            save_shortcut_weights(l, fp);
        } if (l.type == IMPLICIT) {
            save_implicit_weights(l, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        } if(l.type == BATCHNORM){
            save_batchnorm_weights(l, fp);
        } if(l.type == RNN){
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
        } if(l.type == GRU){
            save_connected_weights(*(l.input_z_layer), fp);
            save_connected_weights(*(l.input_r_layer), fp);
            save_connected_weights(*(l.input_h_layer), fp);
            save_connected_weights(*(l.state_z_layer), fp);
            save_connected_weights(*(l.state_r_layer), fp);
            save_connected_weights(*(l.state_h_layer), fp);
        } if(l.type == LSTM){
            save_connected_weights(*(l.wf), fp);
            save_connected_weights(*(l.wi), fp);
            save_connected_weights(*(l.wg), fp);
            save_connected_weights(*(l.wo), fp);
            save_connected_weights(*(l.uf), fp);
            save_connected_weights(*(l.ui), fp);
            save_connected_weights(*(l.ug), fp);
            save_connected_weights(*(l.uo), fp);
        } if (l.type == CONV_LSTM) {
            if (l.peephole) {
                save_convolutional_weights(*(l.vf), fp);
                save_convolutional_weights(*(l.vi), fp);
                save_convolutional_weights(*(l.vo), fp);
            }
            save_convolutional_weights(*(l.wf), fp);
            if (!l.bottleneck) {
                save_convolutional_weights(*(l.wi), fp);
                save_convolutional_weights(*(l.wg), fp);
                save_convolutional_weights(*(l.wo), fp);
            }
            save_convolutional_weights(*(l.uf), fp);
            save_convolutional_weights(*(l.ui), fp);
            save_convolutional_weights(*(l.ug), fp);
            save_convolutional_weights(*(l.uo), fp);
        } if(l.type == CRNN){
            save_convolutional_weights(*(l.input_layer), fp);
            save_convolutional_weights(*(l.self_layer), fp);
            save_convolutional_weights(*(l.output_layer), fp);
        } if(l.type == LOCAL){
#ifdef GPU
            if(gpu_index >= 0){
                pull_local_layer(l);
            }
#endif
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.weights, sizeof(float), size, fp);
        }
        fflush(fp);
    }
    fclose(fp);
}
void save_weights(network net, char *filename)
{
    save_weights_upto(net, filename, net.n, 0);
}

void transpose_matrix(float *a, int rows, int cols)
{
    float* transpose = (float*)xcalloc(rows * cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if(transpose){
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_connected_layer(l);
    }
#endif
}

void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.c, fp);
    fread(l.scales, sizeof(float), l.c, fp);
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    fread(l.rolling_variance, sizeof(float), l.c, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_batchnorm_layer(l);
    }
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    int size = (l.c / l.groups)*l.size*l.size;
    int i, j, k;
    for(i = 0; i < l.n; ++i){
        float mean = 0;
        fread(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            fread(&c, sizeof(char), 1, fp);
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                l.weights[index + k] = (c & 1<<k) ? mean : -mean;
            }
        }
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    int num = l.nweights;
    int read_bytes;
    read_bytes = fread(l.biases, sizeof(float), l.n, fp);
    if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.biases - l.index = %d \n", l.index);
    //fread(l.weights, sizeof(float), num, fp); // as in connected layer
    if (l.batch_normalize && (!l.dontloadscales)){
        read_bytes = fread(l.scales, sizeof(float), l.n, fp);
        if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.scales - l.index = %d \n", l.index);
        read_bytes = fread(l.rolling_mean, sizeof(float), l.n, fp);
        if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.rolling_mean - l.index = %d \n", l.index);
        read_bytes = fread(l.rolling_variance, sizeof(float), l.n, fp);
        if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.rolling_variance - l.index = %d \n", l.index);
        if(0){
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
        if(0){
            fill_cpu(l.n, 0, l.rolling_mean, 1);
            fill_cpu(l.n, 0, l.rolling_variance, 1);
        }
    }
    read_bytes = fread(l.weights, sizeof(float), num, fp);
    if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", l.index);
    //if(l.adam){
    //    fread(l.m, sizeof(float), num, fp);
    //    fread(l.v, sizeof(float), num, fp);
    //}
    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l.flipped) {
        transpose_matrix(l.weights, (l.c/l.groups)*l.size*l.size, l.n);
    }
    //if (l.binary) binarize_weights(l.weights, l.n, (l.c/l.groups)*l.size*l.size, l.weights);
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void load_shortcut_weights(layer l, FILE *fp)
{
    int num = l.nweights;
    int read_bytes;
    read_bytes = fread(l.weights, sizeof(float), num, fp);
    if (read_bytes > 0 && read_bytes < num) printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", l.index);
    //for (int i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    //printf(" read_bytes = %d \n\n", read_bytes);
#ifdef GPU
    if (gpu_index >= 0) {
        push_shortcut_layer(l);
    }
#endif
}

void load_implicit_weights(layer l, FILE *fp)
{
    int num = l.nweights;
    int read_bytes;
    read_bytes = fread(l.weights, sizeof(float), num, fp);
    if (read_bytes > 0 && read_bytes < num) printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", l.index);
    //for (int i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    //printf(" read_bytes = %d \n\n", read_bytes);
#ifdef GPU
    if (gpu_index >= 0) {
        push_implicit_layer(l);
    }
#endif
}

void load_weights_upto(network *net, char *filename, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int32_t major;
    int32_t minor;
    int32_t revision;
    fread(&major, sizeof(int32_t), 1, fp);
    fread(&minor, sizeof(int32_t), 1, fp);
    fread(&revision, sizeof(int32_t), 1, fp);
    if ((major * 10 + minor) >= 2) {
        printf("\n seen 64");
        uint64_t iseen = 0;
        fread(&iseen, sizeof(uint64_t), 1, fp);
        *net->seen = iseen;
    }
    else {
        printf("\n seen 32");
        uint32_t iseen = 0;
        fread(&iseen, sizeof(uint32_t), 1, fp);
        *net->seen = iseen;
    }
    *net->cur_iteration = get_current_batch(*net);
    printf(", trained: %.0f K-images (%.0f Kilo-batches_64) \n", (float)(*net->seen / 1000), (float)(*net->seen / 64000));
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL && l.share_layer == NULL){
            load_convolutional_weights(l, fp);
        }
        if (l.type == SHORTCUT && l.nweights > 0) {
            load_shortcut_weights(l, fp);
        }
        if (l.type == IMPLICIT) {
            load_implicit_weights(l, fp);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
        if(l.type == BATCHNORM){
            load_batchnorm_weights(l, fp);
        }
        if(l.type == CRNN){
            load_convolutional_weights(*(l.input_layer), fp);
            load_convolutional_weights(*(l.self_layer), fp);
            load_convolutional_weights(*(l.output_layer), fp);
        }
        if(l.type == RNN){
            load_connected_weights(*(l.input_layer), fp, transpose);
            load_connected_weights(*(l.self_layer), fp, transpose);
            load_connected_weights(*(l.output_layer), fp, transpose);
        }
        if(l.type == GRU){
            load_connected_weights(*(l.input_z_layer), fp, transpose);
            load_connected_weights(*(l.input_r_layer), fp, transpose);
            load_connected_weights(*(l.input_h_layer), fp, transpose);
            load_connected_weights(*(l.state_z_layer), fp, transpose);
            load_connected_weights(*(l.state_r_layer), fp, transpose);
            load_connected_weights(*(l.state_h_layer), fp, transpose);
        }
        if(l.type == LSTM){
            load_connected_weights(*(l.wf), fp, transpose);
            load_connected_weights(*(l.wi), fp, transpose);
            load_connected_weights(*(l.wg), fp, transpose);
            load_connected_weights(*(l.wo), fp, transpose);
            load_connected_weights(*(l.uf), fp, transpose);
            load_connected_weights(*(l.ui), fp, transpose);
            load_connected_weights(*(l.ug), fp, transpose);
            load_connected_weights(*(l.uo), fp, transpose);
        }
        if (l.type == CONV_LSTM) {
            if (l.peephole) {
                load_convolutional_weights(*(l.vf), fp);
                load_convolutional_weights(*(l.vi), fp);
                load_convolutional_weights(*(l.vo), fp);
            }
            load_convolutional_weights(*(l.wf), fp);
            if (!l.bottleneck) {
                load_convolutional_weights(*(l.wi), fp);
                load_convolutional_weights(*(l.wg), fp);
                load_convolutional_weights(*(l.wo), fp);
            }
            load_convolutional_weights(*(l.uf), fp);
            load_convolutional_weights(*(l.ui), fp);
            load_convolutional_weights(*(l.ug), fp);
            load_convolutional_weights(*(l.uo), fp);
        }
        if(l.type == LOCAL){
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.weights, sizeof(float), size, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_local_layer(l);
            }
#endif
        }
        if (feof(fp)) break;
    }
    fprintf(stderr, "Done! Loaded %d layers from weights-file \n", i);
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, net->n);
}

// load network & force - set batch size
network *load_network_custom(char *cfg, char *weights, int clear, int batch)
{
    printf(" Try to load cfg: %s, weights: %s, clear = %d \n", cfg, weights, clear);
    network* net = (network*)xcalloc(1, sizeof(network));
    *net = parse_network_cfg_custom(cfg, batch, 1);
    if (weights && weights[0] != 0) {
        printf(" Try to load weights: %s \n", weights);
        load_weights(net, weights);
    }
    fuse_conv_batchnorm(*net);
    if (clear) {
        (*net->seen) = 0;
        (*net->cur_iteration) = 0;
    }
    return net;
}

// load network & get batch size from cfg-file
network *load_network(char *cfg, char *weights, int clear)
{
    printf(" Try to load cfg: %s, clear = %d \n", cfg, clear);
    network* net = (network*)xcalloc(1, sizeof(network));
    *net = parse_network_cfg(cfg);
    if (weights && weights[0] != 0) {
        printf(" Try to load weights: %s \n", weights);
        load_weights(net, weights);
    }
    if (clear) {
        (*net->seen) = 0;
        (*net->cur_iteration) = 0;
    }
    return net;
}
