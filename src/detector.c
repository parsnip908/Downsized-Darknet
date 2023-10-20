#include <stdlib.h>
#include "darknet.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int (*__compar_fn_t)(const void*, const void*);
#ifdef __USE_GNU
typedef __compar_fn_t comparison_fn_t;
#endif
#endif

#include "http_stream.h"

static int coco_ids[] = { 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90 };

static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if (c) p = c;
    return atoi(p + 1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    //int image_id = get_coco_image_id(image_path);
    char *p = basecfg(image_path);
    int image_id = atoi(p);
    for (i = 0; i < num_boxes; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j] > 0) {
                char buff[1024];
                sprintf(buff, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
                fprintf(fp, "%s", buff);
                //printf("%s", buff);
            }
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j) {
            int myclass = j;
            if (dets[i].prob[myclass] > 0) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1, dets[i].prob[myclass],
                xmin, ymin, xmax, ymax);
        }
    }
}

static void print_kitti_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h, char *outfile, char *prefix)
{
    char *kitti_ids[] = { "car", "pedestrian", "cyclist" };
    FILE *fpd = 0;
    char buffd[1024];
    snprintf(buffd, 1024, "%s/%s/data/%s.txt", prefix, outfile, id);

    fpd = fopen(buffd, "w");
    int i, j;
    for (i = 0; i < total; ++i)
    {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j)
        {
            //if (dets[i].prob[j]) fprintf(fpd, "%s 0 0 0 %f %f %f %f -1 -1 -1 -1 0 0 0 %f\n", kitti_ids[j], xmin, ymin, xmax, ymax, dets[i].prob[j]);
            if (dets[i].prob[j]) fprintf(fpd, "%s -1 -1 -10 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 %f\n", kitti_ids[j], xmin, ymin, xmax, ymax, dets[i].prob[j]);
        }
    }
    fclose(fpd);
}

static void eliminate_bdd(char *buf, char *a)
{
    int n = 0;
    int i, k;
    for (i = 0; buf[i] != '\0'; i++)
    {
        if (buf[i] == a[n])
        {
            k = i;
            while (buf[i] == a[n])
            {
                if (a[++n] == '\0')
                {
                    for (k; buf[k + n] != '\0'; k++)
                    {
                        buf[k] = buf[k + n];
                    }
                    buf[k] = '\0';
                    break;
                }
                i++;
            }
            n = 0; i--;
        }
    }
}

static void get_bdd_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    eliminate_bdd(p, ".jpg");
    eliminate_bdd(p, "/");
    strcpy(filename, p);
}

static void print_bdd_detections(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    char *bdd_ids[] = { "bike" , "bus" , "car" , "motor" ,"person", "rider", "traffic light", "traffic sign", "train", "truck" };
    get_bdd_image_id(image_path);
    int i, j;

    for (i = 0; i < num_boxes; ++i)
    {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx1 = xmin;
        float by1 = ymin;
        float bx2 = xmax;
        float by2 = ymax;

        for (j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j])
            {
                fprintf(fp, "\t{\n\t\t\"name\":\"%s\",\n\t\t\"category\":\"%s\",\n\t\t\"bbox\":[%f, %f, %f, %f],\n\t\t\"score\":%f\n\t},\n", image_path, bdd_ids[j], bx1, by1, bx2, by2, dets[i].prob[j]);
            }
        }
    }
}

typedef struct {
    box b;
    float p;
    int class_id;
    int image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

int detections_comparator(const void *pa, const void *pb)
{
    box_prob a = *(const box_prob *)pa;
    box_prob b = *(const box_prob *)pb;
    float diff = a.p - b.p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

typedef struct {
    float w, h;
} anchors_t;

int anchors_comparator(const void *pa, const void *pb)
{
    anchors_t a = *(const anchors_t *)pa;
    anchors_t b = *(const anchors_t *)pb;
    float diff = b.w*b.h - a.w*a.h;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

int anchors_data_comparator(const float **pa, const float **pb)
{
    float *a = (float *)*pa;
    float *b = (float *)*pb;
    float diff = b[0] * b[1] - a[0] * a[1];
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}


void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show)
{
    printf("\n num_of_clusters = %d, width = %d, height = %d \n", num_of_clusters, width, height);
    if (width < 0 || height < 0) {
        printf("Usage: darknet detector calc_anchors data/voc.data -num_of_clusters 9 -width 416 -height 416 \n");
        printf("Error: set width and height \n");
        return;
    }

    //float pointsdata[] = { 1,1, 2,2, 6,6, 5,5, 10,10 };
    float* rel_width_height_array = (float*)xcalloc(1000, sizeof(float));


    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    list *plist = get_paths(train_images);
    int number_of_images = plist->size;
    char **paths = (char **)list_to_array(plist);

    int classes = option_find_int(options, "classes", 1);
    int* counter_per_class = (int*)xcalloc(classes, sizeof(int));

    srand(time(0));
    int number_of_boxes = 0;
    printf(" read labels from %d images \n", number_of_images);

    int i, j;
    for (i = 0; i < number_of_images; ++i) {
        char *path = paths[i];
        char labelpath[4096];
        replace_image_to_label(path, labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        //printf(" new path: %s \n", labelpath);
        char *buff = (char*)xcalloc(6144, sizeof(char));
        for (j = 0; j < num_labels; ++j)
        {
            if (truth[j].x > 1 || truth[j].x <= 0 || truth[j].y > 1 || truth[j].y <= 0 ||
                truth[j].w > 1 || truth[j].w <= 0 || truth[j].h > 1 || truth[j].h <= 0)
            {
                printf("\n\nWrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f \n",
                    labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
                sprintf(buff, "echo \"Wrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f\" >> bad_label.list",
                    labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
                system(buff);
            }
            if (truth[j].id >= classes) {
                classes = truth[j].id + 1;
                counter_per_class = (int*)xrealloc(counter_per_class, classes * sizeof(int));
            }
            counter_per_class[truth[j].id]++;

            number_of_boxes++;
            rel_width_height_array = (float*)xrealloc(rel_width_height_array, 2 * number_of_boxes * sizeof(float));

            rel_width_height_array[number_of_boxes * 2 - 2] = truth[j].w * width;
            rel_width_height_array[number_of_boxes * 2 - 1] = truth[j].h * height;
            printf("\r loaded \t image: %d \t box: %d", i + 1, number_of_boxes);
        }
        free(buff);
        free(truth);
    }
    printf("\n all loaded. \n");
    printf("\n calculating k-means++ ...");

    matrix boxes_data;
    model anchors_data;
    boxes_data = make_matrix(number_of_boxes, 2);

    printf("\n");
    for (i = 0; i < number_of_boxes; ++i) {
        boxes_data.vals[i][0] = rel_width_height_array[i * 2];
        boxes_data.vals[i][1] = rel_width_height_array[i * 2 + 1];
        //if (w > 410 || h > 410) printf("i:%d,  w = %f, h = %f \n", i, w, h);
    }

    // Is used: distance(box, centroid) = 1 - IoU(box, centroid)

    // K-means
    anchors_data = do_kmeans(boxes_data, num_of_clusters);

    qsort((void*)anchors_data.centers.vals, num_of_clusters, 2 * sizeof(float), (__compar_fn_t)anchors_data_comparator);

    //gen_anchors.py = 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66
    //float orig_anch[] = { 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66 };

    printf("\n");
    float avg_iou = 0;
    for (i = 0; i < number_of_boxes; ++i) {
        float box_w = rel_width_height_array[i * 2]; //points->data.fl[i * 2];
        float box_h = rel_width_height_array[i * 2 + 1]; //points->data.fl[i * 2 + 1];
                                                         //int cluster_idx = labels->data.i[i];
        int cluster_idx = 0;
        float min_dist = FLT_MAX;
        float best_iou = 0;
        for (j = 0; j < num_of_clusters; ++j) {
            float anchor_w = anchors_data.centers.vals[j][0];   // centers->data.fl[j * 2];
            float anchor_h = anchors_data.centers.vals[j][1];   // centers->data.fl[j * 2 + 1];
            float min_w = (box_w < anchor_w) ? box_w : anchor_w;
            float min_h = (box_h < anchor_h) ? box_h : anchor_h;
            float box_intersect = min_w*min_h;
            float box_union = box_w*box_h + anchor_w*anchor_h - box_intersect;
            float iou = box_intersect / box_union;
            float distance = 1 - iou;
            if (distance < min_dist) {
              min_dist = distance;
              cluster_idx = j;
              best_iou = iou;
            }
        }

        float anchor_w = anchors_data.centers.vals[cluster_idx][0]; //centers->data.fl[cluster_idx * 2];
        float anchor_h = anchors_data.centers.vals[cluster_idx][1]; //centers->data.fl[cluster_idx * 2 + 1];
        if (best_iou > 1 || best_iou < 0) { // || box_w > width || box_h > height) {
            printf(" Wrong label: i = %d, box_w = %f, box_h = %f, anchor_w = %f, anchor_h = %f, iou = %f \n",
                i, box_w, box_h, anchor_w, anchor_h, best_iou);
        }
        else avg_iou += best_iou;
    }

    char buff[1024];
    FILE* fwc = fopen("counters_per_class.txt", "wb");
    if (fwc) {
        sprintf(buff, "counters_per_class = ");
        printf("\n%s", buff);
        fwrite(buff, sizeof(char), strlen(buff), fwc);
        for (i = 0; i < classes; ++i) {
            sprintf(buff, "%d", counter_per_class[i]);
            printf("%s", buff);
            fwrite(buff, sizeof(char), strlen(buff), fwc);
            if (i < classes - 1) {
                fwrite(", ", sizeof(char), 2, fwc);
                printf(", ");
            }
        }
        printf("\n");
        fclose(fwc);
    }
    else {
        printf(" Error: file counters_per_class.txt can't be open \n");
    }

    avg_iou = 100 * avg_iou / number_of_boxes;
    printf("\n avg IoU = %2.2f %% \n", avg_iou);


    FILE* fw = fopen("anchors.txt", "wb");
    if (fw) {
        printf("\nSaving anchors to the file: anchors.txt \n");
        printf("anchors = ");
        for (i = 0; i < num_of_clusters; ++i) {
            float anchor_w = anchors_data.centers.vals[i][0]; //centers->data.fl[i * 2];
            float anchor_h = anchors_data.centers.vals[i][1]; //centers->data.fl[i * 2 + 1];
            if (width > 32) sprintf(buff, "%3.0f,%3.0f", anchor_w, anchor_h);
            else sprintf(buff, "%2.4f,%2.4f", anchor_w, anchor_h);
            printf("%s", buff);
            fwrite(buff, sizeof(char), strlen(buff), fw);
            if (i + 1 < num_of_clusters) {
                fwrite(", ", sizeof(char), 2, fw);
                printf(", ");
            }
        }
        printf("\n");
        fclose(fw);
    }
    else {
        printf(" Error: file anchors.txt can't be open \n");
    }

    if (show) {
#ifdef OPENCV
        show_acnhors(number_of_boxes, num_of_clusters, rel_width_height_array, anchors_data, width, height);
#endif // OPENCV
    }
    free(rel_width_height_array);
    free(counter_per_class);
}


void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    if (net.letter_box) letter_box = 1;
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
    }
    srand(2222222);
    char buff[256];
    char *input = buff;
    char *json_buf = NULL;
    int json_image_id = 0;
    FILE* json_file = NULL;
    if (outfile) {
        json_file = fopen(outfile, "wb");
        if(!json_file) {
            error("fopen failed", DARKNET_LOC);
        }
        char *tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    int j;
    float nms = .45;    // 0.4F
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;
        if(letter_box) sized = letterbox_image(im, net.w, net.h);
        else sized = resize_image(im, net.w, net.h);

        layer l = net.layers[net.n - 1];
        int k;
        for (k = 0; k < net.n; ++k) {
            layer lk = net.layers[k];
            if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
                l = lk;
                printf(" Detection layer: %d - type = %d \n", k, l.type);
            }
        }

        //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        //float **probs = calloc(l.w*l.h*l.n, sizeof(float*));
        //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)xcalloc(l.classes, sizeof(float));

        float *X = sized.data;

        //time= what_time_is_it_now();
        double time = get_time_point();
        network_predict(net, X);
        //network_predict_image(&net, im); letterbox = 1;
        printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);
        //printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));

        int nboxes = 0;
        detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }
        draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);
        save_image(im, "predictions");
        if (!dont_show) {
            show_image(im, "predictions");
        }

        if (json_file) {
            if (json_buf) {
                char *tmp = ", \n";
                fwrite(tmp, sizeof(char), strlen(tmp), json_file);
            }
            ++json_image_id;
            json_buf = detection_to_json(dets, nboxes, l.classes, names, json_image_id, input);

            fwrite(json_buf, sizeof(char), strlen(json_buf), json_file);
            free(json_buf);
        }

        // pseudo labeling concept - fast.ai
        if (save_labels)
        {
            char labelpath[4096];
            replace_image_to_label(input, labelpath);

            FILE* fw = fopen(labelpath, "wb");
            int i;
            for (i = 0; i < nboxes; ++i) {
                char buff[1024];
                int class_id = -1;
                float prob = 0;
                for (j = 0; j < l.classes; ++j) {
                    if (dets[i].prob[j] > thresh && dets[i].prob[j] > prob) {
                        prob = dets[i].prob[j];
                        class_id = j;
                    }
                }
                if (class_id >= 0) {
                    sprintf(buff, "%d %2.4f %2.4f %2.4f %2.4f\n", class_id, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
                    fwrite(buff, sizeof(char), strlen(buff), fw);
                }
            }
            fclose(fw);
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        if (filename) break;
    }

    if (json_file) {
        char *tmp = "\n]";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
        fclose(json_file);
    }

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);
    free_alphabet(alphabet);
    free_network(net);
}

#if defined(OPENCV) && defined(GPU)

// adversarial attack dnn
void draw_object(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, int dont_show, int it_num,
    int letter_box, int benchmark_layers)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);// parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    net.adversarial = 1;
    set_batch_network(&net, 1);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    //fuse_conv_batchnorm(net);
    //calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
    }

    srand(2222222);
    char buff[256];
    char *input = buff;

    int j;
    float nms = .45;    // 0.4F
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;
        if (letter_box) sized = letterbox_image(im, net.w, net.h);
        else sized = resize_image(im, net.w, net.h);

        image src_sized = copy_image(sized);

        layer l = net.layers[net.n - 1];
        int k;
        for (k = 0; k < net.n; ++k) {
            layer lk = net.layers[k];
            if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
                l = lk;
                printf(" Detection layer: %d - type = %d \n", k, l.type);
            }
        }

        net.num_boxes = l.max_boxes;
        int num_truth = l.truths;
        float *truth_cpu = (float *)xcalloc(num_truth, sizeof(float));

        int *it_num_set = (int *)xcalloc(1, sizeof(int));
        float *lr_set = (float *)xcalloc(1, sizeof(float));
        int *boxonly = (int *)xcalloc(1, sizeof(int));

        cv_draw_object(sized, truth_cpu, net.num_boxes, num_truth, it_num_set, lr_set, boxonly, l.classes, names);

        net.learning_rate = *lr_set;
        it_num = *it_num_set;

        float *X = sized.data;

        mat_cv* img = NULL;
        float max_img_loss = 5;
        int number_of_lines = 100;
        int img_size = 1000;
        char windows_name[100];
        char *base = basecfg(cfgfile);
        sprintf(windows_name, "chart_%s.png", base);
        img = draw_train_chart(windows_name, max_img_loss, it_num, number_of_lines, img_size, dont_show, NULL);

        int iteration;
        for (iteration = 0; iteration < it_num; ++iteration)
        {
            forward_backward_network_gpu(net, X, truth_cpu);

            float avg_loss = get_network_cost(net);
            draw_train_loss(windows_name, img, img_size, avg_loss, max_img_loss, iteration, it_num, 0, 0, "mAP%", 0, dont_show, 0, 0);

            float inv_loss = 1.0 / max_val_cmp(0.01, avg_loss);
            //net.learning_rate = *lr_set * inv_loss;

            if (*boxonly) {
                int dw = truth_cpu[2] * sized.w, dh = truth_cpu[3] * sized.h;
                int dx = truth_cpu[0] * sized.w - dw / 2, dy = truth_cpu[1] * sized.h - dh / 2;
                image crop = crop_image(sized, dx, dy, dw, dh);
                copy_image_inplace(src_sized, sized);
                embed_image(crop, sized, dx, dy);
            }

            show_image_cv(sized, "image_optimization");
            wait_key_cv(20);
        }

        net.train = 0;
        quantize_image(sized);
        network_predict(net, X);

        save_image_png(sized, "drawn");
        //sized = load_image("drawn.png", 0, 0, net.c);

        int nboxes = 0;
        detection *dets = get_network_boxes(&net, sized.w, sized.h, thresh, 0, 0, 1, &nboxes, letter_box);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }
        draw_detections_v3(sized, dets, nboxes, thresh, names, alphabet, l.classes, 1);
        save_image(sized, "pre_predictions");
        if (!dont_show) {
            show_image(sized, "pre_predictions");
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
        free_image(src_sized);

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        free(lr_set);
        free(it_num_set);

        if (filename) break;
    }

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
}
#else // defined(OPENCV) && defined(GPU)
void draw_object(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, int dont_show, int it_num,
    int letter_box, int benchmark_layers)
{
    error("darknet detector draw ... can't be used without OpenCV and CUDA", DARKNET_LOC);
}
#endif // defined(OPENCV) && defined(GPU)

void run_detector(int argc, char **argv)
{
    int dont_show = find_arg(argc, argv, "-dont_show");
    int benchmark = find_arg(argc, argv, "-benchmark");
    int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
    //if (benchmark_layers) benchmark = 1;
    if (benchmark) dont_show = 1;
    int show = find_arg(argc, argv, "-show");
    int letter_box = find_arg(argc, argv, "-letter_box");
    int calc_map = find_arg(argc, argv, "-map");
    int map_points = find_int_arg(argc, argv, "-points", 0);
    int show_imgs = find_arg(argc, argv, "-show_imgs");
    int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
    int avgframes = find_int_arg(argc, argv, "-avgframes", 3);
    int dontdraw_bbox = find_arg(argc, argv, "-dontdraw_bbox");
    int json_port = find_int_arg(argc, argv, "-json_port", -1);
    char *http_post_host = find_char_arg(argc, argv, "-http_post_host", 0);
    int time_limit_sec = find_int_arg(argc, argv, "-time_limit_sec", 0);
    char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    char *json_file_output = find_char_arg(argc, argv, "-json_file_output", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);    // 0.24
    float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
    int width = find_int_arg(argc, argv, "-width", -1);
    int height = find_int_arg(argc, argv, "-height", -1);
    // extended output in test mode (output of rect bound coords)
    // and for recall mode (extended output table-like format with results for best_class fit)
    int ext_output = find_arg(argc, argv, "-ext_output");
    int save_labels = find_arg(argc, argv, "-save_labels");
    char* chart_path = find_char_arg(argc, argv, "-chart", 0);
    // While training, decide after how many epochs mAP will be calculated. Default value is 4 which means the mAP will be calculated after each 4 epochs
    int mAP_epochs = find_int_arg(argc, argv, "-mAP_epochs", 4);
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [train/test/valid/demo/map] [data] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if (gpu_list) {
        printf("%s\n", gpu_list);
        int len = (int)strlen(gpu_list);
        ngpus = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int*)xcalloc(ngpus, sizeof(int));
        for (i = 0; i < ngpus; ++i) {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',') + 1;
        }
    }
    else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    if (weights)
        if (strlen(weights) > 0)
            if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = (argc > 6) ? argv[6] : 0;
    if (0 == strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers);
    // else if (0 == strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear, dont_show, calc_map, thresh, iou_thresh, mjpeg_port, show_imgs, benchmark_layers, chart_path, mAP_epochs);
    // else if (0 == strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    // else if (0 == strcmp(argv[2], "recall")) validate_detector_recall(datacfg, cfg, weights);
    // else if (0 == strcmp(argv[2], "map")) validate_detector_map(datacfg, cfg, weights, thresh, iou_thresh, map_points, letter_box, NULL);
    else if (0 == strcmp(argv[2], "calc_anchors")) calc_anchors(datacfg, num_of_clusters, width, height, show);
    else if (0 == strcmp(argv[2], "draw")) {
        int it_num = 100;
        draw_object(datacfg, cfg, weights, filename, thresh, dont_show, it_num, letter_box, benchmark_layers);
    }
    else if (0 == strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        if (filename)
            if (strlen(filename) > 0)
                if (filename[strlen(filename) - 1] == 0x0d) filename[strlen(filename) - 1] = 0;
        demo(cfg, weights, thresh, hier_thresh, cam_index, filename, names, classes, avgframes, frame_skip, prefix, out_filename,
            mjpeg_port, dontdraw_bbox, json_port, dont_show, ext_output, letter_box, time_limit_sec, http_post_host, benchmark, benchmark_layers, json_file_output);

        free_list_contents_kvp(options);
        free_list(options);
    }
    else printf(" There isn't such command: %s", argv[2]);

    if (gpus && gpu_list && ngpus > 1) free(gpus);
}
