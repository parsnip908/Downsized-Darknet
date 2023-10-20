#include "data.h"
#include "utils.h"
#include "image.h"
#include "box.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUMCHARS 37

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

/*
char **get_random_paths_indexes(char **paths, int n, int m, int *indexes)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        int index = random_gen()%m;
        indexes[i] = index;
        random_paths[i] = paths[index];
        if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}
*/

char **get_sequential_paths(char **paths, int n, int m, int mini_batch, int augment_speed, int contrastive)
{
    int speed = rand_int(1, augment_speed);
    if (speed < 1) speed = 1;
    char** sequentia_paths = (char**)xcalloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    //printf("n = %d, mini_batch = %d \n", n, mini_batch);
    unsigned int *start_time_indexes = (unsigned int *)xcalloc(mini_batch, sizeof(unsigned int));
    for (i = 0; i < mini_batch; ++i) {
        if (contrastive && (i % 2) == 1) start_time_indexes[i] = start_time_indexes[i - 1];
        else start_time_indexes[i] = random_gen() % m;

        //printf(" start_time_indexes[i] = %u, ", start_time_indexes[i]);
    }

    for (i = 0; i < n; ++i) {
        do {
            int time_line_index = i % mini_batch;
            unsigned int index = start_time_indexes[time_line_index] % m;
            start_time_indexes[time_line_index] += speed;

            //int index = random_gen() % m;
            sequentia_paths[i] = paths[index];
            //printf(" index = %d, ", index);
            //if(i == 0) printf("%s\n", paths[index]);
            //printf(" index = %u - grp: %s \n", index, paths[index]);
            if (strlen(sequentia_paths[i]) <= 4) printf(" Very small path to the image: %s \n", sequentia_paths[i]);
        } while (strlen(sequentia_paths[i]) == 0);
    }
    free(start_time_indexes);
    pthread_mutex_unlock(&mutex);
    return sequentia_paths;
}

char **get_random_paths_custom(char **paths, int n, int m, int contrastive)
{
    char** random_paths = (char**)xcalloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    int old_index = 0;
    //printf("n = %d \n", n);
    for(i = 0; i < n; ++i){
        do {
            int index = random_gen() % m;
            if (contrastive && (i % 2 == 1)) index = old_index;
            else old_index = index;
            random_paths[i] = paths[index];
            //if(i == 0) printf("%s\n", paths[index]);
            //printf("grp: %s\n", paths[index]);
            if (strlen(random_paths[i]) <= 4) printf(" Very small path to the image: %s \n", random_paths[i]);
        } while (strlen(random_paths[i]) == 0);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}

char **get_random_paths(char **paths, int n, int m)
{
    return get_random_paths_custom(paths, n, m, 0);
}

char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
    char** replace_paths = (char**)xcalloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        char replaced[4096];
        find_replace(paths[i], find, replace, replaced);
        replace_paths[i] = copy_string(replaced);
    }
    return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = (float**)xcalloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image(paths[i], w, h, 3);

        image gray = grayscale_image(im);
        free_image(im);
        im = gray;

        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_paths(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = (float**)xcalloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], w, h);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_augment_paths(char **paths, int n, int use_flip, int min, int max, int w, int h, float angle, float aspect, float hue, float saturation, float exposure, int dontuse_opencv, int contrastive)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = (float**)xcalloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        int size = w > h ? w : h;
        image im;
        const int img_index = (contrastive) ? (i / 2) : i;
        if(dontuse_opencv) im = load_image_stb_resize(paths[img_index], 0, 0, 3);
        else im = load_image_color(paths[img_index], 0, 0);

        image crop = random_augment_image(im, angle, aspect, min, max, size);
        int flip = use_flip ? random_gen() % 2 : 0;
        if (flip)
            flip_image(crop);
        random_distort_image(crop, hue, saturation, exposure);

        image sized = resize_image(crop, w, h);

        //show_image(im, "orig");
        //show_image(sized, "sized");
        //show_image(sized, paths[img_index]);
        //wait_until_press_key_cv();
        //printf("w = %d, h = %d \n", sized.w, sized.h);

        free_image(im);
        free_image(crop);
        X.vals[i] = sized.data;
        X.cols = sized.h*sized.w*sized.c;
    }
    return X;
}


box_label *read_boxes(char *filename, int *n)
{
    box_label* boxes = (box_label*)xcalloc(1, sizeof(box_label));
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Can't open label file. (This can be normal only if you use MSCOCO): %s \n", filename);
        //file_error(filename);
        FILE* fw = fopen("bad.list", "a");
        fwrite(filename, sizeof(char), strlen(filename), fw);
        char *new_line = "\n";
        fwrite(new_line, sizeof(char), strlen(new_line), fw);
        fclose(fw);

        *n = 0;
        return boxes;
    }
    const int max_obj_img = 4000;// 30000;
    const int img_hash = (custom_hash(filename) % max_obj_img)*max_obj_img;
    //printf(" img_hash = %d, filename = %s; ", img_hash, filename);
    float x, y, h, w;
    int id;
    int count = 0;
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        boxes = (box_label*)xrealloc(boxes, (count + 1) * sizeof(box_label));
        boxes[count].track_id = count + img_hash;
        //printf(" boxes[count].track_id = %d, count = %d \n", boxes[count].track_id, count);
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

void randomize_boxes(box_label *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        box_label swap = b[i];
        int index = random_gen()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
    int i;
    for(i = 0; i < n; ++i){
        if(boxes[i].x == 0 && boxes[i].y == 0) {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        if ((boxes[i].x + boxes[i].w / 2) < 0 || (boxes[i].y + boxes[i].h / 2) < 0 ||
            (boxes[i].x - boxes[i].w / 2) > 1 || (boxes[i].y - boxes[i].h / 2) > 1)
        {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        if(flip){
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        boxes[i].left =  constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count && i < 30; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .0 || h < .0) continue;

        int index = (4+classes) * i;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;

        if (id < classes) truth[index+id] = 1;
    }
    free(boxes);
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .001 || h < .001) continue;

        int col = (int)(x*num_boxes);
        int row = (int)(y*num_boxes);

        x = x*num_boxes - col;
        y = y*num_boxes - row;

        int index = (col+row*num_boxes)*(5+classes);
        if (truth[index]) continue;
        truth[index++] = 1;

        if (id < classes) truth[index+id] = 1;
        index += classes;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;
    }
    free(boxes);
}

int fill_truth_detection(const char *path, int num_boxes, int truth_size, float *truth, int classes, int flip, float dx, float dy, float sx, float sy,
    int net_w, int net_h)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    int i;
    box_label *boxes = read_boxes(labelpath, &count);
    int min_w_h = 0;
    float lowest_w = 1.F / net_w;
    float lowest_h = 1.F / net_h;
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    if (count > num_boxes) count = num_boxes;
    float x, y, w, h;
    int id;
    int sub = 0;

    for (i = 0; i < count; ++i) {
        x = boxes[i].x;
        y = boxes[i].y;
        w = boxes[i].w;
        h = boxes[i].h;
        id = boxes[i].id;
        int track_id = boxes[i].track_id;

        // not detect small objects
        //if ((w < 0.001F || h < 0.001F)) continue;
        // if truth (box for object) is smaller than 1x1 pix
        char buff[256];
        if (id >= classes) {
            printf("\n Wrong annotation: class_id = %d. But class_id should be [from 0 to %d], file: %s \n", id, (classes-1), labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: class_id = %d. But class_id should be [from 0 to %d]\" >> bad_label.list", labelpath, id, (classes-1));
            system(buff);
            ++sub;
            continue;
        }
        if ((w < lowest_w || h < lowest_h)) {
            //sprintf(buff, "echo %s \"Very small object: w < lowest_w OR h < lowest_h\" >> bad_label.list", labelpath);
            //system(buff);
            ++sub;
            continue;
        }
        if (x == 999999 || y == 999999) {
            printf("\n Wrong annotation: x = 0, y = 0, < 0 or > 1, file: %s \n", labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: x = 0 or y = 0\" >> bad_label.list", labelpath);
            system(buff);
            ++sub;
            continue;
        }
        if (x <= 0 || x > 1 || y <= 0 || y > 1) {
            printf("\n Wrong annotation: x = %f, y = %f, file: %s \n", x, y, labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: x = %f, y = %f\" >> bad_label.list", labelpath, x, y);
            system(buff);
            ++sub;
            continue;
        }
        if (w > 1) {
            printf("\n Wrong annotation: w = %f, file: %s \n", w, labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: w = %f\" >> bad_label.list", labelpath, w);
            system(buff);
            w = 1;
        }
        if (h > 1) {
            printf("\n Wrong annotation: h = %f, file: %s \n", h, labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: h = %f\" >> bad_label.list", labelpath, h);
            system(buff);
            h = 1;
        }
        if (x == 0) x += lowest_w;
        if (y == 0) y += lowest_h;

        truth[(i-sub)*truth_size +0] = x;
        truth[(i-sub)*truth_size +1] = y;
        truth[(i-sub)*truth_size +2] = w;
        truth[(i-sub)*truth_size +3] = h;
        truth[(i-sub)*truth_size +4] = id;
        truth[(i-sub)*truth_size +5] = track_id;
        //float val = track_id;
        //printf(" i = %d, sub = %d, truth_size = %d, track_id = %d, %f, %f\n", i, sub, truth_size, track_id, truth[(i - sub)*truth_size + 5], val);

        if (min_w_h == 0) min_w_h = w*net_w;
        if (min_w_h > w*net_w) min_w_h = w*net_w;
        if (min_w_h > h*net_h) min_w_h = h*net_h;
    }
    free(boxes);
    return min_w_h;
}


void print_letters(float *pred, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        int index = max_index(pred+i*NUMCHARS, NUMCHARS);
        printf("%c", int_to_alphanum(index));
    }
    printf("\n");
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    int count = 0;
    for(i = 0; i < k; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
            ++count;
        }
    }
    if (count != 1) {
        printf("Too many or too few labels: %d, %s\n", count, path);
        count = 0;
        for (i = 0; i < k; ++i) {
            if (strstr(path, labels[i])) {
                printf("\t label %d: %s  \n", count, labels[i]);
                count++;
            }
        }
    }
}

void fill_truth_smooth(char *path, char **labels, int k, float *truth, float label_smooth_eps)
{
    int i;
    memset(truth, 0, k * sizeof(float));
    int count = 0;
    for (i = 0; i < k; ++i) {
        if (strstr(path, labels[i])) {
            truth[i] = (1 - label_smooth_eps);
            ++count;
        }
        else {
            truth[i] = label_smooth_eps / (k - 1);
        }
    }
    if (count != 1) {
        printf("Too many or too few labels: %d, %s\n", count, path);
        count = 0;
        for (i = 0; i < k; ++i) {
            if (strstr(path, labels[i])) {
                printf("\t label %d: %s  \n", count, labels[i]);
                count++;
            }
        }
    }
}

void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
    int j;
    for(j = 0; j < k; ++j){
        if(truth[j]){
            int parent = hierarchy->parent[j];
            while(parent >= 0){
                truth[parent] = 1;
                parent = hierarchy->parent[parent];
            }
        }
    }
    int i;
    int count = 0;
    for(j = 0; j < hierarchy->groups; ++j){
        //printf("%d\n", count);
        int mask = 1;
        for(i = 0; i < hierarchy->group_size[j]; ++i){
            if(truth[count + i]){
                mask = 0;
                break;
            }
        }
        if (mask) {
            for(i = 0; i < hierarchy->group_size[j]; ++i){
                truth[count + i] = SECRET_NUM;
            }
        }
        count += hierarchy->group_size[j];
    }
}

int find_max(float *arr, int size) {
    int i;
    float max = 0;
    int n = 0;
    for (i = 0; i < size; ++i) {
        if (arr[i] > max) {
            max = arr[i];
            n = i;
        }
    }
    return n;
}

char **get_labels_custom(char *filename, int *size)
{
    list *plist = get_paths(filename);
    if(size) *size = plist->size;
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

char **get_labels(char *filename)
{
    return get_labels_custom(filename, NULL);
}

void free_data(data d)
{
    free(d.X.vals);
    free(d.y.vals);
    // if(!d.shallow){
    //     free_matrix(d.X);
    //     free_matrix(d.y);
    // }else{
    //     free(d.X.vals);
    //     free(d.y.vals);
    // }
}


void blend_truth(float *new_truth, int boxes, int truth_size, float *old_truth)
{
    int count_new_truth = 0;
    int t;
    for (t = 0; t < boxes; ++t) {
        float x = new_truth[t*truth_size];
        if (!x) break;
        count_new_truth++;

    }
    for (t = count_new_truth; t < boxes; ++t) {
        float *new_truth_ptr = new_truth + t*truth_size;
        float *old_truth_ptr = old_truth + (t - count_new_truth)*truth_size;
        float x = old_truth_ptr[0];
        if (!x) break;

        new_truth_ptr[0] = old_truth_ptr[0];
        new_truth_ptr[1] = old_truth_ptr[1];
        new_truth_ptr[2] = old_truth_ptr[2];
        new_truth_ptr[3] = old_truth_ptr[3];
        new_truth_ptr[4] = old_truth_ptr[4];
    }
    //printf("\n was %d bboxes, now %d bboxes \n", count_new_truth, t);
}


void blend_truth_mosaic(float *new_truth, int boxes, int truth_size, float *old_truth, int w, int h, float cut_x, float cut_y, int i_mixup,
    int left_shift, int right_shift, int top_shift, int bot_shift,
    int net_w, int net_h, int mosaic_bound)
{
    const float lowest_w = 1.F / net_w;
    const float lowest_h = 1.F / net_h;

    int count_new_truth = 0;
    int t;
    for (t = 0; t < boxes; ++t) {
        float x = new_truth[t*truth_size];
        if (!x) break;
        count_new_truth++;

    }
    int new_t = count_new_truth;
    for (t = count_new_truth; t < boxes; ++t) {
        float *new_truth_ptr = new_truth + new_t*truth_size;
        new_truth_ptr[0] = 0;
        float *old_truth_ptr = old_truth + (t - count_new_truth)*truth_size;
        float x = old_truth_ptr[0];
        if (!x) break;

        float xb = old_truth_ptr[0];
        float yb = old_truth_ptr[1];
        float wb = old_truth_ptr[2];
        float hb = old_truth_ptr[3];



        // shift 4 images
        if (i_mixup == 0) {
            xb = xb - (float)(w - cut_x - right_shift) / w;
            yb = yb - (float)(h - cut_y - bot_shift) / h;
        }
        if (i_mixup == 1) {
            xb = xb + (float)(cut_x - left_shift) / w;
            yb = yb - (float)(h - cut_y - bot_shift) / h;
        }
        if (i_mixup == 2) {
            xb = xb - (float)(w - cut_x - right_shift) / w;
            yb = yb + (float)(cut_y - top_shift) / h;
        }
        if (i_mixup == 3) {
            xb = xb + (float)(cut_x - left_shift) / w;
            yb = yb + (float)(cut_y - top_shift) / h;
        }

        int left = (xb - wb / 2)*w;
        int right = (xb + wb / 2)*w;
        int top = (yb - hb / 2)*h;
        int bot = (yb + hb / 2)*h;

        if(mosaic_bound)
        {
            // fix out of Mosaic-bound
            float left_bound = 0, right_bound = 0, top_bound = 0, bot_bound = 0;
            if (i_mixup == 0) {
                left_bound = 0;
                right_bound = cut_x;
                top_bound = 0;
                bot_bound = cut_y;
            }
            if (i_mixup == 1) {
                left_bound = cut_x;
                right_bound = w;
                top_bound = 0;
                bot_bound = cut_y;
            }
            if (i_mixup == 2) {
                left_bound = 0;
                right_bound = cut_x;
                top_bound = cut_y;
                bot_bound = h;
            }
            if (i_mixup == 3) {
                left_bound = cut_x;
                right_bound = w;
                top_bound = cut_y;
                bot_bound = h;
            }


            if (left < left_bound) {
                //printf(" i_mixup = %d, left = %d, left_bound = %f \n", i_mixup, left, left_bound);
                left = left_bound;
            }
            if (right > right_bound) {
                //printf(" i_mixup = %d, right = %d, right_bound = %f \n", i_mixup, right, right_bound);
                right = right_bound;
            }
            if (top < top_bound) top = top_bound;
            if (bot > bot_bound) bot = bot_bound;


            xb = ((float)(right + left) / 2) / w;
            wb = ((float)(right - left)) / w;
            yb = ((float)(bot + top) / 2) / h;
            hb = ((float)(bot - top)) / h;
        }
        else
        {
            // fix out of bound
            if (left < 0) {
                float diff = (float)left / w;
                xb = xb - diff / 2;
                wb = wb + diff;
            }

            if (right > w) {
                float diff = (float)(right - w) / w;
                xb = xb - diff / 2;
                wb = wb - diff;
            }

            if (top < 0) {
                float diff = (float)top / h;
                yb = yb - diff / 2;
                hb = hb + diff;
            }

            if (bot > h) {
                float diff = (float)(bot - h) / h;
                yb = yb - diff / 2;
                hb = hb - diff;
            }

            left = (xb - wb / 2)*w;
            right = (xb + wb / 2)*w;
            top = (yb - hb / 2)*h;
            bot = (yb + hb / 2)*h;
        }


        // leave only within the image
        if(left >= 0 && right <= w && top >= 0 && bot <= h &&
            wb > 0 && wb < 1 && hb > 0 && hb < 1 &&
            xb > 0 && xb < 1 && yb > 0 && yb < 1 &&
            wb > lowest_w && hb > lowest_h)
        {
            new_truth_ptr[0] = xb;
            new_truth_ptr[1] = yb;
            new_truth_ptr[2] = wb;
            new_truth_ptr[3] = hb;
            new_truth_ptr[4] = old_truth_ptr[4];
            new_t++;
        }
    }
    //printf("\n was %d bboxes, now %d bboxes \n", count_new_truth, t);
}

void blend_images(image new_img, float alpha, image old_img, float beta)
{
    int data_size = new_img.w * new_img.h * new_img.c;
    int i;
    #pragma omp parallel for
    for (i = 0; i < data_size; ++i)
        new_img.data[i] = new_img.data[i] * alpha + old_img.data[i] * beta;
}

// data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int truth_size, int classes, int use_flip, int gaussian_noise, int use_blur, int use_mixup,
//     float jitter, float resize, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int mosaic_bound, int contrastive, int contrastive_jit_flip, int contrastive_color, int show_imgs)
// {
//     const int random_index = random_gen();
//     c = c ? c : 3;
//     char **random_paths;
//     char **mixup_random_paths = NULL;
//     if(track) random_paths = get_sequential_paths(paths, n, m, mini_batch, augment_speed, contrastive);
//     else random_paths = get_random_paths_custom(paths, n, m, contrastive);

//     //assert(use_mixup < 2);
//     if (use_mixup == 2) {
//         error("cutmix=1 - isn't supported for Detector", DARKNET_LOC);
//     }
//     if (use_mixup == 3 || use_mixup == 4) {
//         error("mosaic=1 - compile Darknet with OpenCV for using mosaic=1", DARKNET_LOC);
//     }
//     int mixup = use_mixup ? random_gen() % 2 : 0;
//     //printf("\n mixup = %d \n", mixup);
//     if (mixup) {
//         if (track) mixup_random_paths = get_sequential_paths(paths, n, m, mini_batch, augment_speed, contrastive);
//         else mixup_random_paths = get_random_paths(paths, n, m);
//     }

//     int i;
//     data d = { 0 };
//     d.shallow = 0;

//     d.X.rows = n;
//     d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
//     d.X.cols = h*w*c;

//     float r1 = 0, r2 = 0, r3 = 0, r4 = 0, r_scale;
//     float resize_r1 = 0, resize_r2 = 0;
//     float dhue = 0, dsat = 0, dexp = 0, flip = 0;
//     int augmentation_calculated = 0;

//     d.y = make_matrix(n, truth_size * boxes);
//     int i_mixup = 0;
//     for (i_mixup = 0; i_mixup <= mixup; i_mixup++) {
//         if (i_mixup) augmentation_calculated = 0;
//         for (i = 0; i < n; ++i) {
//             float *truth = (float*)xcalloc(truth_size * boxes, sizeof(float));
//             char *filename = (i_mixup) ? mixup_random_paths[i] : random_paths[i];

//             image orig = load_image(filename, 0, 0, c);

//             int oh = orig.h;
//             int ow = orig.w;

//             int dw = (ow*jitter);
//             int dh = (oh*jitter);

//             float resize_down = resize, resize_up = resize;
//             if (resize_down > 1.0) resize_down = 1 / resize_down;
//             int min_rdw = ow*(1 - (1 / resize_down)) / 2;
//             int min_rdh = oh*(1 - (1 / resize_down)) / 2;

//             if (resize_up < 1.0) resize_up = 1 / resize_up;
//             int max_rdw = ow*(1 - (1 / resize_up)) / 2;
//             int max_rdh = oh*(1 - (1 / resize_up)) / 2;

//             if (!augmentation_calculated || !track)
//             {
//                 augmentation_calculated = 1;
//                 resize_r1 = random_float();
//                 resize_r2 = random_float();

//                 if (!contrastive || contrastive_jit_flip || i % 2 == 0)
//                 {
//                     r1 = random_float();
//                     r2 = random_float();
//                     r3 = random_float();
//                     r4 = random_float();

//                     flip = use_flip ? random_gen() % 2 : 0;
//                 }

//                 r_scale = random_float();

//                 if (!contrastive || contrastive_color || i % 2 == 0)
//                 {
//                     dhue = rand_uniform_strong(-hue, hue);
//                     dsat = rand_scale(saturation);
//                     dexp = rand_scale(exposure);
//                 }
//             }

//             int pleft = rand_precalc_random(-dw, dw, r1);
//             int pright = rand_precalc_random(-dw, dw, r2);
//             int ptop = rand_precalc_random(-dh, dh, r3);
//             int pbot = rand_precalc_random(-dh, dh, r4);

//             if (resize < 1) {
//                 // downsize only
//                 pleft += rand_precalc_random(min_rdw, 0, resize_r1);
//                 pright += rand_precalc_random(min_rdw, 0, resize_r2);
//                 ptop += rand_precalc_random(min_rdh, 0, resize_r1);
//                 pbot += rand_precalc_random(min_rdh, 0, resize_r2);
//             }
//             else {
//                 pleft += rand_precalc_random(min_rdw, max_rdw, resize_r1);
//                 pright += rand_precalc_random(min_rdw, max_rdw, resize_r2);
//                 ptop += rand_precalc_random(min_rdh, max_rdh, resize_r1);
//                 pbot += rand_precalc_random(min_rdh, max_rdh, resize_r2);
//             }

//             if (letter_box)
//             {
//                 float img_ar = (float)ow / (float)oh;
//                 float net_ar = (float)w / (float)h;
//                 float result_ar = img_ar / net_ar;
//                 //printf(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
//                 if (result_ar > 1)  // sheight - should be increased
//                 {
//                     float oh_tmp = ow / net_ar;
//                     float delta_h = (oh_tmp - oh) / 2;
//                     ptop = ptop - delta_h;
//                     pbot = pbot - delta_h;
//                     //printf(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
//                 }
//                 else  // swidth - should be increased
//                 {
//                     float ow_tmp = oh * net_ar;
//                     float delta_w = (ow_tmp - ow) / 2;
//                     pleft = pleft - delta_w;
//                     pright = pright - delta_w;
//                     //printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);
//                 }
//             }

//             int swidth = ow - pleft - pright;
//             int sheight = oh - ptop - pbot;

//             float sx = (float)swidth / ow;
//             float sy = (float)sheight / oh;

//             image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

//             float dx = ((float)pleft / ow) / sx;
//             float dy = ((float)ptop / oh) / sy;

//             image sized = resize_image(cropped, w, h);
//             if (flip) flip_image(sized);
//             distort_image(sized, dhue, dsat, dexp);
//             //random_distort_image(sized, hue, saturation, exposure);

//             fill_truth_detection(filename, boxes, truth_size, truth, classes, flip, dx, dy, 1. / sx, 1. / sy, w, h);

//             if (i_mixup) {
//                 image old_img = sized;
//                 old_img.data = d.X.vals[i];
//                 //show_image(sized, "new");
//                 //show_image(old_img, "old");
//                 //wait_until_press_key_cv();
//                 blend_images(sized, 0.5, old_img, 0.5);
//                 blend_truth(truth, boxes, truth_size, d.y.vals[i]);
//                 free_image(old_img);
//             }

//             d.X.vals[i] = sized.data;
//             memcpy(d.y.vals[i], truth, truth_size * boxes * sizeof(float));

//             if (show_imgs)// && i_mixup)
//             {
//                 char buff[1000];
//                 sprintf(buff, "aug_%d_%d_%s_%d", random_index, i, basecfg(filename), random_gen());

//                 int t;
//                 for (t = 0; t < boxes; ++t) {
//                     box b = float_to_box_stride(d.y.vals[i] + t*truth_size, 1);
//                     if (!b.x) break;
//                     int left = (b.x - b.w / 2.)*sized.w;
//                     int right = (b.x + b.w / 2.)*sized.w;
//                     int top = (b.y - b.h / 2.)*sized.h;
//                     int bot = (b.y + b.h / 2.)*sized.h;
//                     draw_box_width(sized, left, top, right, bot, 1, 150, 100, 50); // 3 channels RGB
//                 }

//                 save_image(sized, buff);
//                 if (show_imgs == 1) {
//                     show_image(sized, buff);
//                     wait_until_press_key_cv();
//                 }
//                 printf("\nYou use flag -show_imgs, so will be saved aug_...jpg images\n");
//             }

//             free_image(orig);
//             free_image(cropped);
//             free(truth);
//         }
//     }
//     free(random_paths);
//     if (mixup_random_paths) free(mixup_random_paths);
//     return d;
// }

void *load_thread(void *ptr)
{
    //srand(time(0));
    //printf("Loading data: %d\n", random_gen());
    load_args a = *(struct load_args*)ptr;
    if(a.exposure == 0) a.exposure = 1;
    if(a.saturation == 0) a.saturation = 1;
    if(a.aspect == 0) a.aspect = 1;

    if (a.type == IMAGE_DATA){
        *(a.im) = load_image(a.path, 0, 0, a.c);
        *(a.resized) = resize_image(*(a.im), a.w, a.h);
    }else if (a.type == LETTERBOX_DATA) {
        *(a.im) = load_image(a.path, 0, 0, a.c);
        *(a.resized) = letterbox_image(*(a.im), a.w, a.h);
    }
    free(ptr);
    return 0;
}

pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    struct load_args* ptr = (load_args*)xcalloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed", DARKNET_LOC);
    return thread;
}

static const int thread_wait_ms = 5;
static volatile int flag_exit;
static volatile int * run_load_data = NULL;
static load_args * args_swap = NULL;
static pthread_t* threads = NULL;

pthread_mutex_t mtx_load_data = PTHREAD_MUTEX_INITIALIZER;

void *run_thread_loop(void *ptr)
{
    const int i = *(int *)ptr;

    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_load_data[i])) {
            if (custom_atomic_load_int(&flag_exit)) {
                free(ptr);
                return 0;
            }
            this_thread_sleep_for(thread_wait_ms);
        }

        pthread_mutex_lock(&mtx_load_data);
        load_args *args_local = (load_args *)xcalloc(1, sizeof(load_args));
        *args_local = args_swap[i];
        pthread_mutex_unlock(&mtx_load_data);

        load_thread(args_local);

        custom_atomic_store_int(&run_load_data[i], 0);
    }
    free(ptr);
    return 0;
}

void *load_threads(void *ptr)
{
    //srand(time(0));
    int i;
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    data *out = args.d;
    int total = args.n;
    free(ptr);
    data* buffers = (data*)xcalloc(args.threads, sizeof(data));
    if (!threads) {
        threads = (pthread_t*)xcalloc(args.threads, sizeof(pthread_t));
        run_load_data = (volatile int *)xcalloc(args.threads, sizeof(int));
        args_swap = (load_args *)xcalloc(args.threads, sizeof(load_args));
        fprintf(stderr, " Create %d permanent cpu-threads \n", args.threads);

        for (i = 0; i < args.threads; ++i) {
            int* ptr = (int*)xcalloc(1, sizeof(int));
            *ptr = i;
            if (pthread_create(&threads[i], 0, run_thread_loop, ptr)) error("Thread creation failed", DARKNET_LOC);
        }
    }

    for (i = 0; i < args.threads; ++i) {
        args.d = buffers + i;
        args.n = (i + 1) * total / args.threads - i * total / args.threads;

        pthread_mutex_lock(&mtx_load_data);
        args_swap[i] = args;
        pthread_mutex_unlock(&mtx_load_data);

        custom_atomic_store_int(&run_load_data[i], 1);  // run thread
    }
    for (i = 0; i < args.threads; ++i) {
        while (custom_atomic_load_int(&run_load_data[i])) this_thread_sleep_for(thread_wait_ms); //   join
    }

    /*
    pthread_t* threads = (pthread_t*)xcalloc(args.threads, sizeof(pthread_t));
    for(i = 0; i < args.threads; ++i){
        args.d = buffers + i;
        args.n = (i+1) * total/args.threads - i * total/args.threads;
        threads[i] = load_data_in_thread(args);
    }
    for(i = 0; i < args.threads; ++i){
        pthread_join(threads[i], 0);
    }
    */

    *out = concat_datas(buffers, args.threads);
    out->shallow = 0;
    for(i = 0; i < args.threads; ++i){
        buffers[i].shallow = 1;
        free_data(buffers[i]);
    }
    free(buffers);
    //free(threads);
    return 0;
}

void free_load_threads(void *ptr)
{
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    int i;
    if (threads) {
        custom_atomic_store_int(&flag_exit, 1);
        for (i = 0; i < args.threads; ++i) {
            pthread_join(threads[i], 0);
        }
        free((void*)run_load_data);
        free(args_swap);
        free(threads);
        threads = NULL;
        custom_atomic_store_int(&flag_exit, 0);
    }
}

pthread_t load_data(load_args args)
{
    pthread_t thread;
    struct load_args* ptr = (load_args*)xcalloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_threads, ptr)) error("Thread creation failed", DARKNET_LOC);
    return thread;
}

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h)
{
    if(m) paths = get_random_paths(paths, n, m);
    char **replace_paths = find_replace_paths(paths, n, ".png", "-label.png");
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_image_paths_gray(replace_paths, n, out_w, out_h);
    if(m) free(paths);
    int i;
    for(i = 0; i < n; ++i) free(replace_paths[i]);
    free(replace_paths);
    return d;
}

// data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h)
// {
//     if(m) paths = get_random_paths(paths, n, m);
//     data d = {0};
//     d.shallow = 0;
//     d.X = load_image_paths(paths, n, w, h);
//     d.y = load_labels_paths(paths, n, labels, k, 0, 0, 0);
//     if(m) free(paths);
//     return d;
// }

/*
   data load_data_study(char **paths, int n, int m, char **labels, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
   {
   data d = {0};
   d.indexes = calloc(n, sizeof(int));
   if(m) paths = get_random_paths_indexes(paths, n, m, d.indexes);
   d.shallow = 0;
   d.X = load_image_augment_paths(paths, n, flip, min, max, size, angle, aspect, hue, saturation, exposure);
   d.y = load_labels_paths(paths, n, labels, k);
   if(m) free(paths);
   return d;
   }
 */

data load_data_super(char **paths, int n, int m, int w, int h, int scale)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;

    int i;
    d.X.rows = n;
    d.X.vals = (float**)xcalloc(n, sizeof(float*));
    d.X.cols = w*h*3;

    d.y.rows = n;
    d.y.vals = (float**)xcalloc(n, sizeof(float*));
    d.y.cols = w*scale * h*scale * 3;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop = random_crop_image(im, w*scale, h*scale);
        int flip = random_gen()%2;
        if (flip) flip_image(crop);
        image resize = resize_image(crop, w, h);
        d.X.vals[i] = resize.data;
        d.y.vals[i] = crop.data;
        free_image(im);
    }

    if(m) free(paths);
    return d;
}


matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    matrix m;
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = (float**)xcalloc(m1.rows + m2.rows, sizeof(float*));
    for(i = 0; i < m1.rows; ++i){
        m.vals[count++] = m1.vals[i];
    }
    for(i = 0; i < m2.rows; ++i){
        m.vals[count++] = m2.vals[i];
    }
    return m;
}

data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    return d;
}

data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i){
        data newdata = concat_data(d[i], out);
        free_data(out);
        out = newdata;
    }
    return out;
}

void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = random_gen()%d.X.rows;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void smooth_data(data d)
{
    int i, j;
    float scale = 1. / d.y.cols;
    float eps = .1;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; ++j){
            d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
        }
    }
}


void randomize_data(data d)
{
    int i;
    for(i = d.X.rows-1; i > 0; --i){
        int index = random_gen()%i;
        float *swap = d.X.vals[index];
        d.X.vals[index] = d.X.vals[i];
        d.X.vals[i] = swap;

        swap = d.y.vals[index];
        d.y.vals[index] = d.y.vals[i];
        d.y.vals[i] = swap;
    }
}

void scale_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        scale_array(d.X.vals[i], d.X.cols, s);
    }
}

void translate_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        translate_array(d.X.vals[i], d.X.cols, s);
    }
}

void normalize_data_rows(data d)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        normalize_array(d.X.vals[i], d.X.cols);
    }
}

data get_data_part(data d, int part, int total)
{
    data p = {0};
    p.shallow = 1;
    p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
    p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
    p.X.cols = d.X.cols;
    p.y.cols = d.y.cols;
    p.X.vals = d.X.vals + d.X.rows * part / total;
    p.y.vals = d.y.vals + d.y.rows * part / total;
    return p;
}

data get_random_data(data d, int num)
{
    data r = {0};
    r.shallow = 1;

    r.X.rows = num;
    r.y.rows = num;

    r.X.cols = d.X.cols;
    r.y.cols = d.y.cols;

    r.X.vals = (float**)xcalloc(num, sizeof(float*));
    r.y.vals = (float**)xcalloc(num, sizeof(float*));

    int i;
    for(i = 0; i < num; ++i){
        int index = random_gen()%d.X.rows;
        r.X.vals[i] = d.X.vals[index];
        r.y.vals[i] = d.y.vals[index];
    }
    return r;
}

data *split_data(data d, int part, int total)
{
    data* split = (data*)xcalloc(2, sizeof(data));
    int i;
    int start = part*d.X.rows/total;
    int end = (part+1)*d.X.rows/total;
    data train ={0};
    data test ={0};
    train.shallow = test.shallow = 1;

    test.X.rows = test.y.rows = end-start;
    train.X.rows = train.y.rows = d.X.rows - (end-start);
    train.X.cols = test.X.cols = d.X.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.X.vals = (float**)xcalloc(train.X.rows, sizeof(float*));
    test.X.vals = (float**)xcalloc(test.X.rows, sizeof(float*));
    train.y.vals = (float**)xcalloc(train.y.rows, sizeof(float*));
    test.y.vals = (float**)xcalloc(test.y.rows, sizeof(float*));

    for(i = 0; i < start; ++i){
        train.X.vals[i] = d.X.vals[i];
        train.y.vals[i] = d.y.vals[i];
    }
    for(i = start; i < end; ++i){
        test.X.vals[i-start] = d.X.vals[i];
        test.y.vals[i-start] = d.y.vals[i];
    }
    for(i = end; i < d.X.rows; ++i){
        train.X.vals[i-(end-start)] = d.X.vals[i];
        train.y.vals[i-(end-start)] = d.y.vals[i];
    }
    split[0] = train;
    split[1] = test;
    return split;
}
