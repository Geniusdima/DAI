#ifndef AI
#define AI

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fcntl.h>

typedef struct neuron neuron;
typedef struct layer layer;
typedef struct network network;
typedef struct schematic schematic;

struct neuron
{
    float * weight;
    float value;
    float gradient;
};

struct layer
{
    neuron * neurons;
    int count;
};

struct network
{
    int count;
    layer * layers;
    schematic * schms;
    int count_schms;
};

struct schematic
{    
    int layer_count;
    int neuron_count; //count neurons in layer 
};

extern network load_network(char * file_name);
extern void save_network(network * net, char * save_name);
extern network init_perceptron(schematic schm[], int schematic_count);
extern void cleanup_network(network*net);
extern float init_weight();
extern float dsigmoid(float x);
extern float sigmoid(float x);
extern void print_weights_perceptron(network * net);
extern void print_values_perceptron(network * net);
extern void forwardprop_perceptron(network * net);
extern float mse(float predicted, float correct);
extern void backprop_perceptron(network * net, float step, float * correct, int correct_count);

#endif