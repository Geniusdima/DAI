#include "aiheaders.h"

extern void cleanup_network(network*net)
{
    for (int i = 0; i < net->count; i++)
    {    
        for (int j = 0; j < net->layers[i].count; j++)
        {
            if(i != net->count -1)
                free(net->layers[i].neurons[j].weight);
        }   
        free(net->layers[i].neurons);
    }
    free(net->layers);
}

extern void print_values_perceptron(network * net)
{
    for (int i = 0; i < net->count; i++)
    {
        for(int j = 0; j < net->layers[i].count;j++)
        {
            printf("value of neuron %d on layer %d %f\n", j, i, net->layers[i].neurons[j].value);
        }
    }
}

extern void print_weights_perceptron(network * net)
{
    for (int i = 0; i < net->count; i++)
        for(int j = 0; j < net->layers[i].count;j++)
            if(i != net->count - 1)
                for(int k = 0; k < net->layers[i+1].count;k++)
                    printf("weight of neuron %d on layer %d %f\n", j, i ,net->layers[i].neurons[j].weight[k]);
}