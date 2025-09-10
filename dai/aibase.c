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
    free(net->schms);
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

extern void save_network(network * net, char * save_name)
{
    FILE * f = fopen(save_name, "w");

    fwrite(&net->count_schms,sizeof(net->count_schms), 1, f);
    
    fwrite(net->schms, sizeof(schematic), net->count_schms, f);

    for(int i = 0; i < net->count-1; i++)
        for(int j = 0; j < net->layers[i].count; j++)
            fwrite(net->layers[i].neurons[j].weight, sizeof(float), net->layers[i+1].count, f);

    fclose(f);
}

extern network load_network(char * file_name)
{
    FILE * f = fopen(file_name, "r");
    int schm_count = 0;
    
    fread(&schm_count, sizeof(int), 1, f);

    schematic schm[schm_count];

    fread(schm, sizeof(schematic), schm_count, f);
    network net = init_perceptron(schm, schm_count);

    int weight_count = 0;
    
    for(int i = 0; i < net.count - 1; i++)
    {
        for(int j = 0; j < net.layers[i].count; j++)
        {
            float * weights = (float*)malloc(sizeof(float) * net.layers[i+1].count);
            fread(weights, sizeof(float), net.layers[i+1].count, f);
            
            for(int k = 0; k < net.layers[i+1].count; k++)
                net.layers[i].neurons[j].weight[k] = weights[k];
            
            free(weights);
        }
    }

    fclose(f);
    return net;
}
