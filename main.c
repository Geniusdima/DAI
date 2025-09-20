#include "dai/aiheaders.h"

#define COUNT 1000000
#define STEP 0.08
#define TRAINING

int main()
{
    srand(time(NULL));
    #ifdef TRAINING
    schematic schm[] = 
    {
        {.layer_count=1, .neuron_count=2},        
        {.layer_count=2, .neuron_count=8},
        {.layer_count=1, .neuron_count=3},
        {.layer_count=1, .neuron_count=1},
    };

    float corr[4][2] =
    {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    float corr1[1] = {0};

    float corr2[1] = {1};
    
    network net = init_perceptron(schm,4);

    srand(time(NULL));

    int r;
    clock_t start, end;
    double cpu_time_used;
    
    start = clock();
    for (int i = 0; i < COUNT; i++)
    {
        r = rand()%4;

        net.layers[0].neurons[0].value=corr[r][0];
        net.layers[0].neurons[1].value=corr[r][1];

        if(r == 3)
        {
            backprop_perceptron(&net, STEP, corr2, 1);
        }
        else
        {
            backprop_perceptron(&net, STEP, corr1, 1);
        }
    }

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Time taken: %.6f seconds\n", cpu_time_used);


    int d1,d2;

    for(int i = 0; i < 5; i++)
    {
        scanf("%d %d",&d1,&d2);
        net.layers[0].neurons[0].value = d1;
        net.layers[0].neurons[1].value = d2;
        forwardprop_perceptron(&net);
        printf("%f\n",net.layers[net.count-1].neurons[0].value);
    }
    
    save_network(&net,"AND.bin");
    cleanup_network(&net);
    #endif
    
    #ifndef TRAINING

    network net = load_network("./bestNetworks/XOR.bin");
    int d1,d2;
    while(1)
    {
        scanf("%d %d",&d1,&d2);
        net.layers[0].neurons[0].value = d1;
        net.layers[0].neurons[1].value = d2;
        forwardprop_perceptron(&net);
        printf("%f\n",net.layers[net.count-1].neurons[0].value);
    }

    cleanup_network(&net);

    #endif

    return 0;    
}