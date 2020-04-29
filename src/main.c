#include <stdio.h>
#include <stdlib.h>
#include <loki/alloc.h>
#include <loki/channel_map_table.h>
#include <loki/control_registers.h>
#include <nn/layers.h>

// Random data used to select output channels.
// For any section of the array, discarding values below X will give a result
// which is roughly X% sparse.
// Generated using:
// python -c "import random; print([random.randint(0,99) for _ in range(5000)])"
int random[5000] = {
#include "data.txt"
};

// A compressed sparse tensor, with only the selected channels stored.
// Stored channels are stored in the normal dense way.
typedef struct {
  activation_config_t data;
  int* channels;
  int numChannels;
} sparse_activations_t;

// Create an activation tensor. Allocation of data and assignment to a memory
// group is not done. (User must set `address` and `memoryConfigEncoded`.)
// Dimension order is BCHW.
activation_config_t* init_activations(int batch_size, int channels, int height,
                                      int width) {
  activation_config_t* a = loki_malloc(sizeof(activation_config_t));
  a->rowSkip = sizeof(data_t);
  a->columnSkip = width * a->rowSkip;
  a->channelSkip = height * a->columnSkip;
  a->batchSkip = channels * a->channelSkip;

  return a;
}

// Specialisation of activation_config_t for sparse activations.
// User must also set `channels` array.
sparse_activations_t* init_sparse(int batch_size, int channels, int height,
                                  int width) {
  sparse_activations_t* a = loki_malloc(sizeof(sparse_activations_t));
  a->data.rowSkip = sizeof(data_t);
  a->data.columnSkip = width * a->data.rowSkip;
  a->data.channelSkip = height * a->data.columnSkip;
  a->data.batchSkip = channels * a->data.channelSkip;
  a->numChannels = channels;

  return a;
}

// Create a weight tensor. Allocation of data and assignment to a memory
// group is not done. (User must set `address` and `memoryConfigEncoded`.)
// Dimension order is OIHW.
filter_config_t* init_weights(int in_channels, int out_channels,
                              int filter_height, int filter_width) {
  filter_config_t* f = loki_malloc(sizeof(filter_config_t));
  f->rowSkip = sizeof(data_t);
  f->columnSkip = filter_width * f->rowSkip;
  f->outChannelSkip = filter_height * f->columnSkip;
  f->inChannelSkip = out_channels * f->outChannelSkip;

  // TODO: groupSkip. Not used for anything in this test.

  return f;
}

void test_none(int in_channels, int in_size, int in_sparsity,
               int out_channels, int out_sparsity, int filter_size) {

  // Create some memory groups, allowing data to be physically partitioned.
  // The CPU group should be used wherever an array is accessed in C code.
  channel_t mem_group_cpu = get_channel_map(1);
  channel_t mem_group_1 = mem_group_cpu; // TODO: use something specific
  channel_t mem_group_2 = mem_group_cpu; // TODO: use something specific
  channel_t mem_group_3 = mem_group_cpu; // TODO: use something specific

  // Use uninitialised data for weights and activations.
  // This will not affect the result unless fine-grained sparsity is exploited,
  // or data is compressed.
  data_t* input_ptr = loki_malloc(in_channels * in_size * in_size * sizeof(data_t));
  data_t* weight_ptr = loki_malloc(in_channels * out_channels * filter_size * filter_size * sizeof(data_t));
  int out_size = in_size - filter_size + 1;
  data_t* output_ptr = loki_malloc(out_channels * out_size * out_size * sizeof(data_t));
  assert(input_ptr != NULL);
  assert(weight_ptr != NULL);
  assert(output_ptr != NULL);

  // Step 0: create all necessary data buffers.
  // Assuming batch size of 1. Having larger batches is difficult for dynamic
  // workloads because each sample can take a different computation path.
  activation_config_t* input = init_activations(1, in_channels, in_size, in_size);
  input->address = input_ptr;
  input->memoryConfigEncoded = mem_group_1;

  filter_config_t* weights = init_weights(in_channels, out_channels, filter_size, filter_size);
  weights->address = weight_ptr;
  weights->memoryConfigEncoded = mem_group_2;

  activation_config_t* output = init_activations(1, out_channels, out_size, out_size);
  output->address = output_ptr;
  output->memoryConfigEncoded = mem_group_3;

  conv_shape_t shape;
  shape.batchSize = 1;
  shape.inChannels = in_channels;
  shape.outChannels = out_channels;
  shape.imageWidth = in_size;
  shape.imageHeight = in_size;
  shape.filterWidth = filter_size;
  shape.filterHeight = filter_size;
  shape.groups = 1;

  // Start timer.
  unsigned long start = get_cycle_count();

  // Step 1: downsample inputs. Unused.
  // Step 2: auxiliary convolution. Unused.
  // Steps 3+4: discard any features below a threshold. Unused.

  // Step 5: sparse convolution.
  // 'none' mode: convolution isn't sparse at all.
  lat_conv2d(input, weights, output, &shape, 1, 0);

  // Stop timer.
  unsigned long duration = get_cycle_count() - start;
  printf("%d inputs -> %d outputs took %lu cycles\n",
         in_channels, out_channels, duration);

  loki_free(output->address);
  loki_free(input);
  loki_free(weights);
  loki_free(output);

}

void test_simple(int in_channels, int in_size, int in_sparsity,
                 int out_channels, int out_sparsity, int filter_size) {

  // Create some memory groups, allowing data to be physically partitioned.
  // The CPU group should be used wherever an array is accessed in C code.
  channel_t mem_group_cpu = get_channel_map(1);
  channel_t mem_group_1 = mem_group_cpu; // TODO: use something specific
  channel_t mem_group_2 = mem_group_cpu; // TODO: use something specific
  channel_t mem_group_3 = mem_group_cpu; // TODO: use something specific

  // Determine how many input channels to use, given the sparsity.
  // In practice, we would be given the sparse data, but here we use random
  // numbers to select them.
  int in_channels_count = 0;
  int* in_channels_used = loki_malloc(in_channels * sizeof(int));
  for (int i=0; i<in_channels; i++)
    if (random[i] > in_sparsity)
      in_channels_used[in_channels_count++] = i;

  // Use uninitialised data for weights and activations.
  // This will not affect the result unless fine-grained sparsity is exploited,
  // or data is compressed.
  data_t* input_ptr = loki_malloc(in_channels_count * in_size * in_size * sizeof(data_t));
  data_t* weight_ptr = loki_malloc(in_channels * out_channels * filter_size * filter_size * sizeof(data_t));
  data_t* aux_weight_ptr = loki_malloc(in_channels * out_channels * sizeof(data_t));
  assert(input_ptr != NULL);
  assert(weight_ptr != NULL);
  assert(aux_weight_ptr != NULL);

  // Step 0: create all necessary data buffers.
  // Assuming batch size of 1. Having larger batches is difficult for dynamic
  // workloads because each sample can take a different computation path.
  sparse_activations_t* input = init_sparse(1, in_channels_count, in_size, in_size);
  input->data.address = input_ptr;
  input->data.memoryConfigEncoded = mem_group_1;
  input->channels = in_channels_used;

  sparse_activations_t* downsampled = init_sparse(1, in_channels_count, 1, 1);
  downsampled->data.address = loki_malloc(in_channels_count * sizeof(int));
  downsampled->data.memoryConfigEncoded = mem_group_cpu;
  downsampled->channels = input->channels;

  filter_config_t* weights = init_weights(in_channels, out_channels, filter_size, filter_size);
  weights->address = weight_ptr;
  weights->memoryConfigEncoded = mem_group_2;

  activation_config_t* aux_in = init_activations(1, in_channels, 1, 1);
  aux_in->address = loki_malloc(in_channels * sizeof(int));
  aux_in->memoryConfigEncoded = mem_group_cpu;

  filter_config_t* aux_weights = init_weights(in_channels, out_channels, 1, 1);
  aux_weights->address = aux_weight_ptr;
  aux_weights->memoryConfigEncoded = mem_group_2;

  activation_config_t* aux_out = init_activations(1, out_channels, 1, 1);
  aux_out->address = loki_malloc(out_channels * sizeof(int));
  aux_out->memoryConfigEncoded = mem_group_cpu;

  int out_size = in_size - filter_size + 1;
  // Can't allocate output until we've computed the auxiliary layer and know how
  // many channels there will be.

  // Start timer.
  unsigned long start = get_cycle_count();

  // Step 1: downsample inputs.
  pool_shape_t pool_params;
  pool_params.batchSize = 1;
  pool_params.channels = input->numChannels;
  pool_params.inputWidth = in_size;
  pool_params.inputHeight = in_size;
  pool_params.windowWidth = in_size;
  pool_params.windowHeight = in_size;
  pool_params.stride = in_size;
  lat_max_pool_2d(&(input->data), &(downsampled->data), &pool_params);

  // Step 2: auxiliary convolution. Since we downsampled the inputs to 1x1, this
  // is equivalent to a fully-connected/linear layer.
  // First scatter the sparse data into the dense auxiliary input.
  for (int i=0; i<downsampled->numChannels; i++) {
    int channel = downsampled->channels[i];
    aux_in->address[channel] = downsampled->data.address[i];
  }
  lat_linear(aux_in, aux_weights, aux_out, 1, in_channels, out_channels);

  // Steps 3+4: discard any features below a threshold.
  // In order to have more control over the sparsity achieved, a predetermined
  // random sequence is used for this, instead of the output of step 2.
  int out_channels_count = 0;
  int* out_channels_used = loki_malloc(out_channels * sizeof(int));
  for (int i=0; i<out_channels; i++)
    if (random[5000 - i] > out_sparsity)
      out_channels_used[out_channels_count++] = i;

  // We now know how much output will be produced.
  sparse_activations_t* output = init_sparse(1, out_channels_count, out_size, out_size);
  output->data.address = loki_malloc(out_channels_count * out_size * out_size * sizeof(int));
  output->data.memoryConfigEncoded = mem_group_3;
  output->channels = out_channels_used;

  // Step 5: sparse convolution.
  // 'simple' mode: repeatedly apply one filter to one input channel.
  conv_shape_t shape;
  shape.batchSize = 1;
  shape.inChannels = 1;
  shape.outChannels = 1;
  shape.imageWidth = in_size;
  shape.imageHeight = in_size;
  shape.filterWidth = filter_size;
  shape.filterHeight = filter_size;
  shape.groups = 1;

  activation_config_t conv_i;
  conv_i.memoryConfigEncoded = input->data.memoryConfigEncoded;
  conv_i.batchSkip = input->data.batchSkip;
  conv_i.channelSkip = input->data.channelSkip;
  conv_i.columnSkip = input->data.columnSkip;
  conv_i.rowSkip = input->data.rowSkip;

  filter_config_t conv_w;
  conv_w.memoryConfigEncoded = weights->memoryConfigEncoded;
  conv_w.inChannelSkip = weights->inChannelSkip;
  conv_w.outChannelSkip = weights->outChannelSkip;
  conv_w.columnSkip = weights->columnSkip;
  conv_w.rowSkip = weights->rowSkip;
  conv_w.groupSkip = weights->groupSkip;

  activation_config_t conv_o;
  conv_o.memoryConfigEncoded = output->data.memoryConfigEncoded;
  conv_o.batchSkip = output->data.batchSkip;
  conv_o.channelSkip = output->data.channelSkip;
  conv_o.columnSkip = output->data.columnSkip;
  conv_o.rowSkip = output->data.rowSkip;

  for (int o=0; o<output->numChannels; o++) {
    for (int i=0; i<input->numChannels; i++) {
      int out_c = output->channels[o];
      int in_c = input->channels[i];

      conv_i.address = input->data.address + in_c * input->data.channelSkip;
      conv_w.address = weights->address + in_c * weights->inChannelSkip
                                        + out_c * weights->outChannelSkip;
      conv_o.address = output->data.address + out_c * output->data.channelSkip;

      lat_conv2d(&conv_i, &conv_w, &conv_o, &shape, 1, 0);
    }
  }

  // Stop timer.
  unsigned long duration = get_cycle_count() - start;
  printf("%d/%d inputs -> %d/%d outputs took %lu cycles\n",
         in_channels_count, in_channels, out_channels_count, out_channels,
         duration);

  loki_free(downsampled->data.address);
  loki_free(aux_in->address);
  loki_free(aux_out->address);
  loki_free(output->data.address);
  loki_free(input);
  loki_free(downsampled);
  loki_free(weights);
  loki_free(aux_in);
  loki_free(aux_weights);
  loki_free(aux_out);
  loki_free(output);
  loki_free(in_channels_used);
  loki_free(out_channels_used);

}

// TODO: reduce code duplication.
// This is identical to test_simple except the for loops in step 5.
void test_adaptive(int in_channels, int in_size, int in_sparsity,
                   int out_channels, int out_sparsity, int filter_size) {

  // Create some memory groups, allowing data to be physically partitioned.
  // The CPU group should be used wherever an array is accessed in C code.
  channel_t mem_group_cpu = get_channel_map(1);
  channel_t mem_group_1 = mem_group_cpu; // TODO: use something specific
  channel_t mem_group_2 = mem_group_cpu; // TODO: use something specific
  channel_t mem_group_3 = mem_group_cpu; // TODO: use something specific

  // Determine how many input channels to use, given the sparsity.
  // In practice, we would be given the sparse data, but here we use random
  // numbers to select them.
  int in_channels_count = 0;
  int* in_channels_used = loki_malloc(in_channels * sizeof(int));
  for (int i=0; i<in_channels; i++)
    if (random[i] > in_sparsity)
      in_channels_used[in_channels_count++] = i;

  // Use uninitialised data for weights and activations.
  // This will not affect the result unless fine-grained sparsity is exploited,
  // or data is compressed.
  data_t* input_ptr = loki_malloc(in_channels_count * in_size * in_size * sizeof(data_t));
  data_t* weight_ptr = loki_malloc(in_channels * out_channels * filter_size * filter_size * sizeof(data_t));
  data_t* aux_weight_ptr = loki_malloc(in_channels * out_channels * sizeof(data_t));
  assert(input_ptr != NULL);
  assert(weight_ptr != NULL);
  assert(aux_weight_ptr != NULL);

  // Step 0: create all necessary data buffers.
  // Assuming batch size of 1. Having larger batches is difficult for dynamic
  // workloads because each sample can take a different computation path.
  sparse_activations_t* input = init_sparse(1, in_channels_count, in_size, in_size);
  input->data.address = input_ptr;
  input->data.memoryConfigEncoded = mem_group_1;
  input->channels = in_channels_used;

  sparse_activations_t* downsampled = init_sparse(1, in_channels_count, 1, 1);
  downsampled->data.address = loki_malloc(in_channels_count * sizeof(int));
  downsampled->data.memoryConfigEncoded = mem_group_cpu;
  downsampled->channels = input->channels;

  filter_config_t* weights = init_weights(in_channels, out_channels, filter_size, filter_size);
  weights->address = weight_ptr;
  weights->memoryConfigEncoded = mem_group_2;

  activation_config_t* aux_in = init_activations(1, in_channels, 1, 1);
  aux_in->address = loki_malloc(in_channels * sizeof(int));
  aux_in->memoryConfigEncoded = mem_group_cpu;

  filter_config_t* aux_weights = init_weights(in_channels, out_channels, 1, 1);
  aux_weights->address = aux_weight_ptr;
  aux_weights->memoryConfigEncoded = mem_group_2;

  activation_config_t* aux_out = init_activations(1, out_channels, 1, 1);
  aux_out->address = loki_malloc(out_channels * sizeof(int));
  aux_out->memoryConfigEncoded = mem_group_cpu;

  int out_size = in_size - filter_size + 1;
  // Can't allocate output until we've computed the auxiliary layer and know how
  // many channels there will be.

  // Start timer.
  unsigned long start = get_cycle_count();

  // Step 1: downsample inputs.
  pool_shape_t pool_params;
  pool_params.batchSize = 1;
  pool_params.channels = input->numChannels;
  pool_params.inputWidth = in_size;
  pool_params.inputHeight = in_size;
  pool_params.windowWidth = in_size;
  pool_params.windowHeight = in_size;
  pool_params.stride = in_size;
  lat_max_pool_2d(&(input->data), &(downsampled->data), &pool_params);

  // Step 2: auxiliary convolution. Since we downsampled the inputs to 1x1, this
  // is equivalent to a fully-connected/linear layer.
  // First scatter the sparse data into the dense auxiliary input.
  for (int i=0; i<downsampled->numChannels; i++) {
    int channel = downsampled->channels[i];
    aux_in->address[channel] = downsampled->data.address[i];
  }
  lat_linear(aux_in, aux_weights, aux_out, 1, in_channels, out_channels);

  // Steps 3+4: discard any features below a threshold.
  // In order to have more control over the sparsity achieved, a predetermined
  // random sequence is used for this, instead of the output of step 2.
  int out_channels_count = 0;
  int* out_channels_used = loki_malloc(out_channels * sizeof(int));
  for (int i=0; i<out_channels; i++)
    if (random[5000 - i] > out_sparsity)
      out_channels_used[out_channels_count++] = i;

  // We now know how much output will be produced.
  sparse_activations_t* output = init_sparse(1, out_channels_count, out_size, out_size);
  output->data.address = loki_malloc(out_channels_count * out_size * out_size * sizeof(int));
  output->data.memoryConfigEncoded = mem_group_3;
  output->channels = out_channels_used;

  // Step 5: sparse convolution.
  // 'adaptive' mode: look for sequences of consecutive channels available, and
  //                  apply multi-channel convolutions where possible.
  conv_shape_t shape;
  shape.batchSize = 1;
  shape.imageWidth = in_size;
  shape.imageHeight = in_size;
  shape.filterWidth = filter_size;
  shape.filterHeight = filter_size;
  shape.groups = 1;

  activation_config_t conv_i;
  conv_i.memoryConfigEncoded = input->data.memoryConfigEncoded;
  conv_i.batchSkip = input->data.batchSkip;
  conv_i.channelSkip = input->data.channelSkip;
  conv_i.columnSkip = input->data.columnSkip;
  conv_i.rowSkip = input->data.rowSkip;

  filter_config_t conv_w;
  conv_w.memoryConfigEncoded = weights->memoryConfigEncoded;
  conv_w.inChannelSkip = weights->inChannelSkip;
  conv_w.outChannelSkip = weights->outChannelSkip;
  conv_w.columnSkip = weights->columnSkip;
  conv_w.rowSkip = weights->rowSkip;
  conv_w.groupSkip = weights->groupSkip;

  activation_config_t conv_o;
  conv_o.memoryConfigEncoded = output->data.memoryConfigEncoded;
  conv_o.batchSkip = output->data.batchSkip;
  conv_o.channelSkip = output->data.channelSkip;
  conv_o.columnSkip = output->data.columnSkip;
  conv_o.rowSkip = output->data.rowSkip;

  for (int o=0; o<output->numChannels; /*update within loop*/) {

    // Count contiguous output channels.
    shape.outChannels = 1;
    while ((o + shape.outChannels < out_channels) &&
           (output->channels[o + shape.outChannels] == output->channels[o] + shape.outChannels))
      shape.outChannels++;

    for (int i=0; i<input->numChannels; /*update within loop*/) {
      // Count contiguous input channels.
      shape.inChannels = 1;
      while ((i + shape.inChannels < in_channels) &&
             (input->channels[i + shape.inChannels] == input->channels[i] + shape.inChannels))
        shape.inChannels++;

      int out_c = output->channels[o];
      int in_c = input->channels[i];

      conv_i.address = input->data.address + in_c * input->data.channelSkip;
      conv_w.address = weights->address + in_c * weights->inChannelSkip
                                        + out_c * weights->outChannelSkip;
      conv_o.address = output->data.address + out_c * output->data.channelSkip;

      // printf("%lu x %lu mini-conv\n", shape.inChannels, shape.outChannels);
      lat_conv2d(&conv_i, &conv_w, &conv_o, &shape, 1, 0);
      // Potential optimisation: set up the next convolution while waiting for
      // this one to finish.

      i += shape.inChannels;
    }

    o += shape.outChannels;
  }

  // Stop timer.
  unsigned long duration = get_cycle_count() - start;
  printf("%d/%d inputs -> %d/%d outputs took %lu cycles\n",
         in_channels_count, in_channels, out_channels_count, out_channels,
         duration);

  loki_free(downsampled->data.address);
  loki_free(aux_in->address);
  loki_free(aux_out->address);
  loki_free(output->data.address);
  loki_free(input);
  loki_free(downsampled);
  loki_free(weights);
  loki_free(aux_in);
  loki_free(aux_weights);
  loki_free(aux_out);
  loki_free(output);
  loki_free(in_channels_used);
  loki_free(out_channels_used);

}

int main(int argc, char** argv) {
  if (argc < 7) {
    printf(""
    "Usage: lat-dynamic in-channels in-size in-sparsity out-channels\\ \n"
    "       out-sparsity filter-size [mode]\n"
    "'size' parameters indicate the width/height in pixels\n"
    "'sparsity' parameters are percentages\n"
    "'mode' selects how to exploit sparsity ('none', 'simple', 'adaptive')\n");
    exit(1);
  }

  int in_channels = atoi(argv[1]);
  int in_size = atoi(argv[2]);
  int in_sparsity = atoi(argv[3]);
  int out_channels = atoi(argv[4]);
  int out_sparsity = atoi(argv[5]);
  int filter_size = atoi(argv[6]);
  char* mode = "simple";
  if (argc > 7)
    mode = argv[7];

  // A pre-allocated array of random numbers is used to choose which channels to
  // skip over. (Generating random numbers is expensive to simulate.)
  assert(in_channels + out_channels < 5000);

  if (!strcmp(mode, "none"))
    test_none(in_channels, in_size, in_sparsity, out_channels, out_sparsity,
              filter_size);
  else if (!strcmp(mode, "simple"))
    test_simple(in_channels, in_size, in_sparsity, out_channels, out_sparsity,
                filter_size);
  else if (!strcmp(mode, "adaptive"))
    test_adaptive(in_channels, in_size, in_sparsity, out_channels, out_sparsity,
                  filter_size);
  else {
    printf("Error: unknown mode parameter: '%s'\n", mode);
    exit(1);
  }

  return 0;
}
