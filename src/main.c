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
  activation_config_t dense;
  int* channels;
  int num_channels;
} sparse_activations_t;

// Create an activation tensor. Allocation of data and assignment to a memory
// group is not done. (User must set `address` and `data.memory_config`.)
// Dimension order is BCHW.
activation_config_t* init_activations(int batch_size, int channels, int height,
                                      int width) {
  activation_config_t* a = loki_malloc(sizeof(activation_config_t));
  a->row_stride = sizeof(data_t);
  a->column_stride = width * a->row_stride;
  a->channel_stride = height * a->column_stride;
  a->batch_stride = channels * a->channel_stride;

  return a;
}

// Specialisation of activation_config_t for sparse activations.
// User must also set `channels` array.
sparse_activations_t* init_sparse(int batch_size, int channels, int height,
                                  int width) {
  sparse_activations_t* a = loki_malloc(sizeof(sparse_activations_t));
  a->dense.row_stride = sizeof(data_t);
  a->dense.column_stride = width * a->dense.row_stride;
  a->dense.channel_stride = height * a->dense.column_stride;
  a->dense.batch_stride = channels * a->dense.channel_stride;
  a->num_channels = channels;

  return a;
}

// Create a weight tensor. Allocation of data and assignment to a memory
// group is not done. (User must set `address` and `data.memory_config`.)
// Dimension order is OIHW.
filter_config_t* init_weights(int in_channels, int out_channels,
                              int filter_height, int filter_width) {
  filter_config_t* f = loki_malloc(sizeof(filter_config_t));
  f->row_stride = sizeof(data_t);
  f->column_stride = filter_width * f->row_stride;
  f->out_channel_stride = filter_height * f->column_stride;
  f->in_channel_stride = out_channels * f->out_channel_stride;

  // TODO: group_stride. Not used for anything in this test.

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
  input->data.address = input_ptr;
  input->data.memory_config = mem_group_1;

  filter_config_t* weights = init_weights(in_channels, out_channels, filter_size, filter_size);
  weights->data.address = weight_ptr;
  weights->data.memory_config = mem_group_2;

  activation_config_t* output = init_activations(1, out_channels, out_size, out_size);
  output->data.address = output_ptr;
  output->data.memory_config = mem_group_3;

  conv_shape_t shape;
  shape.batch_size = 1;
  shape.in_channels = in_channels;
  shape.out_channels = out_channels;
  shape.image_width = in_size;
  shape.image_height = in_size;
  shape.filter_width = filter_size;
  shape.filter_height = filter_size;
  shape.groups = 1;
  shape.stride = 1;
  shape.dilation = 1;

  // Start timer.
  unsigned long start = get_cycle_count();

  // Step 1: downsample inputs. Unused.
  // Step 2: auxiliary convolution. Unused.
  // Steps 3+4: discard any features below a threshold. Unused.

  // Step 5: sparse convolution.
  // 'none' mode: convolution isn't sparse at all.
  lat_conv2d(input, weights, output, &shape, &LOOP_NEST_OUTPUT_STATIONARY);

  // Stop timer.
  unsigned long duration = get_cycle_count() - start;
  printf("%d inputs -> %d outputs took %lu cycles\n",
         in_channels, out_channels, duration);

  loki_free(output->data.address);
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
  input->dense.data.address = input_ptr;
  input->dense.data.memory_config = mem_group_1;
  input->channels = in_channels_used;

  sparse_activations_t* downsampled = init_sparse(1, in_channels_count, 1, 1);
  downsampled->dense.data.address = loki_malloc(in_channels_count * sizeof(int));
  downsampled->dense.data.memory_config = mem_group_cpu;
  downsampled->channels = input->channels;

  filter_config_t* weights = init_weights(in_channels, out_channels, filter_size, filter_size);
  weights->data.address = weight_ptr;
  weights->data.memory_config = mem_group_2;

  activation_config_t* aux_in = init_activations(1, in_channels, 1, 1);
  aux_in->data.address = loki_malloc(in_channels * sizeof(int));
  aux_in->data.memory_config = mem_group_cpu;

  filter_config_t* aux_weights = init_weights(in_channels, out_channels, 1, 1);
  aux_weights->data.address = aux_weight_ptr;
  aux_weights->data.memory_config = mem_group_2;

  activation_config_t* aux_out = init_activations(1, out_channels, 1, 1);
  aux_out->data.address = loki_malloc(out_channels * sizeof(int));
  aux_out->data.memory_config = mem_group_cpu;

  int out_size = in_size - filter_size + 1;
  // Can't allocate output until we've computed the auxiliary layer and know how
  // many channels there will be.

  // Start timer.
  unsigned long start = get_cycle_count();

  // Step 1: downsample inputs.
  pool_shape_t pool_params;
  pool_params.batch_size = 1;
  pool_params.channels = input->num_channels;
  pool_params.input_width = in_size;
  pool_params.input_height = in_size;
  pool_params.window_width = in_size;
  pool_params.window_height = in_size;
  pool_params.stride = in_size;
  lat_max_pool_2d(&(input->dense), &(downsampled->dense), &pool_params);

  // Step 2: auxiliary convolution. Since we downsampled the inputs to 1x1, this
  // is equivalent to a fully-connected/linear layer.
  // First scatter the sparse data into the dense auxiliary input.
  for (int i=0; i<downsampled->num_channels; i++) {
    int channel = downsampled->channels[i];
    aux_in->data.address[channel] = downsampled->dense.data.address[i];
  }
  lat_linear(aux_in, aux_weights, aux_out, 1, in_channels, out_channels,
             &LOOP_NEST_OUTPUT_STATIONARY);

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
  output->dense.data.address = loki_malloc(out_channels_count * out_size * out_size * sizeof(int));
  output->dense.data.memory_config = mem_group_3;
  output->channels = out_channels_used;

  // Step 5: sparse convolution.
  // 'simple' mode: repeatedly apply one filter to one input channel.
  conv_shape_t shape;
  shape.batch_size = 1;
  shape.in_channels = 1;
  shape.out_channels = 1;
  shape.image_width = in_size;
  shape.image_height = in_size;
  shape.filter_width = filter_size;
  shape.filter_height = filter_size;
  shape.groups = 1;
  shape.stride = 1;
  shape.dilation = 1;

  activation_config_t conv_i;
  conv_i.data.memory_config = input->dense.data.memory_config;
  conv_i.batch_stride = input->dense.batch_stride;
  conv_i.channel_stride = input->dense.channel_stride;
  conv_i.column_stride = input->dense.column_stride;
  conv_i.row_stride = input->dense.row_stride;

  filter_config_t conv_w;
  conv_w.data.memory_config = weights->data.memory_config;
  conv_w.in_channel_stride = weights->in_channel_stride;
  conv_w.out_channel_stride = weights->out_channel_stride;
  conv_w.column_stride = weights->column_stride;
  conv_w.row_stride = weights->row_stride;
  conv_w.group_stride = weights->group_stride;

  activation_config_t conv_o;
  conv_o.data.memory_config = output->dense.data.memory_config;
  conv_o.batch_stride = output->dense.batch_stride;
  conv_o.channel_stride = output->dense.channel_stride;
  conv_o.column_stride = output->dense.column_stride;
  conv_o.row_stride = output->dense.row_stride;

  for (int o=0; o<output->num_channels; o++) {
    for (int i=0; i<input->num_channels; i++) {
      int out_c = output->channels[o];
      int in_c = input->channels[i];

      conv_i.data.address = input->dense.data.address + in_c * input->dense.channel_stride;
      conv_w.data.address = weights->data.address + in_c * weights->in_channel_stride
                                        + out_c * weights->out_channel_stride;
      conv_o.data.address = output->dense.data.address + out_c * output->dense.channel_stride;

      lat_conv2d(&conv_i, &conv_w, &conv_o, &shape, &LOOP_NEST_OUTPUT_STATIONARY);
    }
  }

  // Stop timer.
  unsigned long duration = get_cycle_count() - start;
  printf("%d/%d inputs -> %d/%d outputs took %lu cycles\n",
         in_channels_count, in_channels, out_channels_count, out_channels,
         duration);

  loki_free(downsampled->dense.data.address);
  loki_free(aux_in->data.address);
  loki_free(aux_out->data.address);
  loki_free(output->dense.data.address);
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
  input->dense.data.address = input_ptr;
  input->dense.data.memory_config = mem_group_1;
  input->channels = in_channels_used;

  sparse_activations_t* downsampled = init_sparse(1, in_channels_count, 1, 1);
  downsampled->dense.data.address = loki_malloc(in_channels_count * sizeof(int));
  downsampled->dense.data.memory_config = mem_group_cpu;
  downsampled->channels = input->channels;

  filter_config_t* weights = init_weights(in_channels, out_channels, filter_size, filter_size);
  weights->data.address = weight_ptr;
  weights->data.memory_config = mem_group_2;

  activation_config_t* aux_in = init_activations(1, in_channels, 1, 1);
  aux_in->data.address = loki_malloc(in_channels * sizeof(int));
  aux_in->data.memory_config = mem_group_cpu;

  filter_config_t* aux_weights = init_weights(in_channels, out_channels, 1, 1);
  aux_weights->data.address = aux_weight_ptr;
  aux_weights->data.memory_config = mem_group_2;

  activation_config_t* aux_out = init_activations(1, out_channels, 1, 1);
  aux_out->data.address = loki_malloc(out_channels * sizeof(int));
  aux_out->data.memory_config = mem_group_cpu;

  int out_size = in_size - filter_size + 1;
  // Can't allocate output until we've computed the auxiliary layer and know how
  // many channels there will be.

  // Start timer.
  unsigned long start = get_cycle_count();

  // Step 1: downsample inputs.
  pool_shape_t pool_params;
  pool_params.batch_size = 1;
  pool_params.channels = input->num_channels;
  pool_params.input_width = in_size;
  pool_params.input_height = in_size;
  pool_params.window_width = in_size;
  pool_params.window_height = in_size;
  pool_params.stride = in_size;
  lat_max_pool_2d(&(input->dense), &(downsampled->dense), &pool_params);

  // Step 2: auxiliary convolution. Since we downsampled the inputs to 1x1, this
  // is equivalent to a fully-connected/linear layer.
  // First scatter the sparse data into the dense auxiliary input.
  for (int i=0; i<downsampled->num_channels; i++) {
    int channel = downsampled->channels[i];
    aux_in->data.address[channel] = downsampled->dense.data.address[i];
  }
  lat_linear(aux_in, aux_weights, aux_out, 1, in_channels, out_channels,
             &LOOP_NEST_OUTPUT_STATIONARY);

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
  output->dense.data.address = loki_malloc(out_channels_count * out_size * out_size * sizeof(int));
  output->dense.data.memory_config = mem_group_3;
  output->channels = out_channels_used;

  // Step 5: sparse convolution.
  // 'adaptive' mode: look for sequences of consecutive channels available, and
  //                  apply multi-channel convolutions where possible.
  conv_shape_t shape;
  shape.batch_size = 1;
  shape.image_width = in_size;
  shape.image_height = in_size;
  shape.filter_width = filter_size;
  shape.filter_height = filter_size;
  shape.groups = 1;
  shape.stride = 1;
  shape.dilation = 1;

  activation_config_t conv_i;
  conv_i.data.memory_config = input->dense.data.memory_config;
  conv_i.batch_stride = input->dense.batch_stride;
  conv_i.channel_stride = input->dense.channel_stride;
  conv_i.column_stride = input->dense.column_stride;
  conv_i.row_stride = input->dense.row_stride;

  filter_config_t conv_w;
  conv_w.data.memory_config = weights->data.memory_config;
  conv_w.in_channel_stride = weights->in_channel_stride;
  conv_w.out_channel_stride = weights->out_channel_stride;
  conv_w.column_stride = weights->column_stride;
  conv_w.row_stride = weights->row_stride;
  conv_w.group_stride = weights->group_stride;

  activation_config_t conv_o;
  conv_o.data.memory_config = output->dense.data.memory_config;
  conv_o.batch_stride = output->dense.batch_stride;
  conv_o.channel_stride = output->dense.channel_stride;
  conv_o.column_stride = output->dense.column_stride;
  conv_o.row_stride = output->dense.row_stride;

  for (int o=0; o<output->num_channels; /*update within loop*/) {

    // Count contiguous output channels.
    shape.out_channels = 1;
    while ((o + shape.out_channels < out_channels) &&
           (output->channels[o + shape.out_channels] == output->channels[o] + shape.out_channels))
      shape.out_channels++;

    for (int i=0; i<input->num_channels; /*update within loop*/) {
      // Count contiguous input channels.
      shape.in_channels = 1;
      while ((i + shape.in_channels < in_channels) &&
             (input->channels[i + shape.in_channels] == input->channels[i] + shape.in_channels))
        shape.in_channels++;

      int out_c = output->channels[o];
      int in_c = input->channels[i];

      conv_i.data.address = input->dense.data.address + in_c * input->dense.channel_stride;
      conv_w.data.address = weights->data.address + in_c * weights->in_channel_stride
                                        + out_c * weights->out_channel_stride;
      conv_o.data.address = output->dense.data.address + out_c * output->dense.channel_stride;

      // printf("%lu x %lu mini-conv\n", shape.in_channels, shape.out_channels);
      lat_conv2d(&conv_i, &conv_w, &conv_o, &shape, &LOOP_NEST_OUTPUT_STATIONARY);
      // Potential optimisation: set up the next convolution while waiting for
      // this one to finish.

      i += shape.in_channels;
    }

    o += shape.out_channels;
  }

  // Stop timer.
  unsigned long duration = get_cycle_count() - start;
  printf("%d/%d inputs -> %d/%d outputs took %lu cycles\n",
         in_channels_count, in_channels, out_channels_count, out_channels,
         duration);

  loki_free(downsampled->dense.data.address);
  loki_free(aux_in->data.address);
  loki_free(aux_out->data.address);
  loki_free(output->dense.data.address);
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
