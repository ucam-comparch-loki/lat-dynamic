#include <loki/alloc.h>
#include "defs.h"

// An array of random numbers.
extern int* random;

// Create an activation tensor. Allocation of data and assignment to a memory
// group is not done. (User must set `address` and `data.memory_config`.)
// Dimension order is BCHW.
void init_activations(activation_config_t* a,
                      int batch_size, int channels, int height, int width) {
  a->row_stride = sizeof(data_t);
  a->column_stride = width * a->row_stride;
  a->channel_stride = height * a->column_stride;
  a->batch_stride = channels * a->channel_stride;
}

// Create a weight tensor. Allocation of data and assignment to a memory
// group is not done. (User must set `address` and `data.memory_config`.)
// Dimension order is OIHW.
void init_weights(filter_config_t* f, int in_channels, int out_channels,
                  int filter_height, int filter_width) {
  f->row_stride = sizeof(data_t);
  f->column_stride = filter_width * f->row_stride;
  f->out_channel_stride = filter_height * f->column_stride;
  f->in_channel_stride = out_channels * f->out_channel_stride;
}

// Specialisation of activation_config_t for sparse activations.
// User must also set `channels` array.
void init_sparse(sparse_activations_t* a,
                 int batch_size, int channels, int height, int width) {
  a->dense.row_stride = sizeof(data_t);
  a->dense.column_stride = width * a->dense.row_stride;
  a->dense.channel_stride = height * a->dense.column_stride;
  a->dense.batch_stride = channels * a->dense.channel_stride;
  a->num_channels = channels;
}


void* init_dense_buffers(const conv_shape_t* shape) {
  // Create some memory groups, allowing data to be physically partitioned.
  // The CPU group should be used wherever an array is accessed in C code.
  channel_t mem_group_cpu = get_channel_map(1);
  channel_t mem_group_1 = mem_group_cpu; // TODO: use something specific
  channel_t mem_group_2 = mem_group_cpu; // TODO: use something specific
  channel_t mem_group_3 = mem_group_cpu; // TODO: use something specific

  dense_buffers_t* data = loki_malloc(sizeof(dense_buffers_t));

  // Use uninitialised data for weights and activations.
  // This will not affect the result unless fine-grained sparsity is exploited,
  // or data is compressed.
  data_t* input_ptr = loki_malloc(shape->in_channels * shape->image_width *
                                  shape->image_height * sizeof(data_t));
  data_t* weight_ptr = loki_malloc(shape->in_channels * shape->out_channels *
                                   shape->filter_width * shape->filter_height * sizeof(data_t));

  // Assuming square input/output.
  int out_size = shape->image_width - shape->filter_width + 1;
  data_t* output_ptr = loki_malloc(shape->out_channels * out_size * out_size *
                                   sizeof(data_t));
  assert(input_ptr != NULL);
  assert(weight_ptr != NULL);
  assert(output_ptr != NULL);

  // Create all necessary data buffers.
  init_activations(&(data->input), shape->batch_size, shape->in_channels, shape->image_height, shape->image_width);
  data->input.data.address = input_ptr;
  data->input.data.memory_config = mem_group_1;

  init_weights(&(data->weights), shape->in_channels, shape->out_channels, shape->filter_height, shape->filter_width);
  data->weights.data.address = weight_ptr;
  data->weights.data.memory_config = mem_group_2;

  init_activations(&(data->output), shape->batch_size, shape->out_channels, out_size, out_size);
  data->output.data.address = output_ptr;
  data->output.data.memory_config = mem_group_3;

  // Flush all data that might be needed by other tiles.
  // Don't need to flush the data arrays themselves because we haven't modified
  // them.
  loki_channel_flush_data(1, data, sizeof(dense_buffers_t));

  return data;
}

void delete_dense_buffers(void* data) {
  dense_buffers_t* d = (dense_buffers_t*) data;

  loki_free(d->input.data.address);
  loki_free(d->weights.data.address);
  loki_free(d->output.data.address);
  loki_free(d);
}

void* init_sparse_buffers(const conv_shape_t* shape, int in_sparsity) {
  // A pre-allocated array of random numbers is used to choose which channels to
  // skip over. (Generating random numbers is expensive to simulate.)
  assert(shape->in_channels + shape->out_channels < 5000);

  sparse_buffers_t* data = loki_malloc(sizeof(sparse_buffers_t));

  // Use uninitialised data for weights and activations.
  // This will not affect the result unless fine-grained sparsity is exploited,
  // or data is compressed.
  data_t* input_ptr = loki_malloc(shape->in_channels * shape->image_width *
                                  shape->image_height * sizeof(data_t));
  data_t* weight_ptr = loki_malloc(shape->in_channels * shape->out_channels *
                                   shape->filter_width * shape->filter_height * sizeof(data_t));

  // Simple but inefficient approach: statically allocate maximum possible
  // buffer size. Assuming square input/output.
  int out_size = shape->image_width - shape->filter_width + 1;
  data_t* output_ptr = loki_malloc(shape->out_channels * out_size * out_size *
                                   sizeof(data_t));
  assert(input_ptr != NULL);
  assert(weight_ptr != NULL);
  assert(output_ptr != NULL);

  // Determine how many input channels to use, given the sparsity.
  // (In practice, this would be done by the previous layer, but we're only
  // simulating one layer at a time.)
  int in_channels_count = 0;
  int* in_channels_used = loki_malloc(shape->in_channels * sizeof(int));
  for (int i=0; i<shape->in_channels; i++)
    if (random[i] > in_sparsity)
      in_channels_used[in_channels_count++] = i;

  init_sparse(&(data->input), shape->batch_size, in_channels_count, shape->image_height, shape->image_width);
  data->input.dense.data.address = input_ptr;
  data->input.channels = in_channels_used;

  init_sparse(&(data->input_downsampled), shape->batch_size, in_channels_count, 1, 1);
  data_t* downsampled_ptr = loki_malloc(in_channels_count * sizeof(data_t));
  assert(downsampled_ptr != NULL);
  data->input_downsampled.dense.data.address = downsampled_ptr;
  data->input_downsampled.channels = data->input.channels;

  // The auxiliary computation is dense and independent of the data.
  // TODO: use a linear layer when available.
  conv_shape_t aux = {
    .batch_size = shape->batch_size, .in_channels = shape->in_channels,
    .out_channels = shape->out_channels, .image_width = 1, .image_height = 1,
    .filter_width = 1, .filter_height = 1, .groups = 1, .stride = 1,
    .dilation = 1
  };
  data->auxiliary = init_dense_buffers(&aux);

  init_weights(&(data->weights), shape->in_channels, shape->out_channels, shape->filter_height, shape->filter_width);
  data->weights.data.address = weight_ptr;

  init_sparse(&(data->output), shape->batch_size, shape->out_channels, out_size, out_size);
  data->output.dense.data.address = output_ptr;
  data->output.channels = loki_malloc(shape->out_channels * sizeof(int));
  assert(data->output.channels != NULL);

  // Memory management.
  channel_t mem_group_cpu = get_channel_map(1);
  channel_t mem_group_1 = mem_group_cpu; // TODO: use something specific
  channel_t mem_group_2 = mem_group_cpu; // TODO: use something specific
  channel_t mem_group_3 = mem_group_cpu; // TODO: use something specific

  // These are transferred between the CPU and accelerator, so use the default
  // CPU memory group. Otherwise, try to use different memory groups for tensors
  // used at the same time to avoid conflicts between them.
  data->input_downsampled.dense.data.memory_config = mem_group_cpu;
  data->auxiliary->input.data.memory_config = mem_group_cpu;
  data->auxiliary->output.data.memory_config = mem_group_cpu;

  data->auxiliary->weights.data.memory_config = mem_group_2;

  data->input.dense.data.memory_config = mem_group_1;
  data->weights.data.memory_config = mem_group_2;
  data->output.dense.data.memory_config = mem_group_3;

  // Flush all data that might be needed by other tiles.
  // Don't need to flush the data arrays themselves because we haven't modified
  // them.
  loki_channel_flush_data(1, data, sizeof(sparse_buffers_t));

  return data;
}

void delete_sparse_buffers(void* data) {
  sparse_buffers_t* d = (sparse_buffers_t*)data;

  loki_free(d->input.dense.data.address);
  loki_free(d->weights.data.address);
  loki_free(d->output.dense.data.address);
  loki_free(d->input_downsampled.dense.data.address);

  loki_free(d->input.channels);
  loki_free(d->output.channels);

  delete_dense_buffers(d->auxiliary);

  loki_free(d);
}
