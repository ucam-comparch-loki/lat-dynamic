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

// All data buffers required for a dense computation.
typedef struct {
  activation_config_t input;
  filter_config_t weights;
  activation_config_t output;
} dense_buffers_t;

// All data buffers required for a sparse computation.
typedef struct {
  sparse_activations_t input;
  filter_config_t weights;
  sparse_activations_t output;

  sparse_activations_t input_downsampled;
  dense_buffers_t* auxiliary;
} sparse_buffers_t;

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
  // CPU memory group.
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

void test_none(const conv_shape_t* shape, void* data,
               int in_sparsity, int out_sparsity, int num_tiles) {
  dense_buffers_t* buffers = (dense_buffers_t*)data;

  // Step 1: downsample inputs. Unused.
  // Step 2: auxiliary convolution. Unused.
  // Steps 3+4: discard any features below a threshold. Unused.

  // Step 5: sparse convolution.
  // 'none' mode: convolution isn't sparse at all.
  // Each tile computes a fraction of the output channels.
  int this_tile = tile2int(get_tile_id());
  int channels_per_tile = shape->out_channels / num_tiles;
  int first_channel = this_tile * channels_per_tile;

  conv_shape_t task = *shape;
  task.out_channels = channels_per_tile;

  filter_config_t weights_slice = buffers->weights;
  weights_slice.data.address += first_channel * weights_slice.out_channel_stride / sizeof(data_t);

  activation_config_t output_slice = buffers->output;
  output_slice.data.address += first_channel * output_slice.channel_stride / sizeof(data_t);

  lat_conv2d(&buffers->input, &weights_slice, &output_slice, &task,
             &LOOP_NEST_OUTPUT_STATIONARY);

}

void test_simple(const conv_shape_t* shape, void* data,
                 int in_sparsity, int out_sparsity, int num_tiles) {
  sparse_buffers_t* buffers = (sparse_buffers_t*)data;

  // For most computations, each tile uses all inputs to compute a fraction of
  // the outputs. For downsampling, only a fraction of inputs are used.
  int this_tile = tile2int(get_tile_id());
  int in_channels_per_tile = shape->in_channels / num_tiles;
  int first_in_channel = this_tile * in_channels_per_tile;
  int out_channels_per_tile = shape->out_channels / num_tiles;
  int first_out_channel = this_tile * out_channels_per_tile;

  int first_sparse_input = buffers->input.num_channels;
  int num_sparse_inputs = 0;
  for (int i=0; i<buffers->input.num_channels; i++) {
    if (buffers->input.channels[i] >= first_in_channel) {
      first_sparse_input = i;
      break;
    }
  }
  for (int i=first_sparse_input; i<buffers->input.num_channels; i++)
    if (buffers->input.channels[i] < first_in_channel+in_channels_per_tile)
      num_sparse_inputs++;
    else
      break;

  sparse_activations_t input_slice = buffers->input;
  input_slice.dense.data.address += input_slice.dense.channel_stride * first_sparse_input / sizeof(data_t);
  input_slice.channels += first_sparse_input;
  input_slice.num_channels = num_sparse_inputs;
  sparse_activations_t downsampled_slice = buffers->input_downsampled;
  downsampled_slice.dense.data.address += downsampled_slice.dense.channel_stride * first_sparse_input / sizeof(data_t);
  downsampled_slice.channels += first_sparse_input;
  downsampled_slice.num_channels = num_sparse_inputs;
  filter_config_t aux_weights_slice = buffers->auxiliary->weights;
  aux_weights_slice.data.address += aux_weights_slice.in_channel_stride * first_in_channel / sizeof(data_t)
                                  + aux_weights_slice.out_channel_stride * first_out_channel / sizeof(data_t);
  activation_config_t aux_out_slice = buffers->auxiliary->output;
  aux_out_slice.data.address += aux_out_slice.channel_stride * first_out_channel / sizeof(data_t);
  sparse_activations_t output_slice = buffers->output;

  // Step 1: downsample inputs.
  pool_shape_t pool_params;
  pool_params.batch_size = shape->batch_size;
  pool_params.channels = input_slice.num_channels;
  pool_params.input_width = shape->image_width;
  pool_params.input_height = shape->image_height;
  pool_params.window_width = shape->image_width;
  pool_params.window_height = shape->image_height;
  pool_params.stride = 1; // irrelevant
  lat_max_pool_2d(&(input_slice.dense), &(downsampled_slice.dense),
                  &pool_params);

  // TODO Share downsampled data with other tiles - needed for aux computation.

  // Step 2: auxiliary convolution. Since we downsampled the inputs to 1x1, this
  // is equivalent to a fully-connected/linear layer.
  // First scatter the sparse data into the dense auxiliary input.
  for (int i=0; i<downsampled_slice.num_channels; i++) {
    int channel = downsampled_slice.channels[i];
    buffers->auxiliary->input.data.address[channel] =
        downsampled_slice.dense.data.address[i];
  }
  lat_linear(&buffers->auxiliary->input, &aux_weights_slice, &aux_out_slice,
             shape->batch_size, shape->in_channels, out_channels_per_tile,
             &LOOP_NEST_OUTPUT_STATIONARY);

  // Steps 3+4: discard any features below a threshold.
  // In order to have more control over the sparsity achieved, a predetermined
  // random sequence is used for this, instead of the output of step 2.
  int out_channels_count = 0;
  for (int i=first_out_channel; i<first_out_channel+out_channels_per_tile; i++)
    if (random[5000 - i] > out_sparsity)
      output_slice.channels[out_channels_count++] = i;
  output_slice.num_channels = out_channels_count;

  // TODO: share with other tiles which channels will be computed. Needed to
  // know where in the total output our slice will go.

  // Step 5: sparse convolution.
  // 'simple' mode: repeatedly apply one filter to one input channel.
  conv_shape_t task;
  task.batch_size = 1;
  task.in_channels = 1;
  task.out_channels = 1;
  task.image_width = shape->image_width;
  task.image_height = shape->image_height;
  task.filter_width = shape->filter_width;
  task.filter_height = shape->filter_height;
  task.groups = 1;
  task.stride = 1;
  task.dilation = 1;

  activation_config_t conv_i = buffers->input.dense;
  filter_config_t conv_w = buffers->weights;
  activation_config_t conv_o = output_slice.dense;

  for (int o=0; o<output_slice.num_channels; o++) {
    for (int i=0; i<buffers->input.num_channels; i++) {
      int out_c = output_slice.channels[o];
      int in_c = buffers->input.channels[i];

      conv_i.data.address = buffers->input.dense.data.address
                          + in_c * buffers->input.dense.channel_stride / sizeof(data_t);
      conv_w.data.address = buffers->weights.data.address
                          + in_c * buffers->weights.in_channel_stride / sizeof(data_t)
                          + out_c * buffers->weights.out_channel_stride / sizeof(data_t);
      conv_o.data.address = output_slice.dense.data.address
                          + out_c * output_slice.dense.channel_stride / sizeof(data_t);

      lat_conv2d(&conv_i, &conv_w, &conv_o, &task, &LOOP_NEST_OUTPUT_STATIONARY);
    }
  }

}

// TODO: reduce code duplication.
// This is identical to test_simple except the for loops in step 5.
void test_adaptive(const conv_shape_t* shape, void* data,
                   int in_sparsity, int out_sparsity, int num_tiles) {
  sparse_buffers_t* buffers = (sparse_buffers_t*)data;

  // For most computations, each tile uses all inputs to compute a fraction of
  // the outputs. For downsampling, only a fraction of inputs are used.
  int this_tile = tile2int(get_tile_id());
  int in_channels_per_tile = shape->in_channels / num_tiles;
  int first_in_channel = this_tile * in_channels_per_tile;
  int out_channels_per_tile = shape->out_channels / num_tiles;
  int first_out_channel = this_tile * out_channels_per_tile;

  int first_sparse_input = buffers->input.num_channels;
  int num_sparse_inputs = 0;
  for (int i=0; i<buffers->input.num_channels; i++) {
    if (buffers->input.channels[i] >= first_in_channel) {
      first_sparse_input = i;
      break;
    }
  }
  for (int i=first_sparse_input; i<buffers->input.num_channels; i++)
    if (buffers->input.channels[i] < first_in_channel+in_channels_per_tile)
      num_sparse_inputs++;
    else
      break;

  sparse_activations_t input_slice = buffers->input;
  input_slice.dense.data.address += input_slice.dense.channel_stride * first_sparse_input / sizeof(data_t);
  input_slice.channels += first_sparse_input;
  input_slice.num_channels = num_sparse_inputs;
  sparse_activations_t downsampled_slice = buffers->input_downsampled;
  downsampled_slice.dense.data.address += downsampled_slice.dense.channel_stride * first_sparse_input / sizeof(data_t);
  downsampled_slice.channels += first_sparse_input;
  downsampled_slice.num_channels = num_sparse_inputs;
  filter_config_t aux_weights_slice = buffers->auxiliary->weights;
  aux_weights_slice.data.address += aux_weights_slice.in_channel_stride * first_in_channel / sizeof(data_t)
                                  + aux_weights_slice.out_channel_stride * first_out_channel / sizeof(data_t);
  activation_config_t aux_out_slice = buffers->auxiliary->output;
  aux_out_slice.data.address += aux_out_slice.channel_stride * first_out_channel / sizeof(data_t);
  sparse_activations_t output_slice = buffers->output;

  // Step 1: downsample inputs.
  pool_shape_t pool_params;
  pool_params.batch_size = shape->batch_size;
  pool_params.channels = input_slice.num_channels;
  pool_params.input_width = shape->image_width;
  pool_params.input_height = shape->image_height;
  pool_params.window_width = shape->image_width;
  pool_params.window_height = shape->image_height;
  pool_params.stride = 1; // irrelevant
  lat_max_pool_2d(&(input_slice.dense), &(downsampled_slice.dense),
                  &pool_params);

  // TODO Share downsampled data with other tiles - needed for aux computation.

  // Step 2: auxiliary convolution. Since we downsampled the inputs to 1x1, this
  // is equivalent to a fully-connected/linear layer.
  // First scatter the sparse data into the dense auxiliary input.
  for (int i=0; i<downsampled_slice.num_channels; i++) {
    int channel = downsampled_slice.channels[i];
    buffers->auxiliary->input.data.address[channel] =
        downsampled_slice.dense.data.address[i];
  }
  lat_linear(&buffers->auxiliary->input, &aux_weights_slice, &aux_out_slice,
             shape->batch_size, shape->in_channels, out_channels_per_tile,
             &LOOP_NEST_OUTPUT_STATIONARY);

  // Steps 3+4: discard any features below a threshold.
  // In order to have more control over the sparsity achieved, a predetermined
  // random sequence is used for this, instead of the output of step 2.
  int out_channels_count = 0;
  for (int i=first_out_channel; i<first_out_channel+out_channels_per_tile; i++)
    if (random[5000 - i] > out_sparsity)
      output_slice.channels[out_channels_count++] = i;
  output_slice.num_channels = out_channels_count;

  // TODO: share with other tiles which channels will be computed. Needed to
  // know where in the total output our slice will go.

  // Step 5: sparse convolution.
  // 'simple' mode: repeatedly apply one filter to one input channel.
  conv_shape_t task;
  task.batch_size = 1;
  task.image_width = shape->image_width;
  task.image_height = shape->image_height;
  task.filter_width = shape->filter_width;
  task.filter_height = shape->filter_height;
  task.groups = 1;
  task.stride = 1;
  task.dilation = 1;

  activation_config_t conv_i = buffers->input.dense;
  filter_config_t conv_w = buffers->weights;
  activation_config_t conv_o = output_slice.dense;

  for (int o=0; o<output_slice.num_channels; /*update within loop*/) {

    // Count contiguous output channels.
    task.out_channels = 1;
    while ((o + task.out_channels < first_out_channel+out_channels_per_tile) &&
           (output_slice.channels[o + task.out_channels] == output_slice.channels[o] + task.out_channels))
      task.out_channels++;

    for (int i=0; i<buffers->input.num_channels; /*update within loop*/) {
      // Count contiguous input channels. Could precompute this instead of doing
      // it every iteration?
      task.in_channels = 1;
      while ((i + task.in_channels < shape->in_channels) &&
             (buffers->input.channels[i + task.in_channels] == buffers->input.channels[i] + task.in_channels))
        task.in_channels++;

      int out_c = output_slice.channels[o];
      int in_c = buffers->input.channels[i];

      conv_i.data.address = buffers->input.dense.data.address
                          + in_c * buffers->input.dense.channel_stride / sizeof(data_t);
      conv_w.data.address = buffers->weights.data.address
                          + in_c * buffers->weights.in_channel_stride / sizeof(data_t)
                          + out_c * buffers->weights.out_channel_stride / sizeof(data_t);
      conv_o.data.address = output_slice.dense.data.address
                          + out_c * output_slice.dense.channel_stride / sizeof(data_t);

      // printf("%lu x %lu mini-conv\n", shape.in_channels, shape.out_channels);
      lat_conv2d(&conv_i, &conv_w, &conv_o, &task, &LOOP_NEST_OUTPUT_STATIONARY);
      // Potential optimisation: set up the next convolution while waiting for
      // this one to finish.

      i += task.in_channels;
    }

    o += task.out_channels;
  }

}
