#include <stdlib.h>
#include <loki/channel_map_table.h>
#include <loki/control_registers.h>
#include <nn/layers.h>
#include "defs.h"

// Random data used to select output channels.
// For any section of the array, discarding values below X will give a result
// which is roughly X% sparse.
// Generated using:
// python -c "import random; print([random.randint(0,99) for _ in range(5000)])"
int random[5000] = {
#include "data.txt"
};

void test_none(const conv_shape_t* shape, void* data,
               int in_sparsity, int out_sparsity, int num_tiles) {
  dense_buffers_t* buffers = (dense_buffers_t*)data;

  // Step 1: downsample inputs. Unused.
  // Step 2: auxiliary convolution. Unused.
  // Steps 3+4: discard any features below a threshold. Unused.

  // Step 5: sparse convolution.
  // 'none' mode: convolution isn't sparse at all.
  int this_tile = tile2int(get_tile_id());
  conv_task_t tile_task = get_tile_conv_task(shape, this_tile, num_tiles);

  conv_shape_t slice = get_conv_slice(shape, &tile_task);
  activation_config_t input_slice = get_input_conv_slice(&buffers->input, &tile_task);
  filter_config_t weights_slice = get_weights_conv_slice(&buffers->weights, &tile_task);
  activation_config_t output_slice = get_output_conv_slice(&buffers->output, &tile_task);

  lat_conv2d(&input_slice, &weights_slice, &output_slice, &slice,
             &LOOP_NEST_OUTPUT_STATIONARY);

}

void test_simple(const conv_shape_t* shape, void* data,
                 int in_sparsity, int out_sparsity, int num_tiles) {
  sparse_buffers_t* buffers = (sparse_buffers_t*)data;

  // For most computations, each tile uses all inputs to compute a fraction of
  // the outputs. For downsampling, only a fraction of inputs are used.
  int this_tile = tile2int(get_tile_id());
  conv_task_t conv_task = get_tile_conv_task(shape, this_tile, num_tiles);

  conv_shape_t conv_slice = get_conv_slice(shape, &conv_task);
  sparse_activations_t input_slice = get_sparse_input_conv_slice(&buffers->input, &conv_task);
  filter_config_t weights_slice = get_weights_conv_slice(&buffers->weights, &conv_task);
  sparse_activations_t output_slice = get_sparse_output_conv_slice(&buffers->output, &conv_task);

  // Step 1: downsample inputs.
  // This pool_shape_t is for the whole layer. Later it is broken down for this
  // tile.
  pool_shape_t pool_params;
  pool_params.batch_size = shape->batch_size;
  pool_params.channels = shape->in_channels;
  pool_params.input_width = shape->image_width;
  pool_params.input_height = shape->image_height;
  pool_params.window_width = shape->image_width;
  pool_params.window_height = shape->image_height;
  pool_params.stride = 1; // irrelevant

  pool_task_t pool_task = get_tile_pool_task(&pool_params, this_tile, num_tiles);
  pool_shape_t pool_slice = get_pool_slice(&pool_params, &pool_task);
  sparse_activations_t pool_in_slice = get_sparse_input_pool_slice(&buffers->input, &pool_task);
  sparse_activations_t pool_out_slice = get_sparse_output_pool_slice(&buffers->input_downsampled, &pool_task);
  // Adjust number of channels because this computation is sparse.
  pool_slice.channels = pool_in_slice.num_channels;

  lat_max_pool_2d(&pool_in_slice.dense, &pool_out_slice.dense,
                  &pool_slice);

  // TODO Share downsampled data with other tiles - needed for aux computation.

  // Step 2: auxiliary convolution. Since we downsampled the inputs to 1x1, this
  // is equivalent to a fully-connected/linear layer.
  // First scatter the sparse data into the dense auxiliary input.
  for (int i=0; i<pool_out_slice.num_channels; i++) {
    int channel = pool_out_slice.channels[i];
    buffers->auxiliary->input.data.address[channel] =
        pool_out_slice.dense.data.address[i];
  }

  activation_config_t aux_in_slice = get_input_conv_slice(&buffers->auxiliary->input, &conv_task);
  filter_config_t aux_weights_slice = get_weights_conv_slice(&buffers->auxiliary->weights, &conv_task);
  activation_config_t aux_out_slice = get_output_conv_slice(&buffers->auxiliary->output, &conv_task);

  lat_linear(&aux_in_slice, &aux_weights_slice, &aux_out_slice,
             conv_slice.batch_size, conv_slice.in_channels, conv_slice.out_channels,
             &LOOP_NEST_OUTPUT_STATIONARY);

  // Steps 3+4: discard any features below a threshold.
  // In order to have more control over the sparsity achieved, a predetermined
  // random sequence is used for this, instead of the output of step 2.
  int out_channels_count = 0;
  int first_out_channel = this_tile * conv_slice.out_channels;
  for (int i=first_out_channel; i<first_out_channel+conv_slice.out_channels; i++)
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

  // i and o iterate through only the channels which have been computed.
  for (int o=0; o<output_slice.num_channels; o++) {
    for (int i=0; i<input_slice.num_channels; i++) {
      // in_c and out_c iterate through all channels (including uncomputed).
      int out_c = output_slice.channels[o];
      int in_c = input_slice.channels[i];

      activation_config_t conv_i = activation_slice(&input_slice.dense, i, i+1);
      filter_config_t conv_w = weight_slice(&weights_slice, in_c, in_c+1, out_c, out_c+1);
      activation_config_t conv_o = activation_slice(&output_slice.dense, o, o+1);

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
  conv_task_t conv_task = get_tile_conv_task(shape, this_tile, num_tiles);

  conv_shape_t conv_slice = get_conv_slice(shape, &conv_task);
  sparse_activations_t input_slice = get_sparse_input_conv_slice(&buffers->input, &conv_task);
  filter_config_t weights_slice = get_weights_conv_slice(&buffers->weights, &conv_task);
  sparse_activations_t output_slice = get_sparse_output_conv_slice(&buffers->output, &conv_task);

  // Step 1: downsample inputs.
  // This pool_shape_t is for the whole layer. Later it is broken down for this
  // tile.
  pool_shape_t pool_params;
  pool_params.batch_size = shape->batch_size;
  pool_params.channels = shape->in_channels;
  pool_params.input_width = shape->image_width;
  pool_params.input_height = shape->image_height;
  pool_params.window_width = shape->image_width;
  pool_params.window_height = shape->image_height;
  pool_params.stride = 1; // irrelevant

  pool_task_t pool_task = get_tile_pool_task(&pool_params, this_tile, num_tiles);
  pool_shape_t pool_slice = get_pool_slice(&pool_params, &pool_task);
  sparse_activations_t pool_in_slice = get_sparse_input_pool_slice(&buffers->input, &pool_task);
  sparse_activations_t pool_out_slice = get_sparse_output_pool_slice(&buffers->input_downsampled, &pool_task);
  // Adjust number of channels because this computation is sparse.
  pool_slice.channels = pool_in_slice.num_channels;

  lat_max_pool_2d(&pool_in_slice.dense, &pool_out_slice.dense,
                  &pool_slice);

  // TODO Share downsampled data with other tiles - needed for aux computation.

  // Step 2: auxiliary convolution. Since we downsampled the inputs to 1x1, this
  // is equivalent to a fully-connected/linear layer.
  // First scatter the sparse data into the dense auxiliary input.
  for (int i=0; i<pool_out_slice.num_channels; i++) {
    int channel = pool_out_slice.channels[i];
    buffers->auxiliary->input.data.address[channel] =
        pool_out_slice.dense.data.address[i];
  }

  activation_config_t aux_in_slice = get_input_conv_slice(&buffers->auxiliary->input, &conv_task);
  filter_config_t aux_weights_slice = get_weights_conv_slice(&buffers->auxiliary->weights, &conv_task);
  activation_config_t aux_out_slice = get_output_conv_slice(&buffers->auxiliary->output, &conv_task);

  lat_linear(&aux_in_slice, &aux_weights_slice, &aux_out_slice,
             conv_slice.batch_size, conv_slice.in_channels, conv_slice.out_channels,
             &LOOP_NEST_OUTPUT_STATIONARY);

  // Steps 3+4: discard any features below a threshold.
  // In order to have more control over the sparsity achieved, a predetermined
  // random sequence is used for this, instead of the output of step 2.
  int out_channels_count = 0;
  int first_out_channel = this_tile * conv_slice.out_channels;
  for (int i=first_out_channel; i<first_out_channel+conv_slice.out_channels; i++)
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

  // i and o iterate through only the channels which have been computed.
  for (int o=0; o<output_slice.num_channels; /*update within loop*/) {

    // Count contiguous output channels.
    task.out_channels = 1;
    while ((o + task.out_channels < first_out_channel+conv_slice.out_channels) &&
           (output_slice.channels[o + task.out_channels] == output_slice.channels[o] + task.out_channels))
      task.out_channels++;

    for (int i=0; i<input_slice.num_channels; /*update within loop*/) {
      // Count contiguous input channels. Could precompute this instead of doing
      // it every iteration?
      task.in_channels = 1;
      while ((i + task.in_channels < shape->in_channels) &&
             (input_slice.channels[i + task.in_channels] == input_slice.channels[i] + task.in_channels))
        task.in_channels++;

      // in_c and out_c iterate through all channels (including uncomputed).
      int out_c = output_slice.channels[o];
      int in_c = input_slice.channels[i];

      activation_config_t conv_i = activation_slice(&input_slice.dense, i, i+task.in_channels);
      filter_config_t conv_w = weight_slice(&weights_slice, in_c, in_c+task.in_channels, out_c, out_c+task.out_channels);
      activation_config_t conv_o = activation_slice(&output_slice.dense, o, o+task.out_channels);

      // printf("%lu x %lu mini-conv\n", shape.in_channels, shape.out_channels);
      lat_conv2d(&conv_i, &conv_w, &conv_o, &task, &LOOP_NEST_OUTPUT_STATIONARY);
      // Potential optimisation: set up the next convolution while waiting for
      // this one to finish.

      i += task.in_channels;
    }

    o += task.out_channels;
  }

}
