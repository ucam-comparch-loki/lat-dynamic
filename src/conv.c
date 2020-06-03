#include <stdlib.h>
#include <loki/channel_map_table.h>
#include <loki/control_registers.h>
#include <nn/layers.h>
#include "defs.h"

#define LOAD_BALANCE

// Random data used to select output channels.
// For any section of the array, discarding values below X will give a result
// which is roughly X% sparse.
// Generated using:
// python -c "import random; print([random.randint(0,99) for _ in range(5000)])"
int random[5000] = {
#include "data.txt"
};

// Some optimised loop orders for the specific computations we're doing.
// To be used with:
// lokisim --accelerator-accumulate-rows=0 --accelerator-accumulate-columns=1

// Parallelise across channels.
enum Loop LOOPS_MANY_CHANNELS[6] = {FILTER_HEIGHT_OS, FILTER_WIDTH_OS,
  IMAGE_HEIGHT, IMAGE_WIDTH, OUT_CHANNELS, IN_CHANNELS};
loop_nest_t LOOP_NEST_MANY_CHANNELS = {
  .loop_count = 6,
  .loops = LOOPS_MANY_CHANNELS
};

// Can't do the above: parallelise within a single channel.
// The image is larger than the filter, so I would prefer to parallelise those
// loops, but I think there's a lokisim bug which prevents that.
enum Loop LOOPS_FEW_CHANNELS[6] = {OUT_CHANNELS, IN_CHANNELS,
  IMAGE_HEIGHT, IMAGE_WIDTH, FILTER_HEIGHT_IS, FILTER_WIDTH_OS};
loop_nest_t LOOP_NEST_FEW_CHANNELS = {
  .loop_count = 6,
  .loops = LOOPS_FEW_CHANNELS
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
             &LOOP_NEST_MANY_CHANNELS);

}

void test_simple(const conv_shape_t* shape, void* data,
                 int in_sparsity, int out_sparsity, int num_tiles) {
  sparse_buffers_t* buffers = (sparse_buffers_t*)data;

  // For most computations, each tile uses all inputs to compute a fraction of
  // the outputs. For downsampling, only a fraction of inputs are used.
  int this_tile = tile2int(get_tile_id());
  conv_task_t conv_task = get_tile_conv_task(shape, this_tile, num_tiles);

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

  conv_shape_t conv_slice = get_conv_slice(shape, &conv_task);
  lat_linear(&aux_in_slice, &aux_weights_slice, &aux_out_slice,
             conv_slice.batch_size, conv_slice.in_channels, conv_slice.out_channels,
             &LOOP_NEST_MANY_CHANNELS);

  // Steps 3+4: discard any features below a threshold.
  // In order to have more control over the sparsity achieved, a predetermined
  // random sequence is used for this, instead of the output of step 2.
  int out_channels_count = 0;
  int first_out_channel = this_tile * conv_slice.out_channels;
  for (int i=first_out_channel; i<first_out_channel+conv_slice.out_channels; i++)
    if (random[5000 - i] > out_sparsity)
      buffers->output.channels[first_out_channel + out_channels_count++] = i;
  buffers->output.num_channels = out_channels_count; // NEEDS SYNC

  // TODO: share with other tiles how many channels will be computed. Needed to
  // know where in the total output our slice will go.

  // This tile's initial work allocation for the sparse convolution.
  // This task may be modified as computation progresses, as work is
  // redistributed among the parallel tiles.
  conv_task_t task;
  task.first_in_channel = 0;
  task.last_in_channel = buffers->input.num_channels;
  task.first_out_channel = out_channels_count * this_tile; // FAKE
  task.last_out_channel = task.first_out_channel + out_channels_count;

  // Step 5: sparse convolution.
  // 'simple' mode: repeatedly apply one filter to one input channel.
  conv_shape_t unit;
  unit.batch_size = 1;
  unit.in_channels = 1;
  unit.out_channels = 1;
  unit.image_width = shape->image_width;
  unit.image_height = shape->image_height;
  unit.filter_width = shape->filter_width;
  unit.filter_height = shape->filter_height;
  unit.groups = 1;
  unit.stride = 1;
  unit.dilation = 1;

#ifdef LOAD_BALANCE
  // TODO: Make load balancing optional.
  lb_state_t load_balance;
  init_lb_state(&load_balance, num_tiles);

  while (!lb_finished(&load_balance)) {
#endif

    // i and o iterate through only the channels which have been computed.
    for (int o=task.first_out_channel; o<task.last_out_channel; o++) {
      for (int i=task.first_in_channel; i<task.last_in_channel; i++) {
        // in_c and out_c iterate through all channels (including uncomputed).
        int out_c = buffers->output.channels[o];
        int in_c = buffers->input.channels[i];

        activation_config_t conv_i = activation_slice(&buffers->input.dense, i, i+1);
        filter_config_t conv_w = weight_slice(&buffers->weights, in_c, in_c+1, out_c, out_c+1);
        activation_config_t conv_o = activation_slice(&buffers->output.dense, o, o+1);

        lat_conv2d(&conv_i, &conv_w, &conv_o, &unit, &LOOP_NEST_FEW_CHANNELS);

#ifdef LOAD_BALANCE
        // Give up any spare work, if requested.
        // TODO: Do this while waiting on the accelerator.
        check_load_balance_requests(&task, &load_balance, i+1, o);
#endif
      }
    }

#ifdef LOAD_BALANCE
    // Request new work from a neighbouring tile. This will update `task`.
    make_load_balance_request(&task, &load_balance, num_tiles);

  }

  // Ensure we respond to any outstanding requests.
  lb_sync(&load_balance);
#endif

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

  conv_shape_t conv_slice = get_conv_slice(shape, &conv_task);
  lat_linear(&aux_in_slice, &aux_weights_slice, &aux_out_slice,
             conv_slice.batch_size, conv_slice.in_channels, conv_slice.out_channels,
             &LOOP_NEST_MANY_CHANNELS);

  // Steps 3+4: discard any features below a threshold.
  // In order to have more control over the sparsity achieved, a predetermined
  // random sequence is used for this, instead of the output of step 2.
  int out_channels_count = 0;
  int first_out_channel = this_tile * conv_slice.out_channels;
  for (int i=first_out_channel; i<first_out_channel+conv_slice.out_channels; i++)
    if (random[5000 - i] > out_sparsity)
      buffers->output.channels[first_out_channel + out_channels_count++] = i;
  buffers->output.num_channels = out_channels_count; // NEEDS SYNC

  // TODO: share with other tiles how many channels will be computed. Needed to
  // know where in the total output our slice will go.

  // This tile's initial work allocation for the sparse convolution.
  // This task may be modified as computation progresses, as work is
  // redistributed among the parallel tiles.
  conv_task_t task;
  task.first_in_channel = 0;
  task.last_in_channel = buffers->input.num_channels;
  task.first_out_channel = out_channels_count * this_tile; // FAKE
  task.last_out_channel = task.first_out_channel + out_channels_count;

  // Step 5: sparse convolution.
  // 'adaptive' mode: look for sequences of consecutive channels available, and
  //                  apply multi-channel convolutions where possible.
  conv_shape_t unit;
  unit.batch_size = 1;
  unit.image_width = shape->image_width;
  unit.image_height = shape->image_height;
  unit.filter_width = shape->filter_width;
  unit.filter_height = shape->filter_height;
  unit.groups = 1;
  unit.stride = 1;
  unit.dilation = 1;

#ifdef LOAD_BALANCE
  // TODO: Make load balancing optional.
  lb_state_t load_balance;
  init_lb_state(&load_balance, num_tiles);

  while (!lb_finished(&load_balance)) {
#endif

    // i and o iterate through only the channels which have been computed.
    for (int o=task.first_out_channel; o<task.last_out_channel; /*update within loop*/) {

      // Count contiguous output channels.
      unit.out_channels = 1;
      while ((o + unit.out_channels < task.last_out_channel) &&
             (buffers->output.channels[o + unit.out_channels] == buffers->output.channels[o] + unit.out_channels))
        unit.out_channels++;

      for (int i=task.first_in_channel; i<task.last_in_channel; /*update within loop*/) {
        // Count contiguous input channels. Could precompute this instead of doing
        // it every iteration?
        unit.in_channels = 1;
        while ((i + unit.in_channels < task.last_in_channel) &&
               (buffers->input.channels[i + unit.in_channels] == buffers->input.channels[i] + unit.in_channels))
          unit.in_channels++;

        // in_c and out_c iterate through all channels (including uncomputed).
        int out_c = buffers->output.channels[o];
        int in_c = buffers->input.channels[i];

        activation_config_t conv_i = activation_slice(&buffers->input.dense, i, i+unit.in_channels);
        filter_config_t conv_w = weight_slice(&buffers->weights, in_c, in_c+unit.in_channels, out_c, out_c+unit.out_channels);
        activation_config_t conv_o = activation_slice(&buffers->output.dense, o, o+unit.out_channels);

        // printf("%lu x %lu mini-conv\n", shape.in_channels, shape.out_channels);
        lat_conv2d(&conv_i, &conv_w, &conv_o, &unit, &LOOP_NEST_FEW_CHANNELS);
        // Potential optimisation: set up the next convolution while waiting for
        // this one to finish.

        i += unit.in_channels;

#ifdef LOAD_BALANCE
        // Give up any spare work, if requested.
        // TODO: Do this while waiting on the accelerator.
        check_load_balance_requests(&task, &load_balance, i, o);
#endif
      }

      o += unit.out_channels;
    }

#ifdef LOAD_BALANCE
    // Request new work from a neighbouring tile. This will update `task`.
    make_load_balance_request(&task, &load_balance, num_tiles);

  }

  // Ensure we respond to any outstanding requests.
  lb_sync(&load_balance);
#endif

}
