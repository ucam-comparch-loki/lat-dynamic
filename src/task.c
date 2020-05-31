#include <nn/layers.h>
#include "defs.h"

// Split a layer up across tiles. This is just an initial split, and can be
// renegotiated later.
conv_task_t get_tile_conv_task(const conv_shape_t* shape, int tile, int num_tiles) {
  conv_task_t task;

  // Each tile uses all input channels to compute a subset of output channels.

  task.first_in_channel = 0;
  task.last_in_channel = shape->in_channels;

  int out_channels_per_tile = shape->out_channels / num_tiles;
  task.first_out_channel = tile * out_channels_per_tile;
  task.last_out_channel = (tile + 1) * out_channels_per_tile;

  if (task.last_out_channel > shape->out_channels)
    task.last_out_channel = shape->out_channels;

  return task;
}

pool_task_t get_tile_pool_task(const pool_shape_t* shape, int tile, int num_tiles) {
  pool_task_t task;

  int channels_per_tile = shape->channels / num_tiles;
  task.first_channel = tile * channels_per_tile;
  task.last_channel = (tile + 1) * channels_per_tile;

  if (task.last_channel > shape->channels)
    task.last_channel = shape->channels;

  return task;
}

// From the whole layer's data, extract the part that this task will use.
// No significant data copying is performed.

activation_config_t activation_slice(const activation_config_t* tensor,
                                     int first_channel, int last_channel) {
  activation_config_t slice = *tensor;
  slice.data.address += slice.channel_stride * first_channel / sizeof(data_t);
  return slice;
}

sparse_activations_t sparse_activation_slice(const sparse_activations_t* tensor,
                                             int first_channel, int last_channel) {
  // Note: in a real computation, this would be a good opportunity to
  // redistribute the workload across parallel tiles. I don't do that here
  // because I'm interested in how well other distribution methods work.

  // Start off assuming there are no channels available.
  int first_sparse_channel = tensor->num_channels;
  int num_sparse_channels = 0;

  // Find the first channel index in our given range.
  for (int i=0; i<tensor->num_channels; i++) {
    if (tensor->channels[i] >= first_channel) {
      first_sparse_channel = i;
      break;
    }
  }

  // Count how many channels are in the range.
  for (int i=first_sparse_channel; i<tensor->num_channels; i++)
    if (tensor->channels[i] < last_channel)
      num_sparse_channels++;
    else
      break;

  sparse_activations_t slice = *tensor;

  slice.dense = activation_slice(&slice.dense, first_sparse_channel,
                                 first_sparse_channel + num_sparse_channels);
  slice.channels += first_sparse_channel;
  slice.num_channels = num_sparse_channels;

  return slice;
}

filter_config_t weight_slice(const filter_config_t* tensor,
                             int first_in_channel, int last_in_channel,
                             int first_out_channel, int last_out_channel) {
  filter_config_t slice = *tensor;
  slice.data.address += slice.in_channel_stride * first_in_channel / sizeof(data_t)
                      + slice.out_channel_stride * first_out_channel / sizeof(data_t);
  return slice;
}

conv_shape_t get_conv_slice(const conv_shape_t* shape, const conv_task_t* task) {
  conv_shape_t slice = *shape;
  slice.in_channels = task->last_in_channel - task->first_in_channel;
  slice.out_channels = task->last_out_channel - task->first_out_channel;
  return slice;
}

activation_config_t get_input_conv_slice(const activation_config_t* input,
                                         const conv_task_t* task) {
  return activation_slice(input, task->first_in_channel, task->last_in_channel);
}

activation_config_t get_output_conv_slice(const activation_config_t* output,
                                          const conv_task_t* task) {
  return activation_slice(output, task->first_out_channel, task->last_out_channel);
}

filter_config_t get_weights_conv_slice(const filter_config_t* weights,
                                       const conv_task_t* task) {
  return weight_slice(weights, task->first_in_channel, task->last_in_channel,
                      task->first_out_channel, task->last_out_channel);
}

sparse_activations_t get_sparse_input_conv_slice(const sparse_activations_t* input,
                                                 const conv_task_t* task) {
  return sparse_activation_slice(input, task->first_in_channel, task->last_in_channel);
}

sparse_activations_t get_sparse_output_conv_slice(const sparse_activations_t* output,
                                                  const conv_task_t* task) {
  return sparse_activation_slice(output, task->first_out_channel, task->last_out_channel);
}

pool_shape_t get_pool_slice(const pool_shape_t* shape, const pool_task_t* task) {
  pool_shape_t slice = *shape;
  slice.channels = task->last_channel - task->first_channel;
  return slice;
}


activation_config_t get_input_pool_slice(const activation_config_t* input,
                                         const pool_task_t* task) {
  return activation_slice(input, task->first_channel, task->last_channel);
}

activation_config_t get_output_pool_slice(const activation_config_t* output,
                                          const pool_task_t* task) {
  return activation_slice(output, task->first_channel, task->last_channel);
}

sparse_activations_t get_sparse_input_pool_slice(const sparse_activations_t* input,
                                                 const pool_task_t* task) {
  return sparse_activation_slice(input, task->first_channel, task->last_channel);
}

sparse_activations_t get_sparse_output_pool_slice(const sparse_activations_t* output,
                                                  const pool_task_t* task) {
  return sparse_activation_slice(output, task->first_channel, task->last_channel);
}
