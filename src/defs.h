#ifndef DEFS_H
#define DEFS_H

#include <nn/layers.h>

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

// Function to set up cores on remote tiles. Must be called before any
// computation is performed.
void init(int num_tiles);

typedef void test_fn(const conv_shape_t* shape, void* data,
                     int in_sparsity, int out_sparsity, int num_tiles);
test_fn test_none;
test_fn test_simple;
test_fn test_adaptive;

void* init_dense_buffers(const conv_shape_t* shape);
void* init_sparse_buffers(const conv_shape_t* shape, int in_sparsity);

typedef void dealloc_fn(void* buffers);
dealloc_fn delete_dense_buffers;
dealloc_fn delete_sparse_buffers;


// TASKS - breaking a computation into smaller units.

// For now, all tasks are defined over an integer number of channels.
// Could also split the work up spatially.
typedef struct {
  int first_in_channel;  // inclusive
  int last_in_channel;   // exclusive
  int first_out_channel; // inclusive
  int last_out_channel;  // exclusive
} conv_task_t;

typedef struct {
  int first_channel; // inclusive
  int last_channel;  // exclusive
} pool_task_t;

// Split a layer up across tiles. This is just an initial split, and can be
// renegotiated later.
conv_task_t get_tile_conv_task(const conv_shape_t* shape, int tile, int num_tiles);
pool_task_t get_tile_pool_task(const pool_shape_t* shape, int tile, int num_tiles);

activation_config_t activation_slice(const activation_config_t* tensor,
                                     int first_channel, int last_channel);
filter_config_t weight_slice(const filter_config_t* tensor,
                             int first_in_channel, int last_in_channel,
                             int first_out_channel, int last_out_channel);

conv_shape_t get_conv_slice(const conv_shape_t* shape, const conv_task_t* task);
pool_shape_t get_pool_slice(const pool_shape_t* shape, const pool_task_t* task);

typedef activation_config_t conv_act_slice_fn(const activation_config_t* tensor,
                                              const conv_task_t* task);
conv_act_slice_fn get_input_conv_slice;
conv_act_slice_fn get_output_conv_slice;

typedef sparse_activations_t conv_sparse_act_slice_fn(const sparse_activations_t* tensor,
                                                      const conv_task_t* task);
conv_sparse_act_slice_fn get_sparse_input_conv_slice;
conv_sparse_act_slice_fn get_sparse_output_conv_slice;

filter_config_t get_weights_conv_slice(const filter_config_t* weights,
                                       const conv_task_t* task);

typedef activation_config_t pool_act_slice_fn(const activation_config_t* tensor,
                                              const pool_task_t* task);
pool_act_slice_fn get_input_pool_slice;
pool_act_slice_fn get_output_pool_slice;

typedef sparse_activations_t pool_sparse_act_slice_fn(const sparse_activations_t* tensor,
                                                      const pool_task_t* task);
pool_sparse_act_slice_fn get_sparse_input_pool_slice;
pool_sparse_act_slice_fn get_sparse_output_pool_slice;

#endif // include guard
