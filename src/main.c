#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <loki/channels.h>
#include <loki/spawn.h>
#include <nn/layers.h>
#include "defs.h"

typedef struct {
  conv_shape_t shape;

  test_fn* test;
  void* buffers;

  int in_sparsity;
  int out_sparsity;

  int num_tiles;
} test_config;

// Function executed by core 0 of every active tile.
static void tile_task(const void* data) {
  const test_config* config = (const test_config*)data;

  config->test(
    &config->shape,
    config->buffers,
    config->in_sparsity,
    config->out_sparsity,
    config->num_tiles
  );

  loki_sync_tiles(config->num_tiles);
}

int main(int argc, char** argv) {
  if (argc < 7) {
    printf(""
    "Usage: lat-dynamic in-channels in-size in-sparsity out-channels\\ \n"
    "                   out-sparsity filter-size [--mode=mode] [--tiles=N]\n"
    "'size' parameters indicate the width/height in pixels\n"
    "'sparsity' parameters are percentages\n"
    "'mode' selects how to exploit sparsity ('none', 'simple', 'adaptive')\n");
    exit(1);
  }

  // TODO: do all memory allocation out here, and pass dense tensors to other
  // tiles.
  test_config config;
  config.shape.in_channels = atoi(argv[1]);
  config.shape.image_width = atoi(argv[2]);
  config.shape.image_height = config.shape.image_width;
  config.in_sparsity = atoi(argv[3]);
  config.shape.out_channels = atoi(argv[4]);
  config.out_sparsity = atoi(argv[5]);
  config.shape.filter_width = atoi(argv[6]);
  config.shape.filter_height = config.shape.filter_width;
  config.shape.batch_size = 1;
  config.shape.groups = 1;
  config.shape.stride = 1;
  config.shape.dilation = 1;

  config.test = test_simple;
  config.num_tiles = 1;

  for (int i=7; i<argc; i++) {
    if (!strncmp(argv[i], "--mode=", 7)) {
      char* mode = argv[i] + 7;

      if (!strcmp(mode, "none")) {
        config.test = test_none;
        config.buffers = init_dense_buffers(&config.shape);
      }
      else if (!strcmp(mode, "simple")) {
        config.test = test_simple;
        config.buffers = init_sparse_buffers(&config.shape, config.in_sparsity);
      }
      else if (!strcmp(mode, "adaptive")) {
        config.test = test_adaptive;
        config.buffers = init_sparse_buffers(&config.shape, config.in_sparsity);
      }
      else {
        printf("Error: unknown mode parameter: '%s'\n", mode);
        exit(1);
      }
    }
    else if (!strncmp(argv[i], "--tiles=", 8)) {
      char* tiles = argv[i] + 8;
      config.num_tiles = atoi(tiles);
    }
    else {
      printf("Unknown argument: %s\n", argv[i]);
      exit(1);
    }
  }

  // Distribution of work across tiles is very simple at the moment.
  assert(config.shape.in_channels % config.num_tiles == 0);
  assert(config.shape.out_channels % config.num_tiles == 0);

  // Can't use libloki initialisation because that assumes 8 cores per tile.
  init(config.num_tiles);

  // Flush function arguments so remote tiles can access them.
  loki_channel_flush_data(1, &config, sizeof(test_config));

  // Start timer.
  unsigned long start = get_cycle_count();

  // Main computation.
  for (int tile = config.num_tiles-1; tile >= 0; tile--) {
    loki_remote_execute(int2tile(tile), 0, &tile_task, &config,
                        sizeof(test_config));
  }

  // Stop timer.
  unsigned long duration = get_cycle_count() - start;
  printf("Computation took %lu cycles\n", duration);

  if (config.test == test_none)
    delete_dense_buffers(config.buffers);
  else
    delete_sparse_buffers(config.buffers);

  return 0;
}
