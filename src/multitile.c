// Functions mostly taken from libloki, but adapted for the different hardware
// configuration when using accelerators.

#include <loki/alloc.h>
#include <loki/channel_map_table.h>
#include <loki/init.h>

#define CORES_PER_ACCELERATOR_TILE 2

// Only initialise core 0 on each tile for now.
void init_tile(const tile_id_t tile, const init_config* config) {

  // Send initial configuration.
  int data_input = loki_core_address(tile, 0, 3, INFINITE_CREDIT_COUNT);
  set_channel_map(2, data_input);
  loki_send(2, config->inst_mem);
  loki_send(2, config->data_mem);
  loki_send(2, (int)config->stack_pointer - tile2int(tile)*CORES_PER_ACCELERATOR_TILE*config->stack_size);

  // Send some instructions to execute.
  int inst_fifo = loki_core_address(tile, 0, 0, INFINITE_CREDIT_COUNT);
  set_channel_map(2, inst_fifo);

  asm volatile (
    "fetchr 0f\n"
    "rmtexecute -> 2\n "        // begin remote execution
    "setchmapi 0, r3\n"         // instruction channel
    "setchmapi 1, r3\n"         // data channel
    "nor r0, r0, r0\n"          // nop after setchmap before channel use
    "or r8, r3, r0\n"           // receive stack pointer
    "or r9, r8, r0\n"           // frame pointer = stack pointer
    "lli r10, %lo(loki_sleep)\n"
    "lui.eop r10, %hi(loki_sleep)\n"// return address = sleep
    "0:\n"
  );

  // init local tile here if needed

}

void init(int num_tiles) {
  if (num_tiles <= 1)
    return;

  init_config *config = loki_malloc(sizeof(init_config));
  config->cores = num_tiles * CORES_PER_ACCELERATOR_TILE;
  config->stack_size = 0x12000;
  config->inst_mem = get_channel_map(0);
  config->data_mem = get_channel_map(1);
  config->config_func = NULL;

  char *stack_pointer; // Core 0's default stack pointer
  asm ("addu %0, r8, r0\nfetchr.eop 0f\n0:\n" : "=r"(stack_pointer) : : );
  stack_pointer += 0x400 - ((int)stack_pointer & 0x3ff); // Guess at the initial sp
  config->stack_pointer = stack_pointer;

  loki_channel_flush_data(1, config, sizeof(init_config));
  for (unsigned int tile = 1; tile < num_tiles; tile++) {
    init_tile(int2tile(tile), config);
  }

  // init local tile here if needed

  loki_free(config);
}
