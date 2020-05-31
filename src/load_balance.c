// Method:
// Each tile maintains a notion of which computations it needs to perform.
// If a tile runs out of work to do, it communicates with a neighbour.
// The neighbour responds with a new task if it has any work left to do.
// This task can be empty if there is no spare work.
// If an empty task is received, the tile requests work from another neighbour.
// Once all 4 neighbours have been checked, the tile stops checking.

// Request = tile_id_t of requestor.
// Response = conv_task_t.

#include <stdio.h>
#include <loki/channels.h>
#include <loki/channel_io.h>
#include <loki/channel_map_table.h>
#include <loki/ids.h>
#include "defs.h"

#define LB_REQUEST_CHANNEL 4
#define LB_RESPONSE_CHANNEL 5

void init_lb_state(lb_state_t* state, int num_tiles) {
  // Counting is a bit hacky.
  // Requests aren't actually sent if there are few enough tiles that a tile's
  // neighbour wraps around to itself. The request is still counted as being
  // sent, but no request will ever be received.

  state->requests_made = 0;

  if (num_tiles > 4)
    state->requests_received = 0;
  else if (num_tiles > 1)
    state->requests_received = 2;
  else
    state->requests_received = 4;
}

// Check whether all load balancing opportunities have been taken.
bool lb_finished(const lb_state_t* state) {
  return state->requests_made == 4;
}

void empty_request_queue(lb_state_t* state);

// Assumes a 4x4 grid of tiles, filled from top to bottom, left to right.
conv_task_t check_neighbour(lb_state_t* state, int num_tiles) {
  tile_id_t this_tile = get_tile_id();
  int row = this_tile & 7;
  int col = this_tile >> 3;

  assert((num_tiles < 4) || (num_tiles % 4 == 0));
  int max_row = num_tiles > 4 ? num_tiles / 4 : 1;
  int max_col = num_tiles > 4 ? 4 : num_tiles;

  // Order is arbitrary. Wrap around ends of the grid.
  switch (state->requests_made) {
    case 0: row = (row == max_row) ? 1 : row+1; break;
    case 1: col = (col == max_col) ? 1 : col+1; break;
    case 2: row = (row == 1) ? max_row : row-1; break;
    case 3: col = (col == 1) ? max_col : col-1; break;
    default:
      printf("Error: tile %d trying to access neighbour %d (max=4)\n",
             tile2int(this_tile), state->requests_made);
      break;
  }

  tile_id_t neighbour = tile_id(col, row);
  conv_task_t response;

  if (neighbour == this_tile) {
    // Don't bother sending a request, but make sure the response is empty.
    response.first_in_channel = 0;
    response.last_in_channel = 0;
    response.first_out_channel = 0;
    response.last_out_channel = 0;
  }
  else {
    // Request = this tile_id_t.
    channel_t channel = loki_core_address(neighbour, COMPONENT_CORE_0, LB_REQUEST_CHANNEL, DEFAULT_CREDIT_COUNT);
    set_channel_map(5, channel);
    loki_send(5, this_tile);

    // Before stalling, check to make sure no one is waiting for a response from
    // us. There's still a small chance of a race condition here.
    empty_request_queue(state);

    // Wait for response.
    loki_receive_data(&response, sizeof(conv_task_t), LB_RESPONSE_CHANNEL);
  }

  return response;
}

// Request more work.
// Store the resulting task in the given parameter, and return whether there is
// any work to do.
bool make_load_balance_request(conv_task_t* task, lb_state_t* state, int num_tiles) {
  while (state->requests_made < 4) {
    *task = check_neighbour(state, num_tiles);
    state->requests_made++;

    // Check whether the neighbour returned a non-zero amount of work.
    if ((task->last_in_channel > task->first_in_channel) ||
        (task->last_out_channel > task->first_out_channel))
      return true;
  }

  return false;
}

// All neighbours write to the same input buffer. This is generally unsafe, but
// works here.
//  * Buffers have size 4
//  * Each tile has 4 neighbours, each of which will send one 1 flit request
//  * All requests come through the local router, so are serialised
bool request_pending() {
  return loki_test_channel(LB_REQUEST_CHANNEL);
}

// Split the given task in two. Update the given task to reduce its size, and
// return the piece that was removed.
conv_task_t split_task(conv_task_t* task, int in_channel_iteration,
                       int out_channel_iteration) {
  conv_task_t new_task;
  new_task.first_in_channel = task->first_in_channel;
  new_task.last_in_channel = task->last_in_channel;
  new_task.last_out_channel = task->last_out_channel;

  // Average of current position and end.
  int split_point = (out_channel_iteration + task->last_out_channel) / 2;
  if (split_point == out_channel_iteration)
    split_point++;
  new_task.first_out_channel = split_point;
  task->last_out_channel = split_point;

  return new_task;
}

// Iterations are counted within the current task only.
// TODO: don't really want to pass current iteration counts.
void check_load_balance_requests(conv_task_t* task, lb_state_t* state,
                                 int in_channel_iteration, int out_channel_iteration) {
  while (request_pending()) {
    tile_id_t tile = loki_receive(LB_REQUEST_CHANNEL);
    channel_t channel = loki_core_address(tile, COMPONENT_CORE_0, LB_RESPONSE_CHANNEL, DEFAULT_CREDIT_COUNT);
    set_channel_map(5, channel);

    conv_task_t spare_work = split_task(task, in_channel_iteration, out_channel_iteration);
    loki_send_data(&spare_work, sizeof(conv_task_t), 5);
    state->requests_received++;
  }
}

void empty_request_queue(lb_state_t* state) {
  while (request_pending()) {
    tile_id_t tile = loki_receive(LB_REQUEST_CHANNEL);
    channel_t channel = loki_core_address(tile, COMPONENT_CORE_0, LB_RESPONSE_CHANNEL, DEFAULT_CREDIT_COUNT);
    set_channel_map(5, channel);

    conv_task_t spare_work = {0,0,0,0};
    loki_send_data(&spare_work, sizeof(conv_task_t), 5);
    state->requests_received++;
  }
}

// Wait until all neighbours have finished. We may need to respond to their
// requests.
void lb_sync(lb_state_t* state) {
  while (state->requests_received < 4) {
    tile_id_t tile = loki_receive(LB_REQUEST_CHANNEL);
    channel_t channel = loki_core_address(tile, COMPONENT_CORE_0, LB_RESPONSE_CHANNEL, DEFAULT_CREDIT_COUNT);
    set_channel_map(5, channel);

    conv_task_t spare_work = {0,0,0,0};
    loki_send_data(&spare_work, sizeof(conv_task_t), 5);
    state->requests_received++;
  }
}
