/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_apple9.h"
#include "agx_abi.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/compiler/agx_compile_apple9.h"
#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_device.h"
#include "compiler/nir/nir.h"
#include "util/u_debug.h"
#include "compiler/nir/nir_builder.h"
#include "gallium/drivers/asahi/agx_apple9.h"
#include "libagx/libagx_dgc.h"
#include "agx_bg_eot.h"
#include "libagx_shaders.h"

#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_shader.h"
#include "pool.h"

bool
hk_apple9_compute_enabled(const struct agx_device *dev)
{
   return agx_apple9_compute_enabled(dev);
}

bool
hk_apple9_coopmat_enabled(const struct agx_device *dev)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("HK_APPLE9_COOPMAT", true);
   return enabled && hk_apple9_compute_enabled(dev);
}

/* ---- Compile ----------------------------------------------------------- */

struct apple9_lower_state {
   struct hk_apple9_shader *a9;
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *const *set_layouts;
   const char *error;
};

static int
apple9_binding_index(struct apple9_lower_state *st,
                     const struct hk_apple9_binding *key)
{
   struct hk_apple9_shader *a9 = st->a9;

   for (unsigned i = 0; i < a9->binding_count; ++i) {
      const struct hk_apple9_binding *b = &a9->bindings[i];
      if (b->kind == key->kind && b->set == key->set &&
          b->binding == key->binding && b->element == key->element)
         return i;
   }

   if (a9->binding_count >= HK_APPLE9_MAX_BINDINGS) {
      st->error = "too many buffer bindings for Apple9 compute";
      return -1;
   }

   a9->bindings[a9->binding_count] = *key;
   return a9->binding_count++;
}

static bool
apple9_describe_binding(struct apple9_lower_state *st, uint32_t set,
                        uint32_t binding, uint32_t element,
                        struct hk_apple9_binding *out)
{
   if (set >= st->set_layout_count || st->set_layouts[set] == NULL) {
      st->error = "descriptor set layout missing";
      return false;
   }

   struct hk_descriptor_set_layout *layout =
      vk_to_hk_descriptor_set_layout(st->set_layouts[set]);
   if (binding >= layout->binding_count) {
      st->error = "descriptor binding out of range";
      return false;
   }

   const struct hk_descriptor_set_binding_layout *bl = &layout->binding[binding];
   *out = (struct hk_apple9_binding){
      .set = set,
      .binding = binding,
      .element = element,
   };

   switch (bl->type) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      if (element >= bl->array_size) {
         st->error = "descriptor array index out of range";
         return false;
      }
      out->kind = HK_APPLE9_BINDING_BUFFER;
      out->offset = bl->offset + element * bl->stride;
      return true;

   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      if (element >= bl->array_size) {
         st->error = "descriptor array index out of range";
         return false;
      }
      out->kind = HK_APPLE9_BINDING_DYNAMIC;
      out->dynamic_index = bl->dynamic_buffer_index + element;
      return true;

   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
      out->kind = HK_APPLE9_BINDING_INLINE;
      out->offset = bl->offset;
      return true;

   default:
      st->error = "Apple9 compute supports only buffer descriptors so far";
      return false;
   }
}

static bool
apple9_lower_resources(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct apple9_lower_state *st = data;
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_vulkan_resource_index: {
      if (!nir_src_is_const(intr->src[0])) {
         st->error = "dynamically indexed descriptor arrays are not supported yet";
         return false;
      }

      struct hk_apple9_binding key;
      if (!apple9_describe_binding(st, nir_intrinsic_desc_set(intr),
                                   nir_intrinsic_binding(intr),
                                   nir_src_as_uint(intr->src[0]), &key))
         return false;

      int index = apple9_binding_index(st, &key);
      if (index < 0)
         return false;

      assert(intr->def.num_components == 2 && intr->def.bit_size == 32);
      nir_def_replace(&intr->def,
                      nir_vec2(b, nir_imm_int(b, index), nir_imm_int(b, 0)));
      return true;
   }

   case nir_intrinsic_vulkan_resource_reindex:
      if (!nir_src_is_const(intr->src[1]) || nir_src_as_uint(intr->src[1])) {
         st->error = "descriptor reindexing is not supported yet";
         return false;
      }
      nir_def_replace(&intr->def, intr->src[0].ssa);
      return true;

   case nir_intrinsic_load_vulkan_descriptor:
      nir_def_replace(&intr->def, intr->src[0].ssa);
      return true;

   case nir_intrinsic_load_push_constant: {
      struct hk_apple9_binding key = {.kind = HK_APPLE9_BINDING_PUSH};
      int index = apple9_binding_index(st, &key);
      if (index < 0)
         return false;

      unsigned align = MAX2(intr->def.bit_size / 8, 1);
      nir_def *offset =
         nir_iadd_imm(b, intr->src[0].ssa, nir_intrinsic_base(intr));
      nir_def *value = nir_load_ubo(
         b, intr->def.num_components, intr->def.bit_size, nir_imm_int(b, index),
         offset, .align_mul = align, .align_offset = 0, .range_base = 0,
         .range = HK_MAX_PUSH_SIZE);
      nir_def_replace(&intr->def, value);
      return true;
   }

   default:
      return false;
   }
}

static void
apple9_optimize(nir_shader *nir)
{
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
   } while (progress);
}

VkResult
hk_apple9_compile_compute(struct hk_device *dev, nir_shader *nir,
                          uint32_t set_layout_count,
                          struct vk_descriptor_set_layout *const *set_layouts,
                          struct hk_shader *shader)
{
   struct hk_apple9_shader *a9 = &shader->apple9;
   memset(a9, 0, sizeof(*a9));
   a9->enabled = true;
   simple_mtx_init(&shader->linked.lock, mtx_plain);
   shader->info.stage = MESA_SHADER_COMPUTE;
   shader->info.set_count = set_layout_count;

   struct apple9_lower_state st = {
      .a9 = a9,
      .set_layout_count = set_layout_count,
      .set_layouts = set_layouts,
   };

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, apple9_lower_resources,
            nir_metadata_control_flow, &st);

   if (st.error) {
      fprintf(stderr, "hk/apple9: %s\n", st.error);
      ralloc_free(nir);
      return vk_errorf(dev, VK_ERROR_UNKNOWN, "Apple9 compute: %s", st.error);
   }

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo | nir_var_mem_ubo,
            nir_address_format_32bit_index_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_2x32bit_global);
   apple9_optimize(nir);

   for (unsigned d = 0; d < 3; ++d)
      a9->local_size[d] = nir->info.workgroup_size[d];

   const char *reason = NULL;
   if (!agx_compile_apple9_tiny(nir, &shader->b, &a9->profile, &reason)) {
      fprintf(stderr, "hk/apple9: compute shader rejected: %s\n",
              reason ?: "unknown reason");
      if (getenv("HK_APPLE9_PRINT_NIR"))
         nir_print_shader(nir, stderr);
      ralloc_free(nir);
      return vk_errorf(dev, VK_ERROR_UNKNOWN, "Apple9 compute compile failed");
   }

   if (!a9->profile.variable_local_size) {
      for (unsigned d = 0; d < 3; ++d)
         a9->local_size[d] = a9->profile.local_size[d];
   }

   ralloc_free(nir);

   shader->code_ptr = shader->b.binary;
   shader->code_size = shader->b.info.binary_size;

   a9->body = agx_bo_create(&dev->dev, shader->b.info.binary_size, 0,
                            AGX_BO_EXEC | AGX_BO_LOW_VA | AGX_BO_WRITEBACK,
                            "Apple9 compute body");
   if (!a9->body)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   memcpy(agx_bo_map(a9->body), shader->b.binary, shader->b.info.binary_size);
   agx_bo_note_cpu_write(a9->body, 0, shader->b.info.binary_size);
   return VK_SUCCESS;
}

/* ---- Dispatch ---------------------------------------------------------- */

/* Read-only buffer descriptors point at Honeykrisp's read-only mirror of the
 * main heap. Apple9 resource pointers use the primary mapping. */
static uint64_t
apple9_primary_address(struct hk_device *dev, uint64_t addr)
{
   uint64_t ro = dev->dev.sparse_ro_offset;
   if (ro && addr < (1ull << 39) && (addr & ro))
      return addr - ro;
   return addr;
}

static const uint8_t *
apple9_set_data(struct hk_descriptor_state *desc, unsigned set)
{
   if (desc->sets[set])
      return desc->sets[set]->mapped_ptr;
   if (desc->push[set])
      return desc->push[set]->data;
   return NULL;
}

static bool
apple9_binding_address(struct hk_cmd_buffer *cmd, struct hk_descriptor_state *desc,
                       const struct hk_apple9_binding *b, uint64_t *out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   switch (b->kind) {
   case HK_APPLE9_BINDING_PUSH:
      *out = hk_pool_upload(cmd, desc->root.push, HK_MAX_PUSH_SIZE, 16);
      return *out != 0;

   case HK_APPLE9_BINDING_DYNAMIC: {
      unsigned i = desc->root.set_dynamic_buffer_start[b->set] + b->dynamic_index;
      if (i >= HK_MAX_DYNAMIC_BUFFERS)
         return false;
      *out = apple9_primary_address(dev, desc->root.dynamic_buffers[i].base_addr);
      return true;
   }

   case HK_APPLE9_BINDING_INLINE:
      *out = desc->root.sets[b->set] + b->offset;
      return desc->root.sets[b->set] != 0;

   case HK_APPLE9_BINDING_BUFFER: {
      const uint8_t *data = apple9_set_data(desc, b->set);
      if (!data)
         return false;
      struct hk_buffer_address addr;
      memcpy(&addr, data + b->offset, sizeof(addr));
      *out = apple9_primary_address(dev, addr.base_addr);
      return true;
   }

   default:
      return false;
   }
}


/* Emit one Apple9 compute launch into a CDM stream.
 *
 * Direct launches take thread counts. Indirect launches take a pointer to
 * three workgroup counts; with indirect_local, the pointer instead holds
 * thread counts, and group_counts points at the matching workgroup counts. */
static bool
apple9_emit_launch(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                   struct agx_bo *body,
                   const struct agx_apple9_compute_profile *profile,
                   const uint64_t *resources, unsigned count,
                   const uint32_t local[3], const struct agx_grid *grid,
                   uint64_t group_counts, const char *what)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   bool indirect = agx_is_indirect(*grid);
   bool indirect_local = grid->mode == AGX_CDM_MODE_INDIRECT_LOCAL;

   if (indirect && !agx_apple9_compute_indirect_dispatch_supported(profile)) {
      fprintf(stderr, "hk/apple9: %s: indirect launch unsupported\n", what);
      return false;
   }

   if (!cmd->apple9_pool_valid) {
      agx_pool_init(&cmd->apple9_pool, &dev->dev, "Apple9 compute launch state",
                    AGX_BO_LOW_VA | AGX_BO_WRITEBACK, true);
      cmd->apple9_pool_valid = true;
   }

   struct agx_apple9_compute_geometry geometry = {
      .mode = indirect ? AGX_APPLE9_COMPUTE_GEOMETRY_INDIRECT
                       : AGX_APPLE9_COMPUTE_GEOMETRY_DIRECT,
      .local = {local[0], local[1], local[2]},
   };
   if (indirect)
      geometry.group_counts = group_counts ?: grid->ptr;
   else
      memcpy(geometry.threads, grid->count, sizeof(geometry.threads));

   uint64_t launch;
   uint64_t preamble =
      profile->preamble_size ? body->va->addr + profile->preamble_offset : 0;
   if (!agx_apple9_prepare_compute_dispatch(&dev->dev, &cmd->apple9_pool, body,
                                            profile, resources, count, 0, 0,
                                            &geometry, preamble, &launch)) {
      fprintf(stderr, "hk/apple9: %s: invalid launch state\n", what);
      return false;
   }

   unsigned size = indirect_local ? AGX_APPLE9_COMPUTE_INDIRECT_LOCAL_CDM_RECORD_SIZE
                   : indirect     ? AGX_APPLE9_COMPUTE_INDIRECT_CDM_RECORD_SIZE
                                  : AGX_APPLE9_COMPUTE_CDM_RECORD_SIZE;
   hk_ensure_cs_has_space(cmd, cs, size + 64);

   bool ok = indirect
      ? agx_apple9_emit_indirect_dispatch(cs->current, launch, grid->ptr,
                                          geometry.local, indirect_local, profile)
      : agx_apple9_emit_direct_dispatch(cs->current, launch, geometry.threads,
                                        geometry.local, profile);
   if (!ok) {
      fprintf(stderr, "hk/apple9: %s: launch geometry rejected\n", what);
      return false;
   }

   static int debug = -1;
   if (debug < 0)
      debug = getenv("HK_APPLE9_DEBUG") != NULL;
   if (debug) {
      fprintf(stderr,
              "hk/apple9: %s abi=%u resources=%u local=%ux%ux%u mode=%u "
              "threads=%ux%ux%u preamble=%u+%u scratch=%u tgmem=%u "
              "body=0x%" PRIx64 " launch=0x%" PRIx64 "\n",
              what, profile->abi, count, local[0], local[1], local[2],
              grid->mode, indirect ? 0 : grid->count[0],
              indirect ? 0 : grid->count[1], indirect ? 0 : grid->count[2],
              profile->preamble_offset, profile->preamble_size,
              profile->scratch_size, profile->required_threadgroup_memory_bytes,
              body->va->addr, launch);
      for (unsigned i = 0; i < count; ++i)
         fprintf(stderr, "hk/apple9:   resource[%u] kind=%u binding=%u addr=0x%" PRIx64 "\n",
                 i, profile->resource_kind[i], profile->resource_binding[i],
                 resources[i]);
   }

   cs->current = (uint8_t *)cs->current + size;
   cs->stats.cmds++;
   cs->stats.calls++;

   /* Profiling aid: HK_APPLE9_DUP=all, or a class key
    * "preamble_offset/local_x/resources/threadgroup_bytes", launches matching
    * work twice. The added time per token over the class's launch count is
    * that class's GPU cost. Results are wrong while it is set. */
   static const char *dup = (const char *)-1;
   if (dup == (const char *)-1)
      dup = getenv("HK_APPLE9_DUP");
   if (dup) {
      char key[64];
      snprintf(key, sizeof(key), "%u/%u/%u/%u", profile->preamble_offset,
               local[0], count, profile->required_threadgroup_memory_bytes);
      if (!strcmp(dup, "all") || !strcmp(dup, key)) {
         uint8_t record[64];
         memcpy(record, (uint8_t *)cs->current - size, size);
         hk_ensure_cs_has_space(cmd, cs, size + 64);
         memcpy(cs->current, record, size);
         cs->current = (uint8_t *)cs->current + size;
      }
   }
   return true;
}

void
hk_apple9_dispatch(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                   const struct agx_grid *grid)
{
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   struct hk_apple9_shader *a9 = &s->apple9;
   const struct agx_apple9_compute_profile *profile = &a9->profile;
   bool indirect = agx_is_indirect(*grid);

   if (!indirect && (!grid->count[0] || !grid->count[1] || !grid->count[2]))
      return;

   uint64_t resources[AGX_APPLE9_COMPUTE_MAX_RESOURCES];
   unsigned count = profile->resource_binding_count;
   for (unsigned i = 0; i < count; ++i) {
      if (profile->resource_kind[i] == AGX_APPLE9_COMPUTE_RESOURCE_SHARED) {
         resources[i] = AGX_APPLE9_COMPUTE_SHARED_ROOT;
         continue;
      }

      unsigned binding = profile->resource_binding[i];
      if (binding >= a9->binding_count ||
          !apple9_binding_address(cmd, desc, &a9->bindings[binding],
                                  &resources[i])) {
         fprintf(stderr, "hk/apple9: unresolved resource %u (binding %u)\n", i,
                 binding);
         return;
      }
      /* Null descriptors read as zero. The compiler also redirects guarded
       * loads to element 0 of their binding, which must stay mapped. */
      if (resources[i] == 0)
         resources[i] = AGX_ZERO_PAGE_ADDRESS;
   }

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true /* compute */);
   if (!cs)
      return;

   /* API grids are in workgroups; Apple9 direct launches take threads. */
   struct agx_grid launch_grid = *grid;
   if (!indirect) {
      for (unsigned d = 0; d < 3; ++d)
         launch_grid.count[d] = grid->count[d] * a9->local_size[d];
   }

   apple9_emit_launch(cmd, cs, a9->body, profile, resources, count,
                      a9->local_size, &launch_grid, 0, "dispatch");
}

bool
hk_apple9_dispatch_precomp(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                           struct agx_precompiled_shader *prog,
                           const struct agx_grid *grid_in, uint64_t args)
{
   struct agx_grid grid = *grid_in;
   const struct agx_apple9_compute_profile *profile = &prog->apple9_profile;

   if (grid.mode == AGX_CDM_MODE_DIRECT &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return true;

   uint64_t resources[AGX_APPLE9_COMPUTE_MAX_RESOURCES];
   unsigned count = profile->resource_binding_count;
   for (unsigned i = 0; i < count; ++i) {
      if (profile->resource_kind[i] == AGX_APPLE9_COMPUTE_RESOURCE_SHARED) {
         resources[i] = AGX_APPLE9_COMPUTE_SHARED_ROOT;
      } else if (profile->resource_kind[i] == AGX_APPLE9_COMPUTE_RESOURCE_UBO &&
                 profile->resource_binding[i] == 0) {
         /* Internal kernels read their argument block as UBO 0. */
         resources[i] = args;
      } else {
         fprintf(stderr, "hk/apple9: internal kernel has unexpected resource %u\n", i);
         return false;
      }
   }

   uint64_t group_counts = 0;
   if (grid.mode == AGX_CDM_MODE_INDIRECT_LOCAL) {
      /* Convert the indirect thread counts to workgroup counts on the GPU. */
      struct agx_ptr groups = hk_pool_alloc(cmd, 12, 4);
      if (!groups.cpu)
         return false;
      libagx_workgroups_from_threads(cmd, agx_1d(1), AGX_BARRIER_ALL,
                                     groups.gpu, grid.ptr);
      group_counts = groups.gpu;
   }

   uint32_t local[3] = {prog->b.workgroup.x, prog->b.workgroup.y,
                        prog->b.workgroup.z};
   return apple9_emit_launch(cmd, cs, prog->bo, profile, resources, count, local,
                             &grid, group_counts, "internal kernel");
}
