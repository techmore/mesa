/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Apple9 (M4, G16G) compute for Honeykrisp.
 *
 * Apple9 compute programs are built by the bounded Apple9 compiler and
 * launched through generated launch records. Buffers reach the program as
 * direct resource pointers, so Vulkan descriptors are flattened to numbered
 * bindings at compile time and resolved to addresses on the CPU at dispatch.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "vulkan/vulkan_core.h"
#include "asahi/compiler/agx_apple9_profile.h"

struct agx_bo;
struct agx_precompiled_shader;
struct hk_cs;
struct agx_device;
struct agx_grid;
struct hk_cmd_buffer;
struct hk_device;
struct hk_shader;
struct nir_shader;
struct vk_descriptor_set_layout;

#define HK_APPLE9_MAX_BINDINGS AGX_APPLE9_COMPUTE_MAX_RESOURCES

enum hk_apple9_binding_kind {
   /* hk_buffer_address in descriptor set memory */
   HK_APPLE9_BINDING_BUFFER,
   /* hk_buffer_address in the root table's dynamic buffer array */
   HK_APPLE9_BINDING_DYNAMIC,
   /* Inline uniform block stored in descriptor set memory */
   HK_APPLE9_BINDING_INLINE,
   /* Client push constants */
   HK_APPLE9_BINDING_PUSH,
};

struct hk_apple9_binding {
   uint8_t kind;
   uint8_t set;
   uint16_t dynamic_index;
   uint32_t binding;
   uint32_t element;
   /* Byte offset of the descriptor (or inline data) within the set */
   uint32_t offset;
};

struct hk_apple9_shader {
   bool enabled;
   struct agx_bo *body;
   struct agx_apple9_compute_profile profile;
   uint32_t local_size[3];
   uint8_t binding_count;
   struct hk_apple9_binding bindings[HK_APPLE9_MAX_BINDINGS];
};

bool hk_apple9_compute_enabled(const struct agx_device *dev);

/* VK_KHR_cooperative_matrix on Apple9 compute (fp32 8x8x8, subgroup scope).
 * Opt-in with HK_APPLE9_COOPMAT=1 until it has wider hardware coverage. */
bool hk_apple9_coopmat_enabled(const struct agx_device *dev);

VkResult hk_apple9_compile_compute(struct hk_device *dev, struct nir_shader *nir,
                                   uint32_t set_layout_count,
                                   struct vk_descriptor_set_layout *const *set_layouts,
                                   struct hk_shader *shader);

/* grid is in workgroups (direct) or points at VkDispatchIndirectCommand. */
void hk_apple9_dispatch(struct hk_cmd_buffer *cmd, struct hk_shader *shader,
                        const struct agx_grid *grid);

/* Launch a precompiled libagx kernel (args uploaded at `args`) on Apple9. */
bool hk_apple9_dispatch_precomp(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                                struct agx_precompiled_shader *prog,
                                const struct agx_grid *grid, uint64_t args);
