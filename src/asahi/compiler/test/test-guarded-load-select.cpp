/* Copyright 2026 Sean Dolbec; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_nir.h"
#include "asahi/lib/agx_abi.h"
}

/*
 * agx_nir_speculate_guarded_loads: `cond ? buffer[i] : 0` over a read-only
 * buffer becomes a load from the selected address (the zero page when the
 * arm is not taken) and an if that peephole select may flatten. Anything
 * else keeps its branch.
 */

/* Index-setting helpers: C++ rejects nir_builder's named arguments. */
static nir_def *
push_constant(nir_builder *b, unsigned bit_size, unsigned offset)
{
   nir_def *def = nir_load_push_constant(b, 1, bit_size, nir_imm_int(b, offset));
   nir_intrinsic_set_range(nir_def_as_intrinsic(def), offset + bit_size / 8);
   return def;
}

static nir_def *
load_global(nir_builder *b, nir_def *addr, gl_access_qualifier access,
            unsigned align_mul = 4, unsigned align_offset = 0)
{
   nir_def *def = nir_load_global(b, 1, 32, addr);
   nir_intrinsic_set_access(nir_def_as_intrinsic(def), access);
   nir_intrinsic_set_align(nir_def_as_intrinsic(def), align_mul, align_offset);
   return def;
}

static const gl_access_qualifier read_only =
   (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);

static void
store_global(nir_builder *b, nir_def *value, nir_def *addr)
{
   nir_intrinsic_instr *intr = nir_store_global(b, value, addr);
   nir_intrinsic_set_align(intr, 4, 0);
   nir_intrinsic_set_write_mask(intr, 1);
}

enum arm_extra {
   ARM_LOAD_ONLY,
   ARM_STORE,
   ARM_WRITABLE_LOAD,
   ARM_VOLATILE_LOAD,
};

struct guarded {
   nir_builder b;
   nir_if *nif;
   nir_intrinsic_instr *load;
   nir_def *base;
   nir_def *index;
};

static guarded
build(mesa_shader_stage stage, arm_extra extra, bool dont_flatten = false)
{
   guarded g;
   g.b = nir_builder_init_simple_shader(stage, &agx_nir_options,
                                        "guarded load");
   nir_builder *b = &g.b;
   g.base = push_constant(b, 64, 0);
   g.index = push_constant(b, 32, 8);
   nir_def *cond = nir_ult_imm(b, g.index, 100);

   g.nif = nir_push_if(b, cond);
   if (dont_flatten)
      g.nif->control = nir_selection_control_dont_flatten;
   nir_def *offset = nir_u2u64(b, nir_imul_imm(b, g.index, 4));
   nir_def *addr = nir_iadd(b, g.base, offset);
   gl_access_qualifier access =
      (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
   if (extra == ARM_WRITABLE_LOAD)
      access = ACCESS_CAN_REORDER;
   else if (extra == ARM_VOLATILE_LOAD)
      access = (gl_access_qualifier)(access | ACCESS_VOLATILE);
   nir_def *value = load_global(b, addr, access);
   g.load = nir_def_as_intrinsic(value);
   if (extra == ARM_STORE)
      store_global(b, value, g.base);
   nir_push_else(b, NULL);
   nir_def *zero = nir_imm_int(b, 0);
   nir_pop_if(b, NULL);

   nir_def *result = nir_if_phi(b, value, zero);
   nir_def *output = nir_iadd_imm(b, g.base, 64);
   store_global(b, result, output);
   return g;
}

static nir_intrinsic_instr *
only_global_load(nir_shader *shader)
{
   nir_intrinsic_instr *found = NULL;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_global ||
                intr->intrinsic == nir_intrinsic_load_global_constant) {
               EXPECT_EQ(found, nullptr);
               found = intr;
            }
         }
      }
   }
   return found;
}

static bool
is_bcsel_of(nir_def *def, nir_def *taken, uint64_t other)
{
   nir_alu_instr *sel = nir_def_as_alu_or_null(def);
   return sel && sel->op == nir_op_bcsel &&
          sel->src[1].src.ssa == taken &&
          nir_src_is_const(sel->src[2].src) &&
          nir_src_as_uint(sel->src[2].src) == other;
}

TEST(GuardedLoadSelect, ReadOnlyLoadReadsZeroPageWhenNotTaken)
{
   guarded g = build(MESA_SHADER_COMPUTE, ARM_LOAD_ONLY);
   ASSERT_TRUE(agx_nir_speculate_guarded_loads(g.b.shader));
   nir_validate_shader(g.b.shader, "guarded load");

   EXPECT_EQ(g.nif->control, nir_selection_control_flatten);
   nir_intrinsic_instr *load = only_global_load(g.b.shader);
   ASSERT_NE(load, nullptr);
   EXPECT_EQ(load->intrinsic, nir_intrinsic_load_global_constant);
   EXPECT_TRUE(nir_intrinsic_access(load) & ACCESS_CAN_SPECULATE);
   EXPECT_TRUE(nir_intrinsic_access(load) & ACCESS_NON_WRITEABLE);

   /* base + u2u64(index * 4) keeps that shape for agx_nir_lower_address,
    * with the base and the unscaled index selected separately. */
   nir_alu_instr *add = nir_def_as_alu_or_null(load->src[0].ssa);
   ASSERT_NE(add, nullptr);
   ASSERT_EQ(add->op, nir_op_iadd);
   EXPECT_TRUE(is_bcsel_of(add->src[0].src.ssa, g.base, AGX_ZERO_PAGE_ADDRESS));
   nir_alu_instr *cvt = nir_def_as_alu_or_null(add->src[1].src.ssa);
   ASSERT_NE(cvt, nullptr);
   EXPECT_EQ(cvt->op, nir_op_u2u64);
   nir_alu_instr *scale = nir_def_as_alu_or_null(cvt->src[0].src.ssa);
   ASSERT_NE(scale, nullptr);
   EXPECT_TRUE(is_bcsel_of(scale->src[0].src.ssa, g.index, 0));

   /* A second run finds nothing more to do. */
   EXPECT_FALSE(agx_nir_speculate_guarded_loads(g.b.shader));
   ralloc_free(g.b.shader);
}

static void
expect_unchanged(mesa_shader_stage stage, arm_extra extra,
                 bool dont_flatten = false)
{
   guarded g = build(stage, extra, dont_flatten);
   EXPECT_FALSE(agx_nir_speculate_guarded_loads(g.b.shader));
   EXPECT_NE(g.nif->control, nir_selection_control_flatten);
   EXPECT_EQ(only_global_load(g.b.shader), g.load);
   EXPECT_EQ(g.load->intrinsic, nir_intrinsic_load_global);
   EXPECT_FALSE(nir_intrinsic_access(g.load) & ACCESS_CAN_SPECULATE);
   ralloc_free(g.b.shader);
}

TEST(GuardedLoadSelect, ArmWithStoreKeepsBranch)
{
   expect_unchanged(MESA_SHADER_COMPUTE, ARM_STORE);
}

TEST(GuardedLoadSelect, WritableLoadKeepsBranch)
{
   expect_unchanged(MESA_SHADER_COMPUTE, ARM_WRITABLE_LOAD);
}

TEST(GuardedLoadSelect, VolatileLoadKeepsBranch)
{
   expect_unchanged(MESA_SHADER_COMPUTE, ARM_VOLATILE_LOAD);
}

TEST(GuardedLoadSelect, DontFlattenIsRespected)
{
   expect_unchanged(MESA_SHADER_COMPUTE, ARM_LOAD_ONLY, true);
}

TEST(GuardedLoadSelect, OnlyComputeShaders)
{
   expect_unchanged(MESA_SHADER_FRAGMENT, ARM_LOAD_ONLY);
}

/* A descriptor read the driver already marked speculatable keeps its
 * address; the if is still flattened for the other load. */
TEST(GuardedLoadSelect, SpeculatableLoadKeepsAddress)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "guarded descriptor");
   nir_def *base = push_constant(&b, 64, 0);
   nir_def *index = push_constant(&b, 32, 8);
   nir_if *nif = nir_push_if(&b, nir_ult_imm(&b, index, 100));
   nir_def *desc = agx_test_load_global_constant(
      &b, 1, 64, base, 8,
      (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER |
                            ACCESS_CAN_SPECULATE));
   nir_def *value = load_global(
      &b, nir_iadd(&b, desc, nir_u2u64(&b, index)),
      (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER));
   nir_push_else(&b, NULL);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_pop_if(&b, NULL);
   store_global(&b, nir_if_phi(&b, value, zero), base);

   ASSERT_TRUE(agx_nir_speculate_guarded_loads(b.shader));
   nir_validate_shader(b.shader, "guarded descriptor");
   EXPECT_EQ(nif->control, nir_selection_control_flatten);
   EXPECT_EQ(nir_def_as_intrinsic(desc)->src[0].ssa, base);
   ralloc_free(b.shader);
}

/* bcsel(cond, taken, other) for the then arm; the else arm's guard is the
 * same select with its operands swapped. */
static bool
is_guard(nir_def *def, nir_def *cond, nir_def *taken, uint64_t other,
         bool then_arm)
{
   nir_alu_instr *sel = nir_def_as_alu_or_null(def);
   if (!sel || sel->op != nir_op_bcsel || sel->src[0].src.ssa != cond)
      return false;
   const nir_src &t = sel->src[then_arm ? 1 : 2].src;
   const nir_src &o = sel->src[then_arm ? 2 : 1].src;
   return t.ssa == taken && nir_src_is_const(o) && nir_src_as_uint(o) == other;
}

/* base + u2u64(index * scale), with the base and the unscaled index guarded
 * separately: when the arm is not taken the sum is exactly `fallback`. */
static void
expect_guarded_address(nir_intrinsic_instr *load, nir_def *cond, nir_def *base,
                       nir_def *index, uint64_t fallback, bool then_arm)
{
   SCOPED_TRACE(then_arm ? "then arm" : "else arm");
   ASSERT_EQ(load->intrinsic, nir_intrinsic_load_global_constant);
   EXPECT_TRUE(nir_intrinsic_access(load) & ACCESS_CAN_SPECULATE);
   nir_alu_instr *add = nir_def_as_alu_or_null(load->src[0].ssa);
   ASSERT_NE(add, nullptr);
   ASSERT_EQ(add->op, nir_op_iadd);
   EXPECT_TRUE(is_guard(add->src[0].src.ssa, cond, base, fallback, then_arm));
   nir_alu_instr *cvt = nir_def_as_alu_or_null(add->src[1].src.ssa);
   ASSERT_NE(cvt, nullptr);
   ASSERT_EQ(cvt->op, nir_op_u2u64);
   nir_alu_instr *scale = nir_def_as_alu_or_null(cvt->src[0].src.ssa);
   ASSERT_NE(scale, nullptr);
   EXPECT_TRUE(is_guard(scale->src[0].src.ssa, cond, index, 0, then_arm));
}

static unsigned
count_ifs(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_cf_node *next = nir_cf_node_next(&block->cf_node);
         n += next && next->type == nir_cf_node_if;
      }
   }
   return n;
}

static bool
peephole_select(nir_shader *shader)
{
   /* The options agx_optimize_nir uses; indirect loads stay not OK. */
   nir_opt_peephole_select_options options = {};
   options.limit = 64;
   options.expensive_alu_ok = true;
   return nir_opt_peephole_select(shader, &options);
}

struct two_arms {
   nir_builder b;
   nir_def *cond, *base[2], *index[2];
   nir_intrinsic_instr *store;
};

/* out = cond ? x[i] : y[j], with x and y read-only. */
static two_arms
build_two_arms()
{
   two_arms t;
   t.b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options,
                                        "guarded loads in both arms");
   nir_builder *b = &t.b;
   t.base[0] = push_constant(b, 64, 0);
   t.base[1] = push_constant(b, 64, 8);
   t.index[0] = push_constant(b, 32, 16);
   t.index[1] = push_constant(b, 32, 20);
   t.cond = nir_ult_imm(b, t.index[0], 100);

   nir_push_if(b, t.cond);
   nir_def *x = load_global(
      b, nir_iadd(b, t.base[0], nir_u2u64(b, nir_imul_imm(b, t.index[0], 4))),
      read_only);
   nir_push_else(b, NULL);
   nir_def *y = load_global(
      b, nir_iadd(b, t.base[1], nir_u2u64(b, nir_ishl_imm(b, t.index[1], 3))),
      read_only, 8, 0);
   nir_pop_if(b, NULL);

   nir_def *out = nir_if_phi(b, x, y);
   t.store = nir_store_global(b, out, nir_iadd_imm(b, t.base[0], 64));
   nir_intrinsic_set_align(t.store, 4, 0);
   nir_intrinsic_set_write_mask(t.store, 1);
   return t;
}

/* Together with peephole select, the branch disappears: each arm's load
 * reads its own address when its arm is taken and the zero page otherwise
 * (the else arm's guard is reversed), and the result selects between the
 * two loads on the original condition. */
TEST(GuardedLoadSelect, BothArmsBecomeOneSelect)
{
   two_arms t = build_two_arms();
   ASSERT_TRUE(agx_nir_speculate_guarded_loads(t.b.shader));
   ASSERT_TRUE(peephole_select(t.b.shader));
   nir_validate_shader(t.b.shader, "guarded loads in both arms");
   EXPECT_EQ(count_ifs(t.b.shader), 0u);

   nir_alu_instr *sel = nir_def_as_alu_or_null(t.store->src[0].ssa);
   ASSERT_NE(sel, nullptr);
   ASSERT_EQ(sel->op, nir_op_bcsel);
   EXPECT_EQ(sel->src[0].src.ssa, t.cond);
   for (unsigned arm = 0; arm < 2; ++arm) {
      nir_def *value = sel->src[1 + arm].src.ssa;
      ASSERT_EQ(nir_def_instr(value)->type, nir_instr_type_intrinsic);
      expect_guarded_address(nir_def_as_intrinsic(value), t.cond,
                             t.base[arm], t.index[arm], AGX_ZERO_PAGE_ADDRESS,
                             arm == 0);
   }
   ralloc_free(t.b.shader);
}

/* Without the pass, peephole select keeps both loads under the branch. */
TEST(GuardedLoadSelect, PeepholeSelectAloneKeepsBranch)
{
   two_arms t = build_two_arms();
   EXPECT_FALSE(peephole_select(t.b.shader));
   EXPECT_EQ(count_ifs(t.b.shader), 1u);
   ralloc_free(t.b.shader);
}

/* The load keeps its alignment facts, so the fallback address must satisfy
 * them too: align_mul 16 / align_offset 4 reads zero page + 4. */
TEST(GuardedLoadSelect, FallbackKeepsAlignmentFacts)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "guarded aligned load");
   nir_def *base = push_constant(&b, 64, 0);
   nir_def *index = push_constant(&b, 32, 8);
   nir_def *cond = nir_ult_imm(&b, index, 100);
   nir_push_if(&b, cond);
   nir_def *value = load_global(
      &b, nir_iadd(&b, base, nir_u2u64(&b, nir_imul_imm(&b, index, 16))),
      read_only, 16, 4);
   nir_push_else(&b, NULL);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_pop_if(&b, NULL);
   store_global(&b, nir_if_phi(&b, value, zero), base);

   ASSERT_TRUE(agx_nir_speculate_guarded_loads(b.shader));
   nir_validate_shader(b.shader, "guarded aligned load");
   nir_intrinsic_instr *load = only_global_load(b.shader);
   ASSERT_NE(load, nullptr);
   EXPECT_EQ(nir_intrinsic_align_mul(load), 16u);
   EXPECT_EQ(nir_intrinsic_align_offset(load), 4u);
   expect_guarded_address(load, cond, base, index, AGX_ZERO_PAGE_ADDRESS + 4,
                          true);
   ralloc_free(b.shader);
}
