/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Lower VK_KHR_cooperative_matrix for Apple9 (G16) compute onto the SIMD-group
 * matrix multiply-accumulate (nir_intrinsic_simd_matrix_fmadd_agx), after
 * Joshua Warren's G13 lowering (github.com/joshuaswarren/mesa-1,
 * agx_nir_lower_simdmat.c).
 *
 * An 8x8 matrix is spread over the 32 lanes of a subgroup, two adjacent
 * elements of one row per lane (the layout of Apple's simdgroup_float8x8,
 * see README.apple9-honeykrisp.md):
 *
 *    row = ((lane >> 4) & 1) * 4 + ((lane >> 1) & 3)
 *    col = ((lane >> 3) & 1) * 4 + (lane & 1) * 2     (and col + 1)
 *
 * so a cooperative matrix becomes a per-lane vec2 (an fp32 register pair).
 * Loads and stores are ordinary per-lane memory accesses; muladd is one
 * hardware instruction per 8x8x8 tile. Larger square shapes (N = 16) tile
 * into 8x8 blocks, block (bi, bj) in vec channels [2 sb, 2 sb + 1],
 * sb = bi * N / 8 + bj.
 *
 * The hardware tile needs all 32 lanes, so callers only use this for compute
 * shaders whose workgroup size is a static multiple of 32. Elements are fp32
 * or fp16; fp16 A and B operands widen to fp32 for the multiply, the
 * accumulator must be fp32. Anything else sets *unsupported so the shader is
 * rejected rather than miscompiled.
 */

#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"
#include "agx_compile_apple9.h"

#define CMAT_LANES 32

static unsigned
cmat_len(struct glsl_cmat_description desc)
{
   return (desc.rows * desc.cols) / CMAT_LANES;
}

static const struct glsl_type *
remap_type(struct hash_table *mapping, const struct glsl_type *orig)
{
   struct hash_entry *e = _mesa_hash_table_search(mapping, orig);
   if (e)
      return e->data;

   const struct glsl_type *nt = orig;
   if (glsl_type_is_cmat(orig)) {
      struct glsl_cmat_description d = *glsl_get_cmat_description(orig);
      nt = glsl_vector_type(d.element_type, cmat_len(d));
   } else if (glsl_type_is_array(orig)) {
      const struct glsl_type *el = glsl_get_array_element(orig);
      const struct glsl_type *ne = remap_type(mapping, el);
      if (ne != el)
         nt = glsl_array_type(ne, glsl_get_length(orig),
                              glsl_get_explicit_stride(orig));
   }
   _mesa_hash_table_insert(mapping, orig, (void *)nt);
   return nt;
}

static struct glsl_cmat_description
src_desc(nir_src s)
{
   return *glsl_get_cmat_description(nir_src_as_deref(s)->type);
}

static bool
desc_supported(struct glsl_cmat_description d)
{
   return (d.element_type == GLSL_TYPE_FLOAT ||
           d.element_type == GLSL_TYPE_FLOAT16) &&
          d.rows == d.cols && (d.rows == 8 || d.rows == 16);
}

static nir_def *
load_src(nir_builder *b, nir_src s, struct glsl_cmat_description desc)
{
   nir_deref_instr *d = nir_src_as_deref(s);
   return nir_build_load_deref(b, cmat_len(desc),
                               glsl_base_type_bit_size(desc.element_type),
                               &d->def, 0);
}

static void
store_src(nir_builder *b, nir_src s, nir_def *v)
{
   nir_store_deref(b, nir_src_as_deref(s), v,
                   nir_component_mask(v->num_components));
}

static void
frag_rc(nir_builder *b, nir_def *lane, nir_def **row, nir_def **col)
{
   *row = nir_iadd(b,
                   nir_imul_imm(b, nir_iand_imm(b, nir_ushr_imm(b, lane, 4), 1), 4),
                   nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 3));
   *col = nir_iadd(b,
                   nir_imul_imm(b, nir_iand_imm(b, nir_ushr_imm(b, lane, 3), 1), 4),
                   nir_imul_imm(b, nir_iand_imm(b, lane, 1), 2));
}

static bool
lower_load_store(nir_builder *b, nir_intrinsic_instr *intr)
{
   const bool is_load = intr->intrinsic == nir_intrinsic_cmat_load;
   struct glsl_cmat_description desc = src_desc(intr->src[!is_load]);
   const bool col_major =
      nir_intrinsic_matrix_layout(intr) == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR;
   nir_deref_instr *cmat_deref = nir_src_as_deref(intr->src[!is_load]);
   nir_deref_instr *deref = nir_src_as_deref(intr->src[is_load]);
   nir_def *stride = intr->src[2].ssa;

   /* The Stride operand counts pointer elements, which may be vectors (a
    * vec4 shared tile); the contiguous axis counts matrix components. */
   const unsigned tsz = glsl_base_type_bit_size(desc.element_type) / 8;
   const unsigned nvec = glsl_get_vector_elements(deref->type);
   const unsigned scalar_bytes = glsl_get_bit_size(deref->type) / 8;
   if (scalar_bytes != tsz)
      return false;

   deref = nir_build_deref_cast(b, &deref->def, deref->modes, deref->type,
                                scalar_bytes * nvec);
   nir_deref_instr *sdref =
      nir_build_deref_cast(b, &deref->def, deref->modes,
                           glsl_scalar_type(desc.element_type), tsz);
   /* The matrix variable's derefs are retyped to the vector later in this
    * reverse walk; access it directly so it stays promotable to SSA. */

   nir_def *row, *col;
   frag_rc(b, nir_load_subgroup_invocation(b), &row, &col);

   const unsigned idx_bits = deref->def.bit_size;
   nir_def *str = nir_imul_imm(b, nir_u2uN(b, stride, idx_bits), nvec);
   const unsigned nblk = desc.rows / 8;
   const unsigned len = cmat_len(desc);
   nir_def *vals[8];

   if (!is_load) {
      nir_def *src = load_src(b, intr->src[!is_load], desc);
      for (unsigned e = 0; e < len; e++)
         vals[e] = nir_channel(b, src, e);
   }

   for (unsigned bi = 0; bi < nblk; bi++) {
      for (unsigned bj = 0; bj < nblk; bj++) {
         const unsigned sb = bi * nblk + bj;
         nir_def *lr = nir_u2uN(b, nir_iadd_imm(b, row, bi * 8), idx_bits);
         nir_def *lc = nir_u2uN(b, nir_iadd_imm(b, col, bj * 8), idx_bits);

         if (!col_major) {
            /* (row, col) and (row, col + 1) are adjacent words. */
            nir_def *flat = nir_iadd(b, nir_imul(b, lr, str), lc);
            nir_deref_instr *it0 = nir_build_deref_ptr_as_array(b, sdref, flat);
            nir_deref_instr *it1 =
               nir_build_deref_ptr_as_array(b, sdref, nir_iadd_imm(b, flat, 1));
            if (is_load) {
               vals[sb * 2 + 0] = nir_load_deref(b, it0);
               vals[sb * 2 + 1] = nir_load_deref(b, it1);
            } else {
               nir_store_deref(b, it0, vals[sb * 2 + 0], 0x1);
               nir_store_deref(b, it1, vals[sb * 2 + 1], 0x1);
            }
         } else {
            for (unsigned e = 0; e < 2; e++) {
               nir_def *c = nir_iadd_imm(b, lc, e);
               nir_def *flat = nir_iadd(b, nir_imul(b, c, str), lr);
               nir_deref_instr *it = nir_build_deref_ptr_as_array(b, sdref, flat);
               if (is_load)
                  vals[sb * 2 + e] = nir_load_deref(b, it);
               else
                  nir_store_deref(b, it, vals[sb * 2 + e], 0x1);
            }
         }
      }
   }

   if (is_load)
      nir_store_deref(b, cmat_deref, nir_vec(b, vals, len),
                      nir_component_mask(len));
   nir_instr_remove(&intr->instr);
   return true;
}

static nir_def *
subblk(nir_builder *b, nir_def *v, unsigned sb)
{
   return nir_vec2(b, nir_channel(b, v, sb * 2), nir_channel(b, v, sb * 2 + 1));
}

static bool
lower_muladd(nir_builder *b, nir_intrinsic_instr *intr)
{
   /* src order: dst, A, B, C */
   struct glsl_cmat_description da = src_desc(intr->src[1]);
   struct glsl_cmat_description db = src_desc(intr->src[2]);
   struct glsl_cmat_description dc = src_desc(intr->src[3]);
   if (!desc_supported(da) || !desc_supported(db) || !desc_supported(dc) ||
       dc.element_type != GLSL_TYPE_FLOAT || da.rows != dc.rows ||
       db.rows != dc.rows || nir_intrinsic_saturate(intr))
      return false;

   /* fp16 A or B (an fp16 activation tile against fp32 weights, for example)
    * widens exactly to fp32 for the fp32-accumulator instruction. */
   const unsigned nblk = da.rows / 8;
   nir_def *a = load_src(b, intr->src[1], da);
   nir_def *bb = load_src(b, intr->src[2], db);
   if (a->bit_size == 16)
      a = nir_f2f32(b, a);
   if (bb->bit_size == 16)
      bb = nir_f2f32(b, bb);
   nir_def *c = load_src(b, intr->src[3], dc);

   nir_def *dch[8];
   for (unsigned di = 0; di < nblk; di++) {
      for (unsigned dj = 0; dj < nblk; dj++) {
         const unsigned sb = di * nblk + dj;
         nir_def *acc = subblk(b, c, sb);
         for (unsigned k = 0; k < nblk; k++) {
            acc = nir_simd_matrix_fmadd_agx(b, subblk(b, a, di * nblk + k),
                                            subblk(b, bb, k * nblk + dj), acc);
         }
         dch[sb * 2] = nir_channel(b, acc, 0);
         dch[sb * 2 + 1] = nir_channel(b, acc, 1);
      }
   }

   store_src(b, intr->src[0], nir_vec(b, dch, cmat_len(dc)));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_construct(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_deref_instr *dst = nir_src_as_deref(intr->src[0]);
   struct glsl_cmat_description desc = *glsl_get_cmat_description(dst->type);
   nir_def *r = nir_replicate(b, intr->src[1].ssa, cmat_len(desc));
   nir_store_deref(b, dst, r, nir_component_mask(r->num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_length(nir_builder *b, nir_intrinsic_instr *intr)
{
   struct glsl_cmat_description desc = nir_intrinsic_cmat_desc(intr);
   nir_def_replace(&intr->def, nir_imm_int(b, cmat_len(desc)));
   return true;
}

/* Element-wise operations act on each lane's own elements. They are layout
 * independent, so they apply per vector component. */
static bool
lower_alu_op(nir_builder *b, nir_intrinsic_instr *intr, unsigned arity)
{
   struct glsl_cmat_description d = src_desc(intr->src[1]);
   nir_def *x = load_src(b, intr->src[1], d);
   nir_def *y = arity == 2 ? load_src(b, intr->src[2], src_desc(intr->src[2]))
                : arity == 3 ? intr->src[2].ssa : NULL;
   uint32_t save = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
   nir_def *r = y ? nir_build_alu2(b, nir_intrinsic_alu_op(intr), x, y)
                  : nir_build_alu1(b, nir_intrinsic_alu_op(intr), x);
   b->fp_math_ctrl = save;
   store_src(b, intr->src[0], r);
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_convert(nir_builder *b, nir_intrinsic_instr *intr)
{
   struct glsl_cmat_description dd = src_desc(intr->src[0]);
   struct glsl_cmat_description sd = src_desc(intr->src[1]);
   enum glsl_cmat_use du = dd.use, su = sd.use;
   /* A and accumulator share the row-pair layout, but B is not consumed in
    * a transposed layout either: every use has the same per-lane layout, so
    * use changes are free. Transposes would need a cross-lane shuffle. */
   if (intr->intrinsic == nir_intrinsic_cmat_transpose ||
       dd.element_type != sd.element_type || dd.rows != sd.rows)
      return false;
   (void)du;
   (void)su;
   store_src(b, intr->src[0], load_src(b, intr->src[1], sd));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_impl(nir_function_impl *impl, struct hash_table *tm, bool *failed)
{
   bool progress = false;

   nir_foreach_function_temp_variable(var, impl) {
      const struct glsl_type *nt = remap_type(tm, var->type);
      if (nt != var->type) {
         var->type = nt;
         progress = true;
      }
   }

   nir_builder b = nir_builder_create(impl);
   nir_foreach_block_reverse_safe(block, impl) {
      nir_foreach_instr_reverse_safe(instr, block) {
         b.cursor = nir_before_instr(instr);

         if (instr->type == nir_instr_type_deref) {
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            const struct glsl_type *nt = remap_type(tm, deref->type);
            if (nt != deref->type) {
               deref->type = nt;
               progress = true;
            }
            continue;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         bool ok = true;
         switch (intr->intrinsic) {
         case nir_intrinsic_cmat_length:
            ok = lower_length(&b, intr);
            break;
         case nir_intrinsic_cmat_construct:
            ok = lower_construct(&b, intr);
            break;
         case nir_intrinsic_cmat_load:
         case nir_intrinsic_cmat_store:
            ok = desc_supported(src_desc(
                    intr->src[intr->intrinsic == nir_intrinsic_cmat_store])) &&
                 lower_load_store(&b, intr);
            break;
         case nir_intrinsic_cmat_muladd:
            ok = lower_muladd(&b, intr);
            break;
         case nir_intrinsic_cmat_copy:
            nir_build_copy_deref(&b, intr->src[0].ssa, intr->src[1].ssa);
            nir_instr_remove(instr);
            break;
         case nir_intrinsic_cmat_binary_op:
            ok = lower_alu_op(&b, intr, 2);
            break;
         case nir_intrinsic_cmat_scalar_op:
            ok = lower_alu_op(&b, intr, 3);
            break;
         case nir_intrinsic_cmat_unary_op:
            ok = lower_alu_op(&b, intr, 1);
            break;
         case nir_intrinsic_cmat_extract: {
            nir_def *mat = load_src(&b, intr->src[0], src_desc(intr->src[0]));
            nir_def_replace(&intr->def,
                            nir_vector_extract(&b, mat, intr->src[1].ssa));
            break;
         }
         case nir_intrinsic_cmat_insert: {
            nir_def *mat = load_src(&b, intr->src[2], src_desc(intr->src[2]));
            store_src(&b, intr->src[0],
                      nir_vector_insert(&b, mat, intr->src[1].ssa,
                                        intr->src[3].ssa));
            nir_instr_remove(instr);
            break;
         }
         case nir_intrinsic_cmat_bitcast:
            store_src(&b, intr->src[0],
                      load_src(&b, intr->src[1], src_desc(intr->src[1])));
            nir_instr_remove(instr);
            break;
         case nir_intrinsic_cmat_convert:
         case nir_intrinsic_cmat_transpose:
            ok = lower_convert(&b, intr);
            break;
         default:
            continue;
         }
         if (!ok) {
            *failed = true;
            return progress;
         }
         progress = true;
      }
   }

   nir_progress(progress, impl, nir_metadata_none);
   return progress;
}

bool
agx_nir_lower_apple9_cmat(nir_shader *shader, bool *unsupported)
{
   bool progress = false, failed = false;
   struct hash_table *tm = _mesa_pointer_hash_table_create(NULL);

   nir_foreach_function_impl(impl, shader) {
      progress |= lower_impl(impl, tm, &failed);
      if (failed)
         break;
   }

   _mesa_hash_table_destroy(tm, NULL);
   *unsupported = failed;
   return progress;
}
