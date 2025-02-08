#![feature(specialization)]
#![feature(generic_const_exprs)]
#![feature(let_chains)]
#![feature(btree_extract_if)]
#![feature(iterator_try_collect)]
#![feature(extract_if)]
#![feature(trait_alias)]
#![feature(slice_swap_unchecked)]
/* this feature is necessary to constrain matrices,
however, a bug inside it prevents using type aliases for other types
*/
 #![feature(lazy_type_alias)]
// visual separator
#![allow(incomplete_features, reason = "we need nightly features")]
#![allow(dead_code, reason = "to be removed later")] // REMOVE THIS LATER

// - - -
/* clippy begin */
#![warn(
    // regular groups
    clippy::all, // just in case
    clippy::nursery,
    clippy::pedantic,
    clippy::style,
    clippy::complexity,
    clippy::perf,

    // debugging remnants
    clippy::dbg_macro,
    //clippy::expect_used,
    //clippy::panic,
    clippy::print_stderr,
    //clippy::print_stdout,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::use_debug,
    //clippy::unwrap_used,

    // restricions
    clippy::arithmetic_side_effects,
    clippy::clone_on_ref_ptr,
    clippy::else_if_without_else,
    clippy::float_cmp_const,
    clippy::fn_to_numeric_cast_any,
    clippy::if_then_some_else_none,
    clippy::let_underscore_must_use,
    clippy::map_err_ignore,
    clippy::missing_assert_message,
    clippy::multiple_inherent_impl,
    clippy::multiple_unsafe_ops_per_block,
    clippy::mutex_atomic,
    clippy::pattern_type_mismatch,
    clippy::rc_buffer,
    clippy::rc_mutex,
    clippy::same_name_method,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::shadow_unrelated,
    clippy::single_char_lifetime_names,
    clippy::str_to_string,
    clippy::string_slice,
    clippy::string_to_string,
    clippy::verbose_file_reads,

    // style
    clippy::decimal_literal_representation,
    clippy::format_push_string,
    clippy::tests_outside_test_module,
)]
#![deny(
    clippy::correctness,

    // restrictions
    clippy::as_conversions,
    clippy::allow_attributes_without_reason,
    clippy::default_numeric_fallback,
    clippy::exit,
    clippy::indexing_slicing,
    clippy::lossy_float_literal,
    clippy::mem_forget,
    clippy::string_add,
    clippy::try_err,

    // style
    clippy::empty_structs_with_brackets,
    clippy::impl_trait_in_params,
    clippy::rest_pat_in_fully_bound_structs,
    clippy::self_named_module_files,
    clippy::semicolon_inside_block,
    clippy::unnecessary_self_imports,
    clippy::unneeded_field_pattern,
    clippy::unseparated_literal_suffix,
)]
#![allow(clippy::module_name_repetitions, reason = "this is a dumb rule")]
/* clippy end */

pub mod vec2d;

use vec2d::Vec2d;

fn main() {
    let dupa = vec![vec![1,2,3], vec![4,5,6], vec![7,8,9]].into_iter().map(|col| col.into_iter());
    let vec2d = Vec2d::from_cols(dupa).unwrap();
    println!("{:?}", vec2d);
    println!("{}", vec2d);
}

