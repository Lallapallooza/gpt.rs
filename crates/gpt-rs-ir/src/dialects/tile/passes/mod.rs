//! The passes of the `tile` dialect.

mod canonicalize;
mod form_groups;
mod inline_static_maps;
mod name_target;
mod pad_with_identity;
mod place_epilogues;
mod plan_layouts;
mod schedule;
mod seal_dispatches;
mod select_atoms;
mod split_lane_folds;
mod walk_order_layouts;
mod widen_reused_maps;
mod winograd_conv;

pub use self::form_groups::FormGroups;
pub use self::inline_static_maps::InlineStaticMaps;
pub use self::name_target::NameTarget;
pub use self::pad_with_identity::PadWithIdentity;
pub use self::place_epilogues::PlaceEpilogues;
pub use self::plan_layouts::PlanLayouts;
pub use self::schedule::Schedule;
pub use self::seal_dispatches::SealDispatches;
pub use self::select_atoms::SelectAtoms;
pub use self::split_lane_folds::SplitLaneFolds;
pub use self::walk_order_layouts::WalkOrderLayouts;
pub use self::widen_reused_maps::WidenReusedMaps;
pub use self::winograd_conv::WinogradConv;
