// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, Pixels,
    Point, Radians, ScaledPixels, Size, bounds_tree::BoundsTree, point, size,
};
use collections::FxHashMap;
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
    sync::{Arc, Weak},
};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

/// A node in the scene's clip tree. Every primitive references one of these via its
/// `clip_id`. Rectangular clips along a primitive's ancestor chain are folded eagerly
/// into `folded_bounds`, so the common rect-only case costs a single bounds test.
/// Rounded-rect clips can't be folded (the intersection of two rounded rects is not a
/// rounded rect), so each one is retained as a node and linked through
/// `parent_rounded`; shaders walk that chain evaluating one rounded-rect SDF per node.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct ClipNode {
    /// Intersection of all rectangular clip bounds along the ancestor chain,
    /// including the bounding rects of rounded clips.
    pub folded_bounds: Bounds<ScaledPixels>,
    /// The bounds this node's own corner radii apply to. Only meaningful when
    /// `corner_radii` is nonzero.
    pub rounded_bounds: Bounds<ScaledPixels>,
    /// This node's own corner radii. All zeros for purely rectangular clips.
    pub corner_radii: Corners<ScaledPixels>,
    /// The nearest node in the ancestor chain (including this node itself) that
    /// has nonzero corner radii, or [`ClipNode::NONE`]. This is where the
    /// fragment shader starts its rounded-clip walk.
    pub rounded_head: u32,
    /// The nearest strict ancestor with nonzero corner radii, or
    /// [`ClipNode::NONE`]. Links the rounded-clip chain.
    pub parent_rounded: u32,
}

impl ClipNode {
    /// Sentinel indicating the absence of a clip node reference.
    pub const NONE: u32 = u32::MAX;
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    stacking_stack: Vec<StackingOrder>,
    slot_owner: Option<StackingOrder>,
    stacking_boundaries: Vec<(
        bool,
        Option<StackingOrder>,
        Arc<[u32]>,
        u8,
        Arc<[StackingOrder]>,
        u32,
    )>,
    current_source_order: Arc<[u32]>,
    current_stacking_order: Arc<[StackingOrder]>,
    stacking_order_id: u32,
    next_stacking_order_id: u32,
    next_stacking_order_prune: u32,
    paint_stacking_order_cache: Option<(u32, u8, u32, Arc<[StackingOrder]>)>,
    stacking_order_cache: FxHashMap<Vec<StackingOrder>, (u32, Weak<[StackingOrder]>)>,
    cached_paint_order_groups: Vec<PaintOrderGroupId>,
    cached_paint_order_stacking: Vec<Arc<[StackingOrder]>>,
    current_stacking_phase: u8,
    paint_phase: u8,
    paint_plane: usize,
    next_paint_ordinal: u64,
    previous_paint_order: Option<PaintOrderKey>,
    previous_paint_order_group: Option<PaintOrderGroupId>,
    paint_order_requires_reordering: bool,
    pub clips: Vec<ClipNode>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct StackingState {
    pub(crate) stack: Vec<StackingOrder>,
    pub(crate) slot_owner: Option<StackingOrder>,
    pub(crate) source_order: Arc<[u32]>,
    pub(crate) phase: u8,
    pub(crate) plane: usize,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.stacking_stack.clear();
        self.slot_owner = None;
        self.stacking_boundaries.clear();
        self.current_source_order = Arc::from([]);
        self.current_stacking_order = Arc::from([]);
        self.stacking_order_id = 0;
        self.current_stacking_phase = 1;
        self.paint_phase = 0;
        self.paint_plane = 0;
        self.next_paint_ordinal = 0;
        self.previous_paint_order = None;
        self.previous_paint_order_group = None;
        self.paint_order_requires_reordering = false;
        self.refresh_stacking_order();
        self.clips.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    /// Push an atomic CSS stacking context for primitives inserted until it is popped.
    pub fn push_stacking_context(&mut self, z_index: i32, source_order: impl Into<Arc<[u32]>>) {
        self.push_stacking_element(source_order, if z_index < 0 { 0 } else { 2 }, z_index, true);
    }

    /// Push the source-order boundary for one element. Non-context elements
    /// retain their source order without trapping descendant contexts.
    pub fn push_stacking_element(
        &mut self,
        source_order: impl Into<Arc<[u32]>>,
        stacking_phase: u8,
        z_index: i32,
        context: bool,
    ) {
        let source_order = source_order.into();
        let old_source = std::mem::replace(&mut self.current_source_order, source_order.clone());
        let old_slot_owner = self.slot_owner.take();
        self.stacking_boundaries.push((
            context,
            old_slot_owner.clone(),
            old_source,
            self.current_stacking_phase,
            self.current_stacking_order.clone(),
            self.stacking_order_id,
        ));
        self.current_stacking_phase = stacking_phase;
        if context {
            self.stacking_stack.push(StackingOrder {
                context: true,
                z_index,
                source_order,
                phase: stacking_phase,
            });
        } else if stacking_phase == 2 {
            // A positioned element with z-index:auto owns one slot in its
            // enclosing context. Its ordinary descendants stay in that slot;
            // positioned and context descendants replace it when they enter.
            self.slot_owner = Some(StackingOrder {
                context: false,
                z_index: 0,
                source_order,
                phase: stacking_phase,
            });
        } else {
            self.slot_owner = old_slot_owner;
        }
        self.refresh_stacking_order();
    }

    /// Push a source-order boundary without creating a stacking context.
    pub fn push_stacking_container(&mut self, source_order: impl Into<Arc<[u32]>>) {
        self.push_stacking_element(source_order, 1, 0, false);
    }

    /// Pop the most recently pushed stacking boundary.
    pub fn pop_stacking_order(&mut self) {
        if let Some((context, slot_owner, source_order, phase, stacking_order, stacking_order_id)) =
            self.stacking_boundaries.pop()
        {
            if context {
                self.stacking_stack.pop();
            }
            self.current_source_order = source_order;
            self.current_stacking_phase = phase;
            self.slot_owner = slot_owner;
            self.current_stacking_order = stacking_order;
            self.stacking_order_id = stacking_order_id;
        }
    }

    /// Select the decoration, content or outline phase for subsequent primitives.
    pub fn set_paint_phase(&mut self, phase: u8) -> u8 {
        std::mem::replace(&mut self.paint_phase, phase)
    }

    pub(crate) fn set_paint_plane(&mut self, plane: usize) -> usize {
        std::mem::replace(&mut self.paint_plane, plane)
    }

    pub(crate) fn paint_plane(&self) -> usize {
        self.paint_plane
    }

    fn refresh_stacking_order(&mut self) {
        let mut order = self.stacking_stack.clone();
        if let Some(slot_owner) = &self.slot_owner {
            order.push(slot_owner.clone());
        }
        order.push(StackingOrder {
            context: false,
            phase: self.current_stacking_phase,
            z_index: 0,
            source_order: self.current_source_order.clone(),
        });
        let (current, order_id) = self.intern_stacking_order(order);
        self.current_stacking_order = current;
        self.stacking_order_id = order_id;
    }

    fn intern_stacking_order(&mut self, order: Vec<StackingOrder>) -> (Arc<[StackingOrder]>, u32) {
        if self.stacking_order_cache.len() > 4096
            && self.next_stacking_order_id >= self.next_stacking_order_prune
        {
            self.stacking_order_cache
                .retain(|_, (_, cached)| cached.strong_count() > 0);
            self.next_stacking_order_prune = self.next_stacking_order_id.wrapping_add(1024);
        }
        if let Some((order_id, cached)) = self.stacking_order_cache.get_mut(&order) {
            if let Some(current) = cached.upgrade() {
                return (current, *order_id);
            }
            let current: Arc<[StackingOrder]> = Arc::from(order.clone());
            *cached = Arc::downgrade(&current);
            return (current, *order_id);
        }
        let current: Arc<[StackingOrder]> = Arc::from(order.clone());
        let order_id = self.next_stacking_order_id;
        self.next_stacking_order_id = self.next_stacking_order_id.wrapping_add(1);
        self.stacking_order_cache
            .insert(order, (order_id, Arc::downgrade(&current)));
        (current, order_id)
    }

    fn current_paint_stacking_order(&mut self) -> (Arc<[StackingOrder]>, u32) {
        if self.slot_owner.is_none() {
            return (self.current_stacking_order.clone(), self.stacking_order_id);
        }
        if let Some((current_id, phase, order_id, order)) = &self.paint_stacking_order_cache
            && *current_id == self.stacking_order_id
            && *phase == self.paint_phase
        {
            return (order.clone(), *order_id);
        }

        let mut order = self.current_stacking_order.to_vec();
        if let Some(current) = order.last_mut() {
            current.phase = self.paint_phase;
        }
        let (order, order_id) = self.intern_stacking_order(order);
        self.paint_stacking_order_cache = Some((
            self.stacking_order_id,
            self.paint_phase,
            order_id,
            order.clone(),
        ));
        (order, order_id)
    }

    pub(crate) fn current_stacking_order(&self) -> Arc<[StackingOrder]> {
        self.current_stacking_order.clone()
    }

    pub(crate) fn stacking_state(&self) -> StackingState {
        StackingState {
            stack: self.stacking_stack.clone(),
            slot_owner: self.slot_owner.clone(),
            source_order: self.current_source_order.clone(),
            phase: self.current_stacking_phase,
            plane: self.paint_plane,
        }
    }

    pub(crate) fn rebase_stacking_order(
        order: &mut Arc<[StackingOrder]>,
        old_state: &StackingState,
        new_state: &StackingState,
    ) {
        let mut order_vec = order.to_vec();
        let mut old_base = old_state.stack.clone();
        old_base.extend(old_state.slot_owner.iter().cloned());
        let mut new_base = new_state.stack.clone();
        new_base.extend(new_state.slot_owner.iter().cloned());
        if !order_vec.starts_with(&old_base) {
            old_base.clone_from(&old_state.stack);
            new_base.clone_from(&new_state.stack);
        }
        let mut rebased = new_base;
        rebased.extend(
            order_vec
                .iter()
                .skip(old_base.len())
                .cloned()
                .map(|mut entry| {
                    entry.source_order = rebase_source_order(
                        &entry.source_order,
                        &old_state.source_order,
                        &new_state.source_order,
                    );
                    if entry.phase == old_state.phase {
                        entry.phase = new_state.phase;
                    }
                    entry
                }),
        );
        *order = Arc::from(rebased);
    }

    /// Return the current paint-order key for a non-primitive paint registry.
    /// The key follows the same stacking context, source order and paint phase
    /// used by [`Scene::finish`].
    pub fn current_paint_order_key(&self) -> PaintOrderKey {
        let (stacking, phase) = if self.slot_owner.is_some() {
            let mut stacking = self.current_stacking_order.to_vec();
            if let Some(current) = stacking.last_mut() {
                current.phase = self.paint_phase;
            }
            (Arc::from(stacking), 0)
        } else {
            (self.current_stacking_order.clone(), self.paint_phase)
        };
        PaintOrderKey {
            plane: self.paint_plane,
            stacking,
            phase,
            ordinal: self.next_paint_ordinal,
        }
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_clip(&mut self, node: ClipNode) -> u32 {
        let id = self.clips.len() as u32;
        self.clips.push(node);
        id
    }

    pub fn clip_folded_bounds(&self, clip_id: u32) -> Bounds<ScaledPixels> {
        debug_assert!((clip_id as usize) < self.clips.len(), "invalid clip id");
        self.clips
            .get(clip_id as usize)
            .map(|node| node.folded_bounds)
            // Fall back to an effectively unbounded rect (no clipping) rather than
            // panicking on an invalid id.
            .unwrap_or_else(|| Bounds {
                origin: point(ScaledPixels(-1e15), ScaledPixels(-1e15)),
                size: size(ScaledPixels(2e15), ScaledPixels(2e15)),
            })
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        if self.current_stacking_order.is_empty() {
            self.refresh_stacking_order();
        }
        let (stacking, stacking_order_id) = self.current_paint_stacking_order();
        self.insert_primitive_with_stacking(
            primitive.into(),
            stacking,
            stacking_order_id,
            self.paint_plane,
            self.paint_phase,
        );
    }

    fn insert_primitive_with_stacking(
        &mut self,
        mut primitive: Primitive,
        stacking: Arc<[StackingOrder]>,
        stacking_order_id: u32,
        plane: usize,
        phase: u8,
    ) {
        let clipped_bounds = primitive
            .bounds()
            .intersect(&self.clip_folded_bounds(primitive.clip_id()));

        if clipped_bounds.is_empty() {
            return;
        }

        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(*shadow);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(*quad);
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(*underline);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(*sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                self.subpixel_sprites.push(*sprite);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(*sprite);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
        }
        let ordinal = self.next_paint_ordinal;
        self.next_paint_ordinal = self.next_paint_ordinal.wrapping_add(1);
        let paint_order_group = PaintOrderGroupId {
            plane,
            stacking_order: stacking_order_id,
            phase,
        };
        // Ordinals increase within a group, so only a transition can make
        // this primitive sort before the previous one.
        if self.previous_paint_order_group != Some(paint_order_group) {
            let paint_order = PaintOrderKey {
                plane,
                stacking: stacking.clone(),
                phase,
                ordinal,
            };
            if self
                .previous_paint_order
                .as_ref()
                .is_some_and(|previous| paint_order.cmp(previous).is_lt())
            {
                self.paint_order_requires_reordering = true;
            }
            self.previous_paint_order = Some(paint_order);
            self.previous_paint_order_group = Some(paint_order_group);
        }
        self.paint_operations.push(PaintOperation::Primitive {
            primitive,
            stacking,
            stacking_order_id,
            plane,
            phase,
        });
    }

    pub(crate) fn replay(
        &mut self,
        range: Range<usize>,
        prev_scene: &Scene,
        old_state: &StackingState,
        new_state: &StackingState,
    ) {
        // Clip ids in the previous scene's primitives point into `prev_scene.clips`,
        // so the referenced nodes (and their rounded-clip chains) must be copied into
        // this scene and the ids rewritten.
        let mut clip_id_map = FxHashMap::default();
        let mut stacking_order_map = FxHashMap::<usize, (Arc<[StackingOrder]>, u32)>::default();
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive {
                    primitive,
                    stacking,
                    plane,
                    phase,
                    ..
                } => {
                    let mut primitive = primitive.clone();
                    let clip_id =
                        self.import_clip(prev_scene, primitive.clip_id(), &mut clip_id_map);
                    *primitive.clip_id_mut() = clip_id;
                    let stacking_key = Arc::as_ptr(stacking) as *const StackingOrder as usize;
                    let (rebased_stacking, stacking_order_id) =
                        if let Some(rebased) = stacking_order_map.get(&stacking_key) {
                            rebased.clone()
                        } else {
                            let mut rebased = stacking.clone();
                            Self::rebase_stacking_order(&mut rebased, old_state, new_state);
                            let (rebased, order_id) = self.intern_stacking_order(rebased.to_vec());
                            stacking_order_map.insert(stacking_key, (rebased.clone(), order_id));
                            (rebased, order_id)
                        };
                    let plane = if *plane == old_state.plane {
                        new_state.plane
                    } else {
                        *plane
                    };
                    self.insert_primitive_with_stacking(
                        primitive,
                        rebased_stacking,
                        stacking_order_id,
                        plane,
                        *phase,
                    );
                }
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    /// Copy a clip node (and, transitively, its rounded-clip chain) from another scene
    /// into this one, returning the node's id in this scene.
    fn import_clip(
        &mut self,
        prev_scene: &Scene,
        clip_id: u32,
        clip_id_map: &mut FxHashMap<u32, u32>,
    ) -> u32 {
        if clip_id == ClipNode::NONE {
            return ClipNode::NONE;
        }
        if let Some(&new_id) = clip_id_map.get(&clip_id) {
            return new_id;
        }
        let Some(node) = prev_scene.clips.get(clip_id as usize).copied() else {
            debug_assert!(false, "replayed primitive references invalid clip id");
            return ClipNode::NONE;
        };
        // Insert the node before resolving its links so that a `rounded_head`
        // pointing back at the node itself terminates via the map.
        let new_id = self.insert_clip(node);
        clip_id_map.insert(clip_id, new_id);
        let rounded_head = self.import_clip(prev_scene, node.rounded_head, clip_id_map);
        let parent_rounded = self.import_clip(prev_scene, node.parent_rounded, clip_id_map);
        if let Some(node) = self.clips.get_mut(new_id as usize) {
            node.rounded_head = rounded_head;
            node.parent_rounded = parent_rounded;
        }
        new_id
    }

    pub fn finish(&mut self) {
        if self.paint_order_requires_reordering {
            let mut groups = Vec::<Vec<usize>>::new();
            let mut group_ids = Vec::<PaintOrderGroupId>::new();
            let mut group_stacking = Vec::<Arc<[StackingOrder]>>::new();
            let mut group_indices = FxHashMap::<PaintOrderGroupId, usize>::default();
            for (index, operation) in self.paint_operations.iter().enumerate() {
                let PaintOperation::Primitive {
                    stacking,
                    stacking_order_id,
                    plane,
                    phase,
                    ..
                } = operation
                else {
                    continue;
                };
                let id = PaintOrderGroupId {
                    plane: *plane,
                    stacking_order: *stacking_order_id,
                    phase: *phase,
                };
                let group_index = *group_indices.entry(id).or_insert_with(|| {
                    let group_index = groups.len();
                    groups.push(Vec::new());
                    group_ids.push(id);
                    group_stacking.push(stacking.clone());
                    group_index
                });
                groups[group_index].push(index);
            }

            let cached_order_is_current = self.cached_paint_order_groups.len() == groups.len()
                && self.cached_paint_order_stacking.len() == groups.len()
                && self
                    .cached_paint_order_groups
                    .iter()
                    .all(|group| group_indices.contains_key(group));
            if !cached_order_is_current {
                let mut sorted_groups = (0..groups.len()).collect::<Vec<_>>();
                sorted_groups.sort_by(|&left, &right| {
                    let left_id = group_ids[left];
                    let right_id = group_ids[right];
                    left_id
                        .plane
                        .cmp(&right_id.plane)
                        .then_with(|| group_stacking[left].cmp(&group_stacking[right]))
                        .then_with(|| left_id.phase.cmp(&right_id.phase))
                });
                self.cached_paint_order_groups = sorted_groups
                    .iter()
                    .map(|&index| group_ids[index])
                    .collect();
                self.cached_paint_order_stacking = sorted_groups
                    .iter()
                    .map(|&index| group_stacking[index].clone())
                    .collect();
            }

            let ordered = self
                .cached_paint_order_groups
                .iter()
                .flat_map(|group| groups[group_indices[group]].iter().copied())
                .collect::<Vec<_>>();

            let mut bounds = BoundsTree::default();
            self.shadows.clear();
            self.quads.clear();
            self.paths.clear();
            self.underlines.clear();
            self.monochrome_sprites.clear();
            self.subpixel_sprites.clear();
            self.polychrome_sprites.clear();
            self.surfaces.clear();
            let clips = self.clips.clone();
            for index in ordered {
                let PaintOperation::Primitive { primitive, .. } = &mut self.paint_operations[index]
                else {
                    unreachable!()
                };
                let clip_id = primitive.clip_id();
                let clip_bounds = clips
                    .get(clip_id as usize)
                    .map(|node| node.folded_bounds)
                    .unwrap_or_else(|| Bounds {
                        origin: point(ScaledPixels(-1e15), ScaledPixels(-1e15)),
                        size: size(ScaledPixels(2e15), ScaledPixels(2e15)),
                    });
                let order = bounds.insert(primitive.bounds().intersect(&clip_bounds));
                primitive.set_order(order);
                match primitive {
                    Primitive::Shadow(shadow) => self.shadows.push(*shadow),
                    Primitive::Quad(quad) => self.quads.push(*quad),
                    Primitive::Path(path) => {
                        path.id = PathId(self.paths.len());
                        self.paths.push(path.clone());
                    }
                    Primitive::Underline(underline) => self.underlines.push(*underline),
                    Primitive::MonochromeSprite(sprite) => self.monochrome_sprites.push(*sprite),
                    Primitive::SubpixelSprite(sprite) => self.subpixel_sprites.push(*sprite),
                    Primitive::PolychromeSprite(sprite) => self.polychrome_sprites.push(*sprite),
                    Primitive::Surface(surface) => self.surfaces.push(surface.clone()),
                }
            }
        }
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites.sort_by_key(|sprite| sprite.order);
        self.subpixel_sprites.sort_by_key(|sprite| sprite.order);
        self.polychrome_sprites.sort_by_key(|sprite| sprite.order);
        self.surfaces.sort_by_key(|surface| surface.order);
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
        }
    }
}

fn rebase_source_order(source_order: &[u32], old: &[u32], new: &[u32]) -> Arc<[u32]> {
    if source_order.starts_with(old) {
        Arc::from(
            new.iter()
                .chain(source_order[old.len()..].iter())
                .copied()
                .collect::<Vec<_>>(),
        )
    } else {
        Arc::from(source_order)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct StackingOrder {
    phase: u8,
    z_index: i32,
    source_order: Arc<[u32]>,
    pub(crate) context: bool,
}

/// A paint-order position suitable for resolving overlap in a non-scene registry.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PaintOrderKey {
    plane: usize,
    stacking: Arc<[StackingOrder]>,
    phase: u8,
    ordinal: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct PaintOrderGroupId {
    plane: usize,
    stacking_order: u32,
    phase: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
}

pub(crate) enum PaintOperation {
    Primitive {
        primitive: Primitive,
        stacking: Arc<[StackingOrder]>,
        stacking_order_id: u32,
        plane: usize,
        phase: u8,
    },
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
}

#[expect(missing_docs)]
impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
        }
    }

    pub fn clip_id(&self) -> u32 {
        match self {
            Primitive::Shadow(shadow) => shadow.clip_id,
            Primitive::Quad(quad) => quad.clip_id,
            Primitive::Path(path) => path.clip_id,
            Primitive::Underline(underline) => underline.clip_id,
            Primitive::MonochromeSprite(sprite) => sprite.clip_id,
            Primitive::SubpixelSprite(sprite) => sprite.clip_id,
            Primitive::PolychromeSprite(sprite) => sprite.clip_id,
            Primitive::Surface(surface) => surface.clip_id,
        }
    }

    pub fn clip_id_mut(&mut self) -> &mut u32 {
        match self {
            Primitive::Shadow(shadow) => &mut shadow.clip_id,
            Primitive::Quad(quad) => &mut quad.clip_id,
            Primitive::Path(path) => &mut path.clip_id,
            Primitive::Underline(underline) => &mut underline.clip_id,
            Primitive::MonochromeSprite(sprite) => &mut sprite.clip_id,
            Primitive::SubpixelSprite(sprite) => &mut sprite.clip_id,
            Primitive::PolychromeSprite(sprite) => &mut sprite.clip_id,
            Primitive::Surface(surface) => &mut surface.clip_id,
        }
    }

    fn set_order(&mut self, order: DrawOrder) {
        match self {
            Primitive::Shadow(shadow) => shadow.order = order,
            Primitive::Quad(quad) => quad.order = order,
            Primitive::Path(path) => path.order = order,
            Primitive::Underline(underline) => underline.order = order,
            Primitive::MonochromeSprite(sprite) => sprite.order = order,
            Primitive::SubpixelSprite(sprite) => sprite.order = order,
            Primitive::PolychromeSprite(sprite) => sprite.order = order,
            Primitive::Surface(surface) => surface.order = order,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub clip_id: u32,
    pub pad: u32, // keep size a multiple of 8 bytes for WGSL struct layout
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub clip_id: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub color: Hsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
    pub clip_id: u32,
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub clip_id: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub clip_id: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub clip_id: u32,
    pub nearest_neighbor: PaddedBool32,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub clip_id: u32,
    #[cfg(target_os = "macos")]
    pub source: PaintSurfaceSource,
    /// Type-erased GPU texture (`Arc<wgpu::Texture>`). Ported from gpui-ce
    /// ([#39](https://github.com/gpui-ce/gpui-ce/commit/6d043b22e477)).
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    pub texture: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    pub texture_size: Size<crate::DevicePixels>,
}

/// The macOS renderer's surface source. Texture ownership stays type-erased at
/// the scene boundary so that GPUI does not depend on a particular Metal bridge.
#[cfg(target_os = "macos")]
#[derive(Clone)]
#[allow(missing_docs)]
pub enum PaintSurfaceSource {
    ImageBuffer(core_video::pixel_buffer::CVPixelBuffer),
    Texture {
        texture: std::sync::Arc<dyn std::any::Any + Send + Sync>,
        texture_size: Size<crate::DevicePixels>,
    },
}

impl std::fmt::Debug for PaintSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("PaintSurface");
        debug
            .field("order", &self.order)
            .field("bounds", &self.bounds)
            .field("clip_id", &self.clip_id);
        #[cfg(target_os = "macos")]
        match &self.source {
            PaintSurfaceSource::ImageBuffer(image_buffer) => {
                debug.field("image_buffer", image_buffer);
            }
            PaintSurfaceSource::Texture { texture_size, .. } => {
                debug.field("texture_size", texture_size);
            }
        }
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        debug.field("texture_size", &self.texture_size);
        debug.finish_non_exhaustive()
    }
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    /// The folded rectangular clip that was active when the path was painted. Paths
    /// aren't uploaded to the GPU as-is, so unlike other primitives they retain their
    /// rectangular mask inline in addition to referencing a clip node via `clip_id`
    /// (which carries the rounded-clip chain, if any).
    pub content_mask: ContentMask<P>,
    pub clip_id: u32,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            clip_id: 0,
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            clip_id: self.clip_id,
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color.scale(factor),
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: crate::size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    fn quad(clip_id: u32, quad_bounds: Bounds<ScaledPixels>) -> Quad {
        Quad {
            order: 0,
            border_style: Default::default(),
            bounds: quad_bounds,
            clip_id,
            pad: 0,
            background: Default::default(),
            border_color: Default::default(),
            corner_radii: Default::default(),
            border_widths: Default::default(),
        }
    }

    #[test]
    fn sprites_sharing_an_order_keep_paint_order() {
        let mut scene = Scene::default();
        let clip_id = scene.insert_clip(ClipNode {
            folded_bounds: bounds(0., 0., 100., 100.),
            rounded_bounds: Default::default(),
            corner_radii: Default::default(),
            rounded_head: ClipNode::NONE,
            parent_rounded: ClipNode::NONE,
        });
        let glyph = |x: f32, tile_id: u32| MonochromeSprite {
            order: 0,
            clip_id,
            bounds: bounds(x, 0., 10., 18.),
            color: Default::default(),
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: crate::AtlasTextureKind::Monochrome,
                },
                tile_id: crate::TileId(tile_id),
                padding: 0,
                bounds: Default::default(),
            },
            transformation: TransformationMatrix::unit(),
        };

        // Overlapping glyphs in one layer share an order. The glyph painted
        // second has the lower tile id, as a reused atlas can allocate it.
        scene.push_layer(bounds(0., 0., 100., 100.));
        scene.insert_primitive(glyph(0., 2));
        scene.insert_primitive(glyph(8., 1));
        scene.pop_layer();
        scene.finish();

        let painted: Vec<f32> = scene
            .monochrome_sprites
            .iter()
            .map(|sprite| sprite.bounds.origin.x.0)
            .collect();
        assert_eq!(painted, [0., 8.]);
    }

    #[test]
    fn scene_finish_orders_primitives_by_stacking_context_and_keeps_ties_in_source_order() {
        let mut scene = Scene::default();
        let clip_id = scene.insert_clip(ClipNode {
            folded_bounds: bounds(0., 0., 100., 100.),
            rounded_bounds: Default::default(),
            corner_radii: Default::default(),
            rounded_head: ClipNode::NONE,
            parent_rounded: ClipNode::NONE,
        });
        let glyph = |tile_id| MonochromeSprite {
            order: 0,
            clip_id,
            bounds: bounds(0., 0., 10., 18.),
            color: Default::default(),
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: crate::AtlasTextureKind::Monochrome,
                },
                tile_id: crate::TileId(tile_id),
                padding: 0,
                bounds: Default::default(),
            },
            transformation: TransformationMatrix::unit(),
        };

        scene.push_stacking_context(2, vec![0]);
        scene.insert_primitive(glyph(3));
        scene.pop_stacking_order();
        scene.push_stacking_context(-1, vec![1]);
        scene.insert_primitive(glyph(1));
        scene.pop_stacking_order();
        scene.push_stacking_context(2, vec![2]);
        scene.insert_primitive(glyph(4));
        scene.pop_stacking_order();
        scene.push_stacking_context(2, vec![0]);
        scene.insert_primitive(glyph(5));
        scene.pop_stacking_order();
        scene.finish();

        let painted: Vec<u32> = scene
            .monochrome_sprites
            .iter()
            .map(|sprite| sprite.tile.tile_id.0)
            .collect();
        assert_eq!(painted, [1, 3, 5, 4]);
    }

    #[test]
    fn scene_finish_skips_reordering_when_paint_order_is_already_monotonic() {
        let mut scene = Scene::default();
        let clip_id = scene.insert_clip(ClipNode {
            folded_bounds: bounds(0., 0., 100., 100.),
            rounded_bounds: Default::default(),
            corner_radii: Default::default(),
            rounded_head: ClipNode::NONE,
            parent_rounded: ClipNode::NONE,
        });
        scene.push_stacking_container(vec![0]);
        scene.insert_primitive(quad(clip_id, bounds(0., 0., 10., 10.)));
        scene.pop_stacking_order();
        scene.push_stacking_container(vec![1]);
        scene.insert_primitive(quad(clip_id, bounds(0., 0., 10., 10.)));
        scene.pop_stacking_order();

        assert!(!scene.paint_order_requires_reordering);
        scene.finish();
        let source_orders = scene
            .paint_operations
            .iter()
            .filter_map(|operation| match operation {
                PaintOperation::Primitive { stacking, .. } => {
                    Some(stacking.last().unwrap().source_order.clone())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            source_orders
                .iter()
                .map(|order| order.as_ref().to_vec())
                .collect::<Vec<_>>(),
            [vec![0], vec![1]]
        );
        assert!(scene.cached_paint_order_groups.is_empty());
    }

    #[test]
    fn consecutive_primitives_in_one_group_reuse_the_previous_order_key() {
        let mut scene = Scene::default();
        let clip_id = scene.insert_clip(ClipNode {
            folded_bounds: bounds(0., 0., 100., 100.),
            rounded_bounds: Default::default(),
            corner_radii: Default::default(),
            rounded_head: ClipNode::NONE,
            parent_rounded: ClipNode::NONE,
        });

        scene.insert_primitive(quad(clip_id, bounds(0., 0., 10., 10.)));
        scene.insert_primitive(quad(clip_id, bounds(10., 0., 10., 10.)));

        assert_eq!(scene.next_paint_ordinal, 2);
        assert_eq!(
            scene.previous_paint_order.as_ref().unwrap().ordinal,
            0,
            "the unchanged group should not rebuild the comparison key"
        );
    }

    #[test]
    fn scene_finish_reuses_sorted_stacking_groups_after_clear() {
        let mut scene = Scene::default();
        let record_primitives = |scene: &mut Scene| {
            let clip_id = scene.insert_clip(ClipNode {
                folded_bounds: bounds(0., 0., 100., 100.),
                rounded_bounds: Default::default(),
                corner_radii: Default::default(),
                rounded_head: ClipNode::NONE,
                parent_rounded: ClipNode::NONE,
            });
            scene.push_stacking_context(2, vec![0]);
            scene.insert_primitive(quad(clip_id, bounds(0., 0., 10., 10.)));
            scene.pop_stacking_order();
            scene.push_stacking_context(-1, vec![1]);
            scene.insert_primitive(quad(clip_id, bounds(0., 0., 10., 10.)));
            scene.pop_stacking_order();
        };

        record_primitives(&mut scene);
        scene.finish();
        let cached_order = scene.cached_paint_order_groups.clone();
        assert_eq!(cached_order.len(), 2);

        scene.clear();
        record_primitives(&mut scene);
        scene.finish();
        assert_eq!(scene.cached_paint_order_groups, cached_order);
    }

    #[test]
    fn scene_finish_sorts_each_contexts_decoration_content_and_outline_phases() {
        let mut scene = Scene::default();
        let clip_id = scene.insert_clip(ClipNode {
            folded_bounds: bounds(0., 0., 100., 100.),
            rounded_bounds: Default::default(),
            corner_radii: Default::default(),
            rounded_head: ClipNode::NONE,
            parent_rounded: ClipNode::NONE,
        });
        let glyph = |tile_id| MonochromeSprite {
            order: 0,
            clip_id,
            bounds: bounds(0., 0., 10., 18.),
            color: Default::default(),
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: crate::AtlasTextureKind::Monochrome,
                },
                tile_id: crate::TileId(tile_id),
                padding: 0,
                bounds: Default::default(),
            },
            transformation: TransformationMatrix::unit(),
        };

        scene.push_stacking_context(0, vec![0]);
        scene.set_paint_phase(2);
        scene.insert_primitive(glyph(3));
        scene.set_paint_phase(1);
        scene.insert_primitive(glyph(2));
        scene.set_paint_phase(0);
        scene.insert_primitive(glyph(1));
        scene.pop_stacking_order();
        scene.finish();

        let painted: Vec<u32> = scene
            .monochrome_sprites
            .iter()
            .map(|sprite| sprite.tile.tile_id.0)
            .collect();
        assert_eq!(painted, [1, 2, 3]);
    }

    #[test]
    fn auto_positioned_container_keeps_its_in_flow_child_inside_its_slot() {
        let mut scene = Scene::default();
        let clip_id = scene.insert_clip(ClipNode {
            folded_bounds: bounds(0., 0., 100., 100.),
            rounded_bounds: Default::default(),
            corner_radii: Default::default(),
            rounded_head: ClipNode::NONE,
            parent_rounded: ClipNode::NONE,
        });
        let glyph = |tile_id| MonochromeSprite {
            order: 0,
            clip_id,
            bounds: bounds(0., 0., 10., 18.),
            color: Default::default(),
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: crate::AtlasTextureKind::Monochrome,
                },
                tile_id: crate::TileId(tile_id),
                padding: 0,
                bounds: Default::default(),
            },
            transformation: TransformationMatrix::unit(),
        };

        // This unrelated context activates Scene::finish's ordering pass.
        scene.push_stacking_context(-1, vec![1]);
        scene.insert_primitive(glyph(1));
        scene.pop_stacking_order();

        // The parent's background and in-flow child's background share the
        // auto-positioned parent's slot. Decoration phase and retained source
        // order keep the parent behind its child.
        scene.push_stacking_element(vec![0], 2, 0, false);
        scene.set_paint_phase(0);
        scene.insert_primitive(glyph(2));
        scene.push_stacking_element(vec![0, 0], 1, 0, false);
        scene.set_paint_phase(0);
        scene.insert_primitive(glyph(3));
        scene.set_paint_phase(1);
        scene.insert_primitive(glyph(4));
        scene.pop_stacking_order();
        scene.pop_stacking_order();
        scene.finish();

        let painted: Vec<u32> = scene
            .monochrome_sprites
            .iter()
            .map(|sprite| sprite.tile.tile_id.0)
            .collect();
        assert_eq!(painted, [1, 2, 3, 4]);
    }

    #[test]
    fn deferred_paint_plane_stays_above_document_z_levels() {
        let mut scene = Scene::default();
        let clip_id = scene.insert_clip(ClipNode {
            folded_bounds: bounds(0., 0., 100., 100.),
            rounded_bounds: Default::default(),
            corner_radii: Default::default(),
            rounded_head: ClipNode::NONE,
            parent_rounded: ClipNode::NONE,
        });
        let glyph = |tile_id| MonochromeSprite {
            order: 0,
            clip_id,
            bounds: bounds(0., 0., 10., 18.),
            color: Default::default(),
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: crate::AtlasTextureKind::Monochrome,
                },
                tile_id: crate::TileId(tile_id),
                padding: 0,
                bounds: Default::default(),
            },
            transformation: TransformationMatrix::unit(),
        };

        scene.push_stacking_context(100, vec![0]);
        scene.insert_primitive(glyph(1));
        scene.pop_stacking_order();
        let old_plane = scene.set_paint_plane(1);
        scene.push_stacking_context(-1, vec![1]);
        scene.insert_primitive(glyph(2));
        scene.pop_stacking_order();
        scene.set_paint_plane(old_plane);
        scene.finish();

        let painted: Vec<u32> = scene
            .monochrome_sprites
            .iter()
            .map(|sprite| sprite.tile.tile_id.0)
            .collect();
        assert_eq!(painted, [1, 2]);
    }

    #[test]
    fn replay_remaps_clip_chains() {
        let mut prev_scene = Scene::default();
        // A rectangular root and a rounded child clip, forming a chain.
        let root = prev_scene.insert_clip(ClipNode {
            folded_bounds: bounds(0., 0., 100., 100.),
            rounded_bounds: Default::default(),
            corner_radii: Default::default(),
            rounded_head: ClipNode::NONE,
            parent_rounded: ClipNode::NONE,
        });
        let rounded = prev_scene.insert_clip(ClipNode {
            folded_bounds: bounds(10., 10., 50., 50.),
            rounded_bounds: bounds(10., 10., 50., 50.),
            corner_radii: Corners::all(ScaledPixels(8.)),
            rounded_head: 1,
            parent_rounded: ClipNode::NONE,
        });
        assert_eq!(rounded, 1);
        prev_scene.insert_primitive(quad(root, bounds(0., 0., 100., 100.)));
        prev_scene.insert_primitive(quad(rounded, bounds(10., 10., 50., 50.)));

        // Replaying into a scene that already has clip nodes must rewrite the
        // replayed primitives' clip ids.
        let mut next_scene = Scene::default();
        for _ in 0..3 {
            next_scene.insert_clip(ClipNode {
                folded_bounds: bounds(0., 0., 1., 1.),
                rounded_bounds: Default::default(),
                corner_radii: Default::default(),
                rounded_head: ClipNode::NONE,
                parent_rounded: ClipNode::NONE,
            });
        }
        next_scene.replay(
            0..prev_scene.len(),
            &prev_scene,
            &prev_scene.stacking_state(),
            &next_scene.stacking_state(),
        );

        assert_eq!(next_scene.quads.len(), 2);
        let replayed_root_quad = &next_scene.quads[0];
        let replayed_rounded_quad = &next_scene.quads[1];

        let root_node = next_scene.clips[replayed_root_quad.clip_id as usize];
        assert_eq!(root_node.folded_bounds, bounds(0., 0., 100., 100.));
        assert_eq!(root_node.rounded_head, ClipNode::NONE);

        let rounded_node = next_scene.clips[replayed_rounded_quad.clip_id as usize];
        assert_eq!(rounded_node.folded_bounds, bounds(10., 10., 50., 50.));
        assert_eq!(rounded_node.corner_radii, Corners::all(ScaledPixels(8.)));
        // The self-referential rounded head must point at the node's new id.
        assert_eq!(rounded_node.rounded_head, replayed_rounded_quad.clip_id);
        assert_eq!(rounded_node.parent_rounded, ClipNode::NONE);
    }

    #[test]
    fn replay_rebases_cached_stacking_paths_to_the_current_parent() {
        let mut previous = Scene::default();
        let clip_id = previous.insert_clip(ClipNode {
            folded_bounds: bounds(0., 0., 100., 100.),
            rounded_bounds: Default::default(),
            corner_radii: Default::default(),
            rounded_head: ClipNode::NONE,
            parent_rounded: ClipNode::NONE,
        });
        previous.push_stacking_context(1, vec![0]);
        let old_state = previous.stacking_state();
        previous.push_stacking_context(5, vec![0, 0]);
        previous.insert_primitive(quad(clip_id, bounds(1., 1., 10., 10.)));
        previous.pop_stacking_order();
        previous.pop_stacking_order();

        let mut next = Scene::default();
        next.insert_clip(previous.clips[0]);
        next.push_stacking_context(3, vec![4]);
        let new_state = next.stacking_state();
        next.replay(0..previous.len(), &previous, &old_state, &new_state);

        let PaintOperation::Primitive { stacking, .. } = next.paint_operations.last().unwrap()
        else {
            panic!("replayed primitive should remain the final paint operation");
        };
        assert_eq!(stacking.len(), 3);
        assert_eq!(stacking[0].z_index, 3);
        assert_eq!(stacking[0].source_order.as_ref(), [4]);
        assert_eq!(stacking[1].z_index, 5);
        assert_eq!(stacking[1].source_order.as_ref(), [4, 0]);
        assert_eq!(stacking[2].source_order.as_ref(), [4, 0]);
    }
}
