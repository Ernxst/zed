use crate::{
    AbsoluteLength, App, Bounds, CalcLength, DefiniteLength, Edges, GridTemplate,
    GridTemplateComponent, GridTrack, GridTrackMax, GridTrackMin, Length, Pixels, Point, Size,
    Style, Window, size,
    util::{
        ceil_to_device_pixel, round_half_toward_zero, round_stroke_to_device_pixel,
        round_to_device_pixel,
    },
};
use collections::{FxHashMap, FxHashSet};
use std::{fmt::Debug, ops::Range};
use taffy::{
    Cache, CacheTree, Display, compute_block_layout, compute_cached_layout, compute_flexbox_layout,
    compute_grid_layout, compute_hidden_layout, compute_leaf_layout, compute_root_layout,
    geometry::{Point as TaffyPoint, Rect as TaffyRect, Size as TaffySize},
    style::AvailableSpace as TaffyAvailableSpace,
    tree::{
        Layout, LayoutBlockContainer, LayoutFlexboxContainer, LayoutGridContainer, LayoutInput,
        LayoutOutput, LayoutPartialTree, NodeId, TraversePartialTree,
    },
};

#[cfg(feature = "stacker")]
type StackSafe<T> = stacksafe::StackSafe<T>;
#[cfg(not(feature = "stacker"))]
type StackSafe<T> = T;

type NodeMeasureFn = StackSafe<
    Box<
        dyn FnMut(
            Size<Option<Pixels>>,
            Size<AvailableSpace>,
            &mut Window,
            &mut App,
        ) -> MeasuredLayout,
    >,
>;

#[derive(Clone, Copy)]
struct MeasuredLayout {
    size: Size<Pixels>,
    first_baseline: Option<Pixels>,
}

enum NodeContext {
    Dynamic(NodeMeasureFn),
    #[cfg(test)]
    Fixed(MeasuredLayout),
}

struct LayoutNode {
    style: taffy::style::Style,
    // Taffy's calc encoding retains an aligned pointer. These Arcs keep the
    // expressions alive for the whole layout pass that dereferences it.
    calc_lengths: Vec<CalcLength>,
    children: Vec<NodeId>,
    parent: Option<NodeId>,
    context: Option<NodeContext>,
    cache: Cache,
    layout: Layout,
}

/// GPUI's Taffy storage and low-level algorithm adaptor.
///
/// `TaffyTree::compute_layout_with_measure` accepts size-only measurements, so
/// it cannot carry a shaped text baseline into `LayoutOutput`. Keeping the
/// nodes here lets measured leaves return that metadata through Taffy's native
/// flex, grid, and block algorithms without a renderer-side correction pass.
#[derive(Default)]
struct GpuiTaffyTree {
    nodes: Vec<LayoutNode>,
}

impl GpuiTaffyTree {
    fn clear(&mut self) {
        self.nodes.clear();
    }

    fn new_node(
        &mut self,
        style: taffy::style::Style,
        calc_lengths: Vec<CalcLength>,
        children: &[LayoutId],
        context: Option<NodeContext>,
    ) -> LayoutId {
        let id = NodeId::from(self.nodes.len() as u64);
        self.nodes.push(LayoutNode {
            style,
            calc_lengths,
            children: children.iter().map(|child| child.0).collect(),
            parent: None,
            context,
            cache: Cache::new(),
            layout: Layout::new(),
        });
        for child in children {
            self.node_mut(child.0).parent = Some(id);
        }
        LayoutId(id)
    }

    fn node(&self, id: NodeId) -> &LayoutNode {
        &self.nodes[u64::from(id) as usize]
    }

    fn node_mut(&mut self, id: NodeId) -> &mut LayoutNode {
        &mut self.nodes[u64::from(id) as usize]
    }

    fn children(&self, id: NodeId) -> Vec<NodeId> {
        self.node(id).children.clone()
    }

    fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.node(id).parent
    }

    fn style(&self, id: NodeId) -> &taffy::style::Style {
        &self.node(id).style
    }

    fn set_style(&mut self, id: NodeId, style: taffy::style::Style) {
        self.node_mut(id).style = style;
        self.clear_cache_upwards(id);
    }

    fn layout(&self, id: NodeId) -> &Layout {
        &self.node(id).layout
    }

    fn clear_cache_upwards(&mut self, mut id: NodeId) {
        loop {
            self.node_mut(id).cache.clear();
            let Some(parent) = self.parent(id) else {
                break;
            };
            id = parent;
        }
    }
}

struct LayoutRun<'a> {
    tree: &'a mut GpuiTaffyTree,
    window: Option<&'a mut Window>,
    cx: Option<&'a mut App>,
    scale_factor: f32,
}

impl TraversePartialTree for LayoutRun<'_> {
    type ChildIter<'a>
        = std::iter::Copied<std::slice::Iter<'a, NodeId>>
    where
        Self: 'a;

    fn child_ids(&self, parent_node_id: NodeId) -> Self::ChildIter<'_> {
        self.tree.node(parent_node_id).children.iter().copied()
    }

    fn child_count(&self, parent_node_id: NodeId) -> usize {
        self.tree.node(parent_node_id).children.len()
    }

    fn get_child_id(&self, parent_node_id: NodeId, child_index: usize) -> NodeId {
        self.tree.node(parent_node_id).children[child_index]
    }
}

impl CacheTree for LayoutRun<'_> {
    fn cache_get(&self, node_id: NodeId, input: &LayoutInput) -> Option<LayoutOutput> {
        self.tree.node(node_id).cache.get(input)
    }

    fn cache_store(&mut self, node_id: NodeId, input: &LayoutInput, output: LayoutOutput) {
        self.tree.node_mut(node_id).cache.store(input, output);
    }

    fn cache_clear(&mut self, node_id: NodeId) {
        self.tree.node_mut(node_id).cache.clear();
    }
}

impl LayoutRun<'_> {
    fn compute_node_layout(
        &mut self,
        node_id: NodeId,
        inputs: LayoutInput,
        block_context: Option<&mut taffy::BlockContext>,
    ) -> LayoutOutput {
        if inputs.run_mode == taffy::RunMode::PerformHiddenLayout {
            return compute_hidden_layout(self, node_id);
        }

        compute_cached_layout(self, node_id, inputs, |tree, node_id, inputs| {
            let style = tree.tree.style(node_id).clone();
            let has_children = tree.child_count(node_id) > 0;

            if style.display == Display::None {
                return compute_hidden_layout(tree, node_id);
            }

            if !has_children {
                let mut first_baseline = None;
                let mut output = compute_leaf_layout(
                    inputs,
                    &style,
                    |_, _| 0.0,
                    |known_dimensions, available_space| {
                        let known_dimensions = Size {
                            width: known_dimensions
                                .width
                                .map(|dimension| Pixels(dimension / tree.scale_factor)),
                            height: known_dimensions
                                .height
                                .map(|dimension| Pixels(dimension / tree.scale_factor)),
                        };
                        let untransform = |space: TaffyAvailableSpace| match space {
                            TaffyAvailableSpace::Definite(pixels) => {
                                AvailableSpace::Definite(Pixels(pixels / tree.scale_factor))
                            }
                            TaffyAvailableSpace::MinContent => AvailableSpace::MinContent,
                            TaffyAvailableSpace::MaxContent => AvailableSpace::MaxContent,
                        };
                        let available_space = size(
                            untransform(available_space.width),
                            untransform(available_space.height),
                        );
                        let measured = match tree.tree.node_mut(node_id).context.as_mut() {
                            Some(NodeContext::Dynamic(measure)) => measure(
                                known_dimensions,
                                available_space,
                                tree.window
                                    .as_deref_mut()
                                    .expect("window required to measure"),
                                tree.cx.as_deref_mut().expect("app required to measure"),
                            ),
                            #[cfg(test)]
                            Some(NodeContext::Fixed(measured)) => *measured,
                            None => MeasuredLayout {
                                size: Size::default(),
                                first_baseline: None,
                            },
                        };
                        // Taffy aligns the baseline metadata by moving the child's box. Both
                        // that box origin and the painted glyph baseline ultimately land on
                        // whole device pixels, so align the same quantized metric here.
                        first_baseline = measured
                            .first_baseline
                            .map(|baseline| round_to_device_pixel(baseline.0, tree.scale_factor));
                        snap_measured_size_to_device_pixels(measured.size, tree.scale_factor).into()
                    },
                );
                output.first_baselines.y = first_baseline;
                return output;
            }

            match style.display {
                Display::None => unreachable!(),
                Display::Block => compute_block_layout(tree, node_id, inputs, block_context),
                Display::FlowRoot => compute_block_layout(tree, node_id, inputs, None),
                Display::Flex => compute_flexbox_layout(tree, node_id, inputs),
                Display::Grid => compute_grid_layout(tree, node_id, inputs),
            }
        })
    }
}

impl LayoutPartialTree for LayoutRun<'_> {
    type CustomIdent = String;
    type CoreContainerStyle<'a>
        = &'a taffy::style::Style
    where
        Self: 'a;

    fn get_core_container_style(&self, node_id: NodeId) -> Self::CoreContainerStyle<'_> {
        self.tree.style(node_id)
    }

    fn resolve_calc_value(&self, val: *const (), basis: f32) -> f32 {
        // The only pointers handed to Taffy come from CalcLength::as_ptr and
        // are retained by LayoutNode::calc_lengths for this layout run.
        self.tree
            .nodes
            .iter()
            .flat_map(|node| node.calc_lengths.iter())
            .find(|length| length.as_ptr() == val)
            // Every current caller builds expressions from pixels. Use the
            // default rem only for GPUI callers that opt into rem expressions
            // before the layout engine has a per-node rem metric.
            .map_or(0.0, |length| {
                length.resolve(basis, crate::px(16.0), self.scale_factor)
            })
    }

    fn set_unrounded_layout(&mut self, node_id: NodeId, layout: &Layout) {
        self.tree.node_mut(node_id).layout = *layout;
    }

    fn compute_child_layout(&mut self, node_id: NodeId, inputs: LayoutInput) -> LayoutOutput {
        self.compute_node_layout(node_id, inputs, None)
    }
}

fn calc_lengths(style: &Style) -> Vec<CalcLength> {
    fn length(value: &Length, values: &mut Vec<CalcLength>) {
        match value {
            Length::Definite(_) => {}
            Length::Calc(value) => values.push(value.clone()),
            Length::Auto => {}
        }
    }

    let mut values = Vec::new();
    for value in [
        &style.inset.top,
        &style.inset.right,
        &style.inset.bottom,
        &style.inset.left,
        &style.size.width,
        &style.size.height,
        &style.min_size.width,
        &style.min_size.height,
        &style.max_size.width,
        &style.max_size.height,
        &style.margin.top,
        &style.margin.right,
        &style.margin.bottom,
        &style.margin.left,
        &style.flex_basis,
    ] {
        length(value, &mut values);
    }
    values
}

impl LayoutFlexboxContainer for LayoutRun<'_> {
    type FlexboxContainerStyle<'a>
        = &'a taffy::style::Style
    where
        Self: 'a;
    type FlexboxItemStyle<'a>
        = &'a taffy::style::Style
    where
        Self: 'a;

    fn get_flexbox_container_style(&self, node_id: NodeId) -> Self::FlexboxContainerStyle<'_> {
        self.tree.style(node_id)
    }

    fn get_flexbox_child_style(&self, child_node_id: NodeId) -> Self::FlexboxItemStyle<'_> {
        self.tree.style(child_node_id)
    }
}

impl LayoutGridContainer for LayoutRun<'_> {
    type GridContainerStyle<'a>
        = &'a taffy::style::Style
    where
        Self: 'a;
    type GridItemStyle<'a>
        = &'a taffy::style::Style
    where
        Self: 'a;

    fn get_grid_container_style(&self, node_id: NodeId) -> Self::GridContainerStyle<'_> {
        self.tree.style(node_id)
    }

    fn get_grid_child_style(&self, child_node_id: NodeId) -> Self::GridItemStyle<'_> {
        self.tree.style(child_node_id)
    }
}

impl LayoutBlockContainer for LayoutRun<'_> {
    type BlockContainerStyle<'a>
        = &'a taffy::style::Style
    where
        Self: 'a;
    type BlockItemStyle<'a>
        = &'a taffy::style::Style
    where
        Self: 'a;

    fn get_block_container_style(&self, node_id: NodeId) -> Self::BlockContainerStyle<'_> {
        self.tree.style(node_id)
    }

    fn get_block_child_style(&self, child_node_id: NodeId) -> Self::BlockItemStyle<'_> {
        self.tree.style(child_node_id)
    }

    fn compute_block_child_layout(
        &mut self,
        node_id: NodeId,
        inputs: LayoutInput,
        context: Option<&mut taffy::BlockContext>,
    ) -> LayoutOutput {
        self.compute_node_layout(node_id, inputs, context)
    }
}

pub struct TaffyLayoutEngine {
    taffy: GpuiTaffyTree,
    absolute_layout_bounds: FxHashMap<LayoutId, Bounds<Pixels>>,
    /// Unrounded absolute border-box top-left per-node coordinate in device pixels.
    absolute_outer_origins: FxHashMap<LayoutId, Point<f32>>,
    computed_layouts: FxHashSet<LayoutId>,
    layout_bounds_scratch_space: Vec<LayoutId>,
}

impl TaffyLayoutEngine {
    pub fn new() -> Self {
        TaffyLayoutEngine {
            taffy: GpuiTaffyTree::default(),
            absolute_layout_bounds: FxHashMap::default(),
            absolute_outer_origins: FxHashMap::default(),
            computed_layouts: FxHashSet::default(),
            layout_bounds_scratch_space: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.taffy.clear();
        self.absolute_layout_bounds.clear();
        self.absolute_outer_origins.clear();
        self.computed_layouts.clear();
    }

    pub fn request_layout(
        &mut self,
        style: Style,
        rem_size: Pixels,
        scale_factor: f32,
        children: &[LayoutId],
    ) -> LayoutId {
        let calc_lengths = calc_lengths(&style);
        let taffy_style = style.to_taffy(rem_size, scale_factor);

        self.taffy.new_node(taffy_style, calc_lengths, children, None)
    }

    pub fn request_measured_layout(
        &mut self,
        style: Style,
        rem_size: Pixels,
        scale_factor: f32,
        mut measure: impl FnMut(
            Size<Option<Pixels>>,
            Size<AvailableSpace>,
            &mut Window,
            &mut App,
        ) -> Size<Pixels>
        + 'static,
    ) -> LayoutId {
        self.request_measured_layout_with_baseline(
            style,
            rem_size,
            scale_factor,
            move |known, available, window, cx| (measure(known, available, window, cx), None),
        )
    }

    pub fn request_measured_layout_with_baseline(
        &mut self,
        style: Style,
        rem_size: Pixels,
        scale_factor: f32,
        mut measure: impl FnMut(
            Size<Option<Pixels>>,
            Size<AvailableSpace>,
            &mut Window,
            &mut App,
        ) -> (Size<Pixels>, Option<Pixels>)
        + 'static,
    ) -> LayoutId {
        let calc_lengths = calc_lengths(&style);
        let taffy_style = style.to_taffy(rem_size, scale_factor);
        let measure = Box::new(move |known, available, window: &mut Window, cx: &mut App| {
            let (size, first_baseline) = measure(known, available, window, cx);
            MeasuredLayout {
                size,
                first_baseline,
            }
        })
            as Box<
                dyn FnMut(
                    Size<Option<Pixels>>,
                    Size<AvailableSpace>,
                    &mut Window,
                    &mut App,
                ) -> MeasuredLayout,
            >;
        #[cfg(feature = "stacker")]
        let measure = StackSafe::new(measure);

        self.taffy
            .new_node(taffy_style, calc_lengths, &[], Some(NodeContext::Dynamic(measure)))
    }

    /// Treats any `auto` dimension of the given node's style as filling `size`.
    ///
    /// This is applied to window roots before layout so they behave like the
    /// root element on the web, which stretches to fill the initial containing
    /// block (the viewport) unless given an explicit size. Explicitly styled
    /// dimensions are preserved.
    pub fn stretch_auto_size_to_fill(
        &mut self,
        id: LayoutId,
        size: Size<Pixels>,
        scale_factor: f32,
    ) {
        let style = self.taffy.style(id.0);
        let stretch_width = style.size.width.is_auto();
        let stretch_height = style.size.height.is_auto();
        if !stretch_width && !stretch_height {
            return;
        }
        let mut style = style.clone();
        if stretch_width {
            style.size.width =
                taffy::style::Dimension::length(round_to_device_pixel(size.width.0, scale_factor));
        }
        if stretch_height {
            style.size.height =
                taffy::style::Dimension::length(round_to_device_pixel(size.height.0, scale_factor));
        }
        self.taffy.set_style(id.0, style);
    }

    // Used to understand performance
    #[allow(dead_code)]
    fn count_all_children(&self, parent: LayoutId) -> anyhow::Result<u32> {
        let mut count = 0;

        for child in self.taffy.children(parent.0) {
            // Count this child.
            count += 1;

            // Count all of this child's children.
            count += self.count_all_children(LayoutId(child))?
        }

        Ok(count)
    }

    // Used to understand performance
    #[allow(dead_code)]
    fn max_depth(&self, depth: u32, parent: LayoutId) -> anyhow::Result<u32> {
        println!(
            "{parent:?} at depth {depth} has {} children",
            self.taffy.node(parent.0).children.len()
        );

        let mut max_child_depth = 0;

        for child in self.taffy.children(parent.0) {
            max_child_depth = std::cmp::max(max_child_depth, self.max_depth(0, LayoutId(child))?);
        }

        Ok(depth + 1 + max_child_depth)
    }

    // Used to understand performance
    #[allow(dead_code)]
    fn get_edges(&self, parent: LayoutId) -> anyhow::Result<Vec<(LayoutId, LayoutId)>> {
        let mut edges = Vec::new();

        for child in self.taffy.children(parent.0) {
            edges.push((parent, LayoutId(child)));

            edges.extend(self.get_edges(LayoutId(child))?);
        }

        Ok(edges)
    }

    #[cfg_attr(feature = "stacker", stacksafe::stacksafe)]
    pub fn compute_layout(
        &mut self,
        id: LayoutId,
        available_space: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) {
        // Leaving this here until we have a better instrumentation approach.
        // println!("Laying out {} children", self.count_all_children(id)?);
        // println!("Max layout depth: {}", self.max_depth(0, id)?);

        // Output the edges (branches) of the tree in Mermaid format for visualization.
        // println!("Edges:");
        // for (a, b) in self.get_edges(id)? {
        //     println!("N{} --> N{}", u64::from(a), u64::from(b));
        // }
        //

        if !self.computed_layouts.insert(id) {
            let stack = &mut self.layout_bounds_scratch_space;
            stack.push(id);
            while let Some(id) = stack.pop() {
                self.absolute_layout_bounds.remove(&id);
                self.absolute_outer_origins.remove(&id);
                stack.extend(
                    self.taffy
                        .children(id.into())
                        .into_iter()
                        .map(LayoutId::from),
                );
            }
        }

        let scale_factor = window.scale_factor();

        let transform = |v: AvailableSpace| match v {
            AvailableSpace::Definite(pixels) => {
                AvailableSpace::Definite(Pixels(pixels.0 * scale_factor))
            }
            AvailableSpace::MinContent => AvailableSpace::MinContent,
            AvailableSpace::MaxContent => AvailableSpace::MaxContent,
        };
        let available_space = size(
            transform(available_space.width),
            transform(available_space.height),
        );

        let mut tree = LayoutRun {
            tree: &mut self.taffy,
            window: Some(window),
            cx: Some(cx),
            scale_factor,
        };
        compute_root_layout(&mut tree, id.into(), available_space.into());
    }

    // Pixel snapping
    //
    // Painting primitives at non-integer pixel coordinates produces blurry
    // output. Pixel snapping converts layout coordinates into integer
    // device-pixel coordinates so painted edges land exactly on physical
    // pixel boundaries.
    //
    // Non-integer coordinates can arise for several reasons, including:
    //   - flex distribution, percentages, centering, and text measurement
    //     can produce fractional element sizes and positions;
    //   - at fractional scale factors (for example 125% or 150%), integer
    //     logical-pixel values can map to non-integer device-pixel values.
    //
    // We pixel-snap by rounding in device-pixel space, after multiplying
    // by `scale_factor`, so that snapping targets physical pixels. Bounds
    // are divided by `scale_factor` before being returned to GPUI.
    //
    // Midpoints are rounded toward zero. This is a stylistic choice: a
    // 1-logical-pixel line at 150% scale should render as 1 dp rather than
    // 2 dp.
    //
    // Pixel snapping is done in two phases:
    //
    //  1. Pre-layout metric snapping. Before Taffy computes layout, all
    //     authored absolute lengths are rounded in `to_taffy`. This
    //     includes borders, padding, gaps, and explicit sizes.
    //     Custom-measured leaf nodes have their measured sizes rounded up
    //     to integer device-pixel lengths.
    //
    //  2. Post-layout edge snapping. After Taffy resolves the tree, layout
    //     relationships such as flex shares, grid tracks, percentages, and
    //     centering can produce new fractional edge positions. Boxes now
    //     have edges in absolute coordinates, and snapping must decide
    //     where those edges land on the device-pixel grid.
    //
    // Ideally, post-layout snapping would satisfy:
    //
    //  - Edge closure. Two raw layout edges at the same absolute position
    //    should snap to the same pixel column.
    //  - Translation stability. A component's internal geometry should not
    //    change when it moves to a new absolute position.
    //
    // These goals are in tension because rounding is not associative.
    // The simple local schemes make different tradeoffs:
    //
    //  - Absolute edge rounding gives each window coordinate one answer,
    //    so coincident edges always close globally. But a span's snapped
    //    length is `round(far) - round(near)`, which may change by 1 dp
    //    as its absolute origin moves.
    //
    //  - Parent-relative edge rounding rounds each child inside its
    //    parent's coordinate space. This guarantees translation stability,
    //    but a shared edge reached through different parents can
    //    accumulate different rounding, causing non-closure between
    //    cousins.
    //
    //  - Length rounding rounds each width, height, and thickness
    //    independently and then places boxes from those rounded lengths.
    //    Sizes stay stable under translation, but neighboring boxes derive
    //    their shared boundary from different sources, so closure is not
    //    guaranteed.
    //
    // We apply absolute edge rounding for each element's outer box in
    // post-layout rounding to preserve closure. Border and padding widths
    // are not touched by post-layout rounding; they keep their pre-layout
    // rounded value so that they remain stable under translation.
    //
    // This gives both closure and translation stability in the case that
    // all local metrics are integer device-pixel lengths. Pre-layout
    // rounding covers that in most cases. The exception is metrics
    // resolved by layout relationships, such as percentages. Outer box
    // edges will still close globally, and painted border widths are still
    // snapped independently, but the raw content-box origin can carry a
    // 1dp residual into descendants.

    pub fn layout_bounds(&mut self, id: LayoutId, scale_factor: f32) -> Bounds<Pixels> {
        if let Some(layout) = self.absolute_layout_bounds.get(&id).cloned() {
            return layout;
        }

        let layout = self.taffy.layout(id.into());
        let layout_location = layout.location;
        let layout_size = layout.size;
        let parent = self.taffy.parent(id.0);

        let absolute_outer_origin = match parent {
            Some(parent_id) => {
                let parent_id = LayoutId::from(parent_id);
                self.layout_bounds(parent_id, scale_factor);
                let parent_origin = *self
                    .absolute_outer_origins
                    .get(&parent_id)
                    .expect("parent absolute outer origin should be cached");
                parent_origin + Point::from(layout_location)
            }
            None => Point::from(layout_location),
        };
        self.absolute_outer_origins
            .insert(id, absolute_outer_origin);

        let absolute_far = absolute_outer_origin + Point::from(Size::from(layout_size));
        let snapped_bounds = Bounds::from_corners(
            absolute_outer_origin.map(round_half_toward_zero),
            absolute_far.map(round_half_toward_zero),
        );

        let bounds = (snapped_bounds / scale_factor).map(Pixels);
        self.absolute_layout_bounds.insert(id, bounds);
        bounds
    }
}

/// A unique identifier for a layout node, generated when requesting a layout from Taffy
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(transparent)]
pub struct LayoutId(NodeId);

impl std::hash::Hash for LayoutId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        u64::from(self.0).hash(state);
    }
}

impl From<NodeId> for LayoutId {
    fn from(node_id: NodeId) -> Self {
        Self(node_id)
    }
}

impl From<LayoutId> for NodeId {
    fn from(layout_id: LayoutId) -> NodeId {
        layout_id.0
    }
}

fn snap_measured_size_to_device_pixels(size: Size<Pixels>, scale_factor: f32) -> Size<f32> {
    size.map(|d| ceil_to_device_pixel(d.0.max(0.0), scale_factor))
}

fn border_widths_to_taffy(
    widths: &Edges<AbsoluteLength>,
    rem_size: Pixels,
    scale_factor: f32,
) -> TaffyRect<taffy::style::LengthPercentage> {
    let snap = |w: &AbsoluteLength| {
        taffy::style::LengthPercentage::length(round_stroke_to_device_pixel(
            w.to_pixels(rem_size).0,
            scale_factor,
        ))
    };
    TaffyRect {
        top: snap(&widths.top),
        right: snap(&widths.right),
        bottom: snap(&widths.bottom),
        left: snap(&widths.left),
    }
}

trait ToTaffy<Output> {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> Output;
}

impl ToTaffy<taffy::style::Style> for Style {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::Style {
        use taffy::style_helpers::{auto, fr, length, max_content, min_content, minmax, repeat};

        fn to_grid_line(
            placement: &Range<crate::GridPlacement>,
        ) -> taffy::Line<taffy::GridPlacement> {
            taffy::Line {
                start: placement.start.into(),
                end: placement.end.into(),
            }
        }

        fn to_min_track(track: &GridTrackMin) -> taffy::style::MinTrackSizingFunction {
            match track {
                GridTrackMin::Px(value) => length(value.0),
                GridTrackMin::Auto => auto(),
                GridTrackMin::MinContent => min_content(),
                GridTrackMin::MaxContent => max_content(),
            }
        }

        fn to_max_track(track: &GridTrackMax) -> taffy::style::MaxTrackSizingFunction {
            match track {
                GridTrackMax::Px(value) => length(value.0),
                GridTrackMax::Fr(value) => fr(*value),
                GridTrackMax::Auto => auto(),
                GridTrackMax::MinContent => min_content(),
                GridTrackMax::MaxContent => max_content(),
            }
        }

        fn to_track(track: &GridTrack) -> taffy::style::TrackSizingFunction {
            match track {
                GridTrack::Px(value) => length(value.0),
                GridTrack::Fr(value) => fr(*value),
                GridTrack::Auto => auto(),
                GridTrack::MinContent => min_content(),
                GridTrack::MaxContent => max_content(),
                GridTrack::MinMax { min, max } => minmax(to_min_track(min), to_max_track(max)),
            }
        }

        fn to_grid_template<T: taffy::style::CheapCloneStr>(
            template: &Option<GridTemplate>,
        ) -> Vec<taffy::GridTemplateComponent<T>> {
            template
                .iter()
                .flat_map(|template| template.tracks.iter())
                .map(|component| match component {
                    GridTemplateComponent::Track(track) => {
                        taffy::GridTemplateComponent::Single(to_track(track))
                    }
                    GridTemplateComponent::Repeat { count, tracks } => {
                        repeat(*count, tracks.iter().map(to_track).collect())
                    }
                })
                .collect()
        }

        taffy::style::Style {
            display: self.display.into(),
            overflow: self.overflow.into(),
            scrollbar_width: self.scrollbar_width.to_taffy(rem_size, scale_factor),
            position: self.position.into(),
            inset: self.inset.to_taffy(rem_size, scale_factor),
            size: self.size.to_taffy(rem_size, scale_factor),
            min_size: self.min_size.to_taffy(rem_size, scale_factor),
            max_size: self.max_size.to_taffy(rem_size, scale_factor),
            aspect_ratio: self.aspect_ratio,
            margin: self.margin.to_taffy(rem_size, scale_factor),
            padding: self.padding.to_taffy(rem_size, scale_factor),
            border: border_widths_to_taffy(&self.border_widths, rem_size, scale_factor),
            align_items: self.align_items.map(|x| x.into()),
            align_self: self.align_self.map(|x| x.into()),
            align_content: self.align_content.map(|x| x.into()),
            justify_content: self.justify_content.map(|x| x.into()),
            gap: self.gap.to_taffy(rem_size, scale_factor),
            flex_direction: self.flex_direction.into(),
            flex_wrap: self.flex_wrap.into(),
            flex_basis: self.flex_basis.to_taffy(rem_size, scale_factor),
            flex_grow: self.flex_grow,
            flex_shrink: self.flex_shrink,
            grid_template_rows: to_grid_template(&self.grid_rows),
            grid_template_columns: to_grid_template(&self.grid_cols),
            grid_row: self
                .grid_location
                .as_ref()
                .map(|location| to_grid_line(&location.row))
                .unwrap_or_default(),
            grid_column: self
                .grid_location
                .as_ref()
                .map(|location| to_grid_line(&location.column))
                .unwrap_or_default(),
            ..Default::default()
        }
    }
}

impl ToTaffy<f32> for AbsoluteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> f32 {
        round_to_device_pixel(self.to_pixels(rem_size).0, scale_factor)
    }
}

impl ToTaffy<taffy::style::LengthPercentageAuto> for Length {
    fn to_taffy(
        &self,
        rem_size: Pixels,
        scale_factor: f32,
    ) -> taffy::prelude::LengthPercentageAuto {
        match self {
            Length::Definite(length) => length.to_taffy(rem_size, scale_factor),
            Length::Calc(length) => taffy::prelude::LengthPercentageAuto::calc(length.as_ptr()),
            Length::Auto => taffy::prelude::LengthPercentageAuto::auto(),
        }
    }
}

impl ToTaffy<taffy::style::Dimension> for Length {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::prelude::Dimension {
        match self {
            Length::Definite(length) => length.to_taffy(rem_size, scale_factor),
            Length::Calc(length) => taffy::prelude::Dimension::calc(length.as_ptr()),
            Length::Auto => taffy::prelude::Dimension::auto(),
        }
    }
}

impl ToTaffy<taffy::style::LengthPercentage> for DefiniteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::LengthPercentage {
        match self {
            DefiniteLength::Absolute(length) => length.to_taffy(rem_size, scale_factor),
            DefiniteLength::Fraction(fraction) => {
                taffy::style::LengthPercentage::percent(*fraction)
            }
        }
    }
}

impl ToTaffy<taffy::style::LengthPercentageAuto> for DefiniteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::LengthPercentageAuto {
        match self {
            DefiniteLength::Absolute(length) => length.to_taffy(rem_size, scale_factor),
            DefiniteLength::Fraction(fraction) => {
                taffy::style::LengthPercentageAuto::percent(*fraction)
            }
        }
    }
}

impl ToTaffy<taffy::style::Dimension> for DefiniteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::Dimension {
        match self {
            DefiniteLength::Absolute(length) => length.to_taffy(rem_size, scale_factor),
            DefiniteLength::Fraction(fraction) => taffy::style::Dimension::percent(*fraction),
        }
    }
}

impl ToTaffy<taffy::style::LengthPercentage> for AbsoluteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::LengthPercentage {
        taffy::style::LengthPercentage::length(self.to_taffy(rem_size, scale_factor))
    }
}

impl ToTaffy<taffy::style::LengthPercentageAuto> for AbsoluteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::LengthPercentageAuto {
        taffy::style::LengthPercentageAuto::length(self.to_taffy(rem_size, scale_factor))
    }
}

impl ToTaffy<taffy::style::Dimension> for AbsoluteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::Dimension {
        taffy::style::Dimension::length(self.to_taffy(rem_size, scale_factor))
    }
}

impl<T, T2> From<TaffyPoint<T>> for Point<T2>
where
    T: Into<T2>,
    T2: Clone + Debug + Default + PartialEq,
{
    fn from(point: TaffyPoint<T>) -> Point<T2> {
        Point {
            x: point.x.into(),
            y: point.y.into(),
        }
    }
}

impl<T, T2> From<Point<T>> for TaffyPoint<T2>
where
    T: Into<T2> + Clone + Debug + Default + PartialEq,
{
    fn from(val: Point<T>) -> Self {
        TaffyPoint {
            x: val.x.into(),
            y: val.y.into(),
        }
    }
}

impl<T, U> ToTaffy<TaffySize<U>> for Size<T>
where
    T: ToTaffy<U> + Clone + Debug + Default + PartialEq,
{
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> TaffySize<U> {
        TaffySize {
            width: self.width.to_taffy(rem_size, scale_factor),
            height: self.height.to_taffy(rem_size, scale_factor),
        }
    }
}

impl<T, U> ToTaffy<TaffyRect<U>> for Edges<T>
where
    T: ToTaffy<U> + Clone + Debug + Default + PartialEq,
{
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> TaffyRect<U> {
        TaffyRect {
            top: self.top.to_taffy(rem_size, scale_factor),
            right: self.right.to_taffy(rem_size, scale_factor),
            bottom: self.bottom.to_taffy(rem_size, scale_factor),
            left: self.left.to_taffy(rem_size, scale_factor),
        }
    }
}

impl<T, U> From<TaffySize<T>> for Size<U>
where
    T: Into<U>,
    U: Clone + Debug + Default + PartialEq,
{
    fn from(taffy_size: TaffySize<T>) -> Self {
        Size {
            width: taffy_size.width.into(),
            height: taffy_size.height.into(),
        }
    }
}

impl<T, U> From<Size<T>> for TaffySize<U>
where
    T: Into<U> + Clone + Debug + Default + PartialEq,
{
    fn from(size: Size<T>) -> Self {
        TaffySize {
            width: size.width.into(),
            height: size.height.into(),
        }
    }
}

/// The space available for an element to be laid out in
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub enum AvailableSpace {
    /// The amount of space available is the specified number of pixels
    Definite(Pixels),
    /// The amount of space available is indefinite and the node should be laid out under a min-content constraint
    #[default]
    MinContent,
    /// The amount of space available is indefinite and the node should be laid out under a max-content constraint
    MaxContent,
}

impl AvailableSpace {
    /// Returns a `Size` with both width and height set to `AvailableSpace::MinContent`.
    ///
    /// This function is useful when you want to create a `Size` with the minimum content constraints
    /// for both dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use gpui::AvailableSpace;
    /// let min_content_size = AvailableSpace::min_size();
    /// assert_eq!(min_content_size.width, AvailableSpace::MinContent);
    /// assert_eq!(min_content_size.height, AvailableSpace::MinContent);
    /// ```
    pub const fn min_size() -> Size<Self> {
        Size {
            width: Self::MinContent,
            height: Self::MinContent,
        }
    }
}

impl From<AvailableSpace> for TaffyAvailableSpace {
    fn from(space: AvailableSpace) -> TaffyAvailableSpace {
        match space {
            AvailableSpace::Definite(Pixels(value)) => TaffyAvailableSpace::Definite(value),
            AvailableSpace::MinContent => TaffyAvailableSpace::MinContent,
            AvailableSpace::MaxContent => TaffyAvailableSpace::MaxContent,
        }
    }
}

impl From<TaffyAvailableSpace> for AvailableSpace {
    fn from(space: TaffyAvailableSpace) -> AvailableSpace {
        match space {
            TaffyAvailableSpace::Definite(value) => AvailableSpace::Definite(Pixels(value)),
            TaffyAvailableSpace::MinContent => AvailableSpace::MinContent,
            TaffyAvailableSpace::MaxContent => AvailableSpace::MaxContent,
        }
    }
}

impl From<Pixels> for AvailableSpace {
    fn from(pixels: Pixels) -> Self {
        AvailableSpace::Definite(pixels)
    }
}

impl From<Size<Pixels>> for Size<AvailableSpace> {
    fn from(size: Size<Pixels>) -> Self {
        Size {
            width: AvailableSpace::Definite(size.width),
            height: AvailableSpace::Definite(size.height),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::px;
    use taffy::{AlignContent, AlignItems, FlexDirection, FlexWrap, style_helpers::length};

    fn fixed_leaf(
        tree: &mut GpuiTaffyTree,
        width: f32,
        height: f32,
        first_baseline: Option<f32>,
    ) -> LayoutId {
        let mut style = taffy::style::Style::default();
        style.size.width = length(width);
        tree.new_node(
            style,
            Vec::new(),
            &[],
            Some(NodeContext::Fixed(MeasuredLayout {
                size: size(Pixels(width), Pixels(height)),
                first_baseline: first_baseline.map(Pixels),
            })),
        )
    }

    fn fixed_sized_leaf(
        tree: &mut GpuiTaffyTree,
        width: f32,
        height: f32,
        first_baseline: Option<f32>,
    ) -> LayoutId {
        let mut style = taffy::style::Style::default();
        style.size = TaffySize {
            width: length(width),
            height: length(height),
        };
        tree.new_node(
            style,
            Vec::new(),
            &[],
            Some(NodeContext::Fixed(MeasuredLayout {
                size: size(Pixels(width), Pixels(height)),
                first_baseline: first_baseline.map(Pixels),
            })),
        )
    }

    fn flex_container(
        tree: &mut GpuiTaffyTree,
        children: &[LayoutId],
        configure: impl FnOnce(&mut taffy::style::Style),
    ) -> LayoutId {
        let mut style = taffy::style::Style {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: Some(AlignItems::BASELINE),
            ..Default::default()
        };
        configure(&mut style);
        tree.new_node(style, Vec::new(), children, None)
    }

    fn compute_test_layout_at_scale(tree: &mut GpuiTaffyTree, root: LayoutId, scale_factor: f32) {
        let mut run = LayoutRun {
            tree,
            window: None,
            cx: None,
            scale_factor,
        };
        compute_root_layout(
            &mut run,
            root.into(),
            TaffySize {
                width: TaffyAvailableSpace::MaxContent,
                height: TaffyAvailableSpace::MaxContent,
            },
        );
    }

    fn compute_test_layout(tree: &mut GpuiTaffyTree, root: LayoutId) {
        compute_test_layout_at_scale(tree, root, 1.);
    }

    fn bottom(layout: &Layout) -> f32 {
        layout.location.y + layout.size.height
    }

    #[test]
    fn measured_baselines_are_scoped_to_each_compute_root() {
        let mut tree = GpuiTaffyTree::default();
        let first_large = fixed_leaf(&mut tree, 20., 32., Some(24.));
        let first_small = fixed_leaf(&mut tree, 20., 12., Some(9.));
        let first_root = flex_container(&mut tree, &[first_large, first_small], |_| {});
        let second_large = fixed_leaf(&mut tree, 20., 32., Some(24.));
        let second_small = fixed_leaf(&mut tree, 20., 12., Some(9.));
        let second_root = flex_container(&mut tree, &[second_large, second_small], |_| {});

        compute_test_layout(&mut tree, first_root);
        compute_test_layout(&mut tree, second_root);

        assert_eq!(tree.layout(first_large.0).location.y, 0.);
        assert_eq!(tree.layout(second_large.0).location.y, 0.);
        assert_eq!(tree.layout(first_small.0).location.y, 15.);
        assert_eq!(tree.layout(second_small.0).location.y, 15.);
    }

    #[test]
    fn wrapped_flex_lines_align_their_own_baseline_groups() {
        let mut tree = GpuiTaffyTree::default();
        let leaves = [
            fixed_leaf(&mut tree, 30., 32., Some(24.)),
            fixed_leaf(&mut tree, 30., 12., Some(9.)),
            fixed_leaf(&mut tree, 30., 32., Some(24.)),
            fixed_leaf(&mut tree, 30., 12., Some(9.)),
        ];
        let root = flex_container(&mut tree, &leaves, |style| {
            style.flex_wrap = FlexWrap::Wrap;
            style.align_content = Some(AlignContent::STRETCH);
            style.size = TaffySize {
                width: length(60.),
                height: length(104.),
            };
        });

        compute_test_layout(&mut tree, root);

        let layouts = leaves.map(|leaf| *tree.layout(leaf.0));
        assert!(layouts[0].location.y < layouts[2].location.y);
        assert!(bottom(&layouts[0]) <= layouts[2].location.y);
        assert!(bottom(&layouts[1]) < bottom(&layouts[0]));
        assert!(bottom(&layouts[3]) < bottom(&layouts[2]));
    }

    #[test]
    fn baseline_less_flex_items_synthesize_their_bottom_edge() {
        let mut tree = GpuiTaffyTree::default();
        let box_node = fixed_leaf(&mut tree, 32., 32., None);
        let text_node = fixed_leaf(&mut tree, 20., 16., Some(12.));
        let root = flex_container(&mut tree, &[box_node, text_node], |_| {});

        compute_test_layout(&mut tree, root);

        assert!(bottom(tree.layout(text_node.0)) > bottom(tree.layout(box_node.0)));
        assert!(tree.layout(root.0).size.height > tree.layout(box_node.0).size.height);
    }

    #[test]
    fn explicitly_sized_measured_leaves_preserve_their_baselines() {
        let mut tree = GpuiTaffyTree::default();
        let large = fixed_sized_leaf(&mut tree, 20., 32., Some(24.));
        let small = fixed_sized_leaf(&mut tree, 20., 12., Some(9.));
        let root = flex_container(&mut tree, &[large, small], |_| {});

        compute_test_layout(&mut tree, root);

        assert_eq!(tree.layout(large.0).location.y, 0.);
        assert_eq!(tree.layout(small.0).location.y, 15.);
    }

    #[test]
    fn nested_flex_containers_export_their_corrected_first_baseline() {
        let mut tree = GpuiTaffyTree::default();
        let nested_small = fixed_leaf(&mut tree, 20., 12., Some(9.));
        let nested_large = fixed_leaf(&mut tree, 20., 32., Some(24.));
        let nested = flex_container(&mut tree, &[nested_small, nested_large], |_| {});
        let outer_small = fixed_leaf(&mut tree, 20., 12., Some(9.));
        let root = flex_container(&mut tree, &[nested, outer_small], |_| {});

        compute_test_layout(&mut tree, root);

        let nested_origin = tree.layout(nested.0).location.y;
        let nested_small_y = nested_origin + tree.layout(nested_small.0).location.y;
        let nested_large_y = nested_origin + tree.layout(nested_large.0).location.y;
        assert_eq!(nested_small_y, tree.layout(outer_small.0).location.y);
        assert!(nested_large_y < nested_small_y);
    }

    #[test]
    fn fractional_baselines_remain_aligned_after_layout_and_paint_snapping() {
        let scale_factor = 1.25;
        let large_baseline = 24.3 / scale_factor;
        let small_baseline = 9.7 / scale_factor;
        let mut tree = GpuiTaffyTree::default();
        let large = fixed_leaf(&mut tree, 20., 32. / scale_factor, Some(large_baseline));
        let small = fixed_leaf(&mut tree, 20., 12. / scale_factor, Some(small_baseline));
        let root = flex_container(&mut tree, &[large, small], |_| {});

        compute_test_layout_at_scale(&mut tree, root, scale_factor);

        let painted_baseline = |node: LayoutId, baseline: f32| {
            let snapped_origin = round_half_toward_zero(tree.layout(node.0).location.y);
            round_half_toward_zero(snapped_origin + baseline * scale_factor)
        };
        assert_eq!(
            painted_baseline(large, large_baseline),
            painted_baseline(small, small_baseline)
        );
    }

    #[test]
    fn custom_tree_dispatches_block_grid_and_hidden_layouts() {
        let mut tree = GpuiTaffyTree::default();

        let first_block_child = fixed_leaf(&mut tree, 10., 10., None);
        let second_block_child = fixed_leaf(&mut tree, 10., 20., None);
        let block = tree.new_node(
            taffy::style::Style {
                display: Display::Block,
                ..Default::default()
            },
            Vec::new(),
            &[first_block_child, second_block_child],
            None,
        );
        compute_test_layout(&mut tree, block);
        assert_eq!(tree.layout(second_block_child.0).location.y, 10.);
        assert_eq!(tree.layout(block.0).size.height, 30.);

        let first_grid_child = fixed_leaf(&mut tree, 10., 10., None);
        let second_grid_child = fixed_leaf(&mut tree, 10., 10., None);
        let grid = tree.new_node(
            taffy::style::Style {
                display: Display::Grid,
                grid_template_columns: vec![length(20.), length(20.)],
                ..Default::default()
            },
            Vec::new(),
            &[first_grid_child, second_grid_child],
            None,
        );
        compute_test_layout(&mut tree, grid);
        assert_eq!(tree.layout(first_grid_child.0).location.x, 0.);
        assert_eq!(tree.layout(second_grid_child.0).location.x, 20.);

        let hidden = fixed_leaf(&mut tree, 40., 40., Some(30.));
        tree.node_mut(hidden.0).style.display = Display::None;
        compute_test_layout(&mut tree, hidden);
        assert_eq!(tree.layout(hidden.0).size, TaffySize::ZERO);
    }

    #[test]
    fn border_widths_to_taffy_use_stroke_snapping() {
        let border_widths = Edges {
            top: Pixels(0.0).into(),
            right: Pixels(0.4).into(),
            bottom: Pixels(0.5).into(),
            left: Pixels(1.6).into(),
        };
        let taffy_border = border_widths_to_taffy(&border_widths, Pixels(16.0), 1.0);

        assert_eq!(
            taffy_border.top,
            taffy::style::LengthPercentage::length(0.0)
        );
        assert_eq!(
            taffy_border.right,
            taffy::style::LengthPercentage::length(1.0)
        );
        assert_eq!(
            taffy_border.bottom,
            taffy::style::LengthPercentage::length(1.0)
        );
        assert_eq!(
            taffy_border.left,
            taffy::style::LengthPercentage::length(2.0)
        );
    }

    #[test]
    fn grid_templates_preserve_mixed_tracks_and_repeat() {
        use taffy::style_helpers::{fr, length, max_content, minmax, repeat};

        let style = Style {
            grid_cols: Some(GridTemplate {
                tracks: vec![
                    GridTemplateComponent::Track(GridTrack::MaxContent),
                    GridTemplateComponent::Track(GridTrack::MinMax {
                        min: GridTrackMin::Px(px(0.)),
                        max: GridTrackMax::Fr(1.),
                    }),
                    GridTemplateComponent::Track(GridTrack::Auto),
                    GridTemplateComponent::Repeat {
                        count: 2,
                        tracks: vec![GridTrack::Px(px(48.)), GridTrack::Fr(2.)],
                    },
                ],
            }),
            grid_rows: Some(GridTemplate {
                tracks: vec![GridTemplateComponent::Track(GridTrack::MinContent)],
            }),
            ..Default::default()
        };

        let taffy_style: taffy::style::Style = style.to_taffy(px(16.), 1.);

        assert_eq!(
            taffy_style.grid_template_columns,
            vec![
                taffy::GridTemplateComponent::Single(max_content()),
                taffy::GridTemplateComponent::Single(minmax(length(0.), fr(1.))),
                taffy::GridTemplateComponent::Single(taffy::style_helpers::auto()),
                repeat(2, vec![length(48.), fr(2.)]),
            ]
        );
        assert_eq!(
            taffy_style.grid_template_rows,
            vec![taffy::GridTemplateComponent::Single(
                taffy::style_helpers::min_content(),
            )]
        );
    }
}
