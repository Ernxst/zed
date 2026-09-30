use crate::{
    AnyElement, App, Bounds, Element, GlobalElementId, InspectorElementId, IntoElement, LayoutId,
    Pixels, Window,
};
use std::sync::Arc;

/// Paints an element inside a CSS stacking boundary while keeping element traversal unchanged.
pub struct Stacking {
    child: AnyElement,
    source_order: Arc<[u32]>,
    stacking_phase: u8,
    z_index: i32,
    context: bool,
}

/// Wrap an element with its retained source-order token and stacking behaviour.
pub fn stacking(
    child: impl IntoElement,
    source_order: impl Into<Arc<[u32]>>,
    stacking_phase: u8,
    z_index: i32,
    context: bool,
) -> Stacking {
    Stacking {
        child: child.into_any_element(),
        source_order: source_order.into(),
        stacking_phase,
        z_index,
        context,
    }
}

impl Element for Stacking {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<crate::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let layout = self.child.request_layout(window, cx);
        (layout, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        window.push_stacking_element(
            self.source_order.clone(),
            self.stacking_phase,
            self.z_index,
            self.context,
        );
        self.child.prepaint(window, cx);
        window.pop_stacking_order();
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        window.push_stacking_element(
            self.source_order.clone(),
            self.stacking_phase,
            self.z_index,
            self.context,
        );
        self.child.paint(window, cx);
        window.pop_stacking_order();
    }
}

impl IntoElement for Stacking {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}
