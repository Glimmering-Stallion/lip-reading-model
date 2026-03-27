//! Temporal hysteresis for speech gating and articulation detection.
//!
//! This module provides the [`SpeechGate`], which is a simple state machine designed to stabilize
//! the "active" speech state during live inference. It prevents prediction flicker
//! by requiring a configurable streak of **"good" frames** (tracker lock + lip motion)
//! to trigger an utterance, and a streak of **"bad" frames** to end one.
//!
//! It acts as the primary filter for the inference sliding window, so that the VSRM
//! only consumes high-confidence, articulating mouth crops while signaling the
//! predictor to clear buffers and stale results on transition to idle.



/// Hysteresis for frame-to-frame speech gating to avoid flickering when tracker confidence wobbles
/// and makes sure model only runs when physical mouth articulation is detected.
pub struct SpeechGate {
    on_threshold: usize,
    off_threshold: usize,
    good_streak: usize,
    bad_streak: usize,
    active: bool,
}



impl SpeechGate {
    /// Builds a gate with consecutive-frame thresholds for opening (`on_frames`) and closing (`off_frames`).
    pub fn new(on_frames: usize, off_frames: usize) -> Self {
        Self {
            on_threshold: on_frames.max(1),
            off_threshold: off_frames.max(1),
            good_streak: 0,
            bad_streak: 0,
            active: false,
        }
    }

    /// Updates the `SpeechGate` activity state.
    /// 
    /// ### Params:
    /// - `has_lock`: Per-frame output of `LipTrackerBackend::has_lock`,
    /// - `has_lip_motion`: Per-frame output of `LipTrackerBackend::has_lip_motion` (visual proxy; both must be true for a “good” frame).
    /// 
    /// ### Returns:
    /// A `(speech_active, just_became_idle)` tuple where `just_became_idle` means a transition of
    /// active --> inactive occurred on this frame (caller clears window / prediction / stale channel).
    pub fn update(&mut self, has_lock: bool, has_lip_motion: bool) -> (bool, bool) {
        let was_active = self.active;

        // both validity conditions must be met to consider frame valid for speech
        let frame_ok = has_lock && has_lip_motion;

        if frame_ok {
            self.good_streak += 1;
            self.bad_streak = 0;
            if !self.active && self.good_streak >= self.on_threshold { self.active = true; }
        } else {
            self.bad_streak += 1;
            self.good_streak = 0;
            if self.active && self.bad_streak >= self.off_threshold { self.active = false; }
        }

        let just_became_idle = was_active && !self.active;
        (self.active, just_became_idle)
    }
}
