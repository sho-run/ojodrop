//! Capture recovery with bounded backoff and sustained-liveness detection.
//!
//! The state machine returns decisions; callers own capture I/O and supply a
//! monotonic clock. This keeps recovery testable without an audio device.
use std::time::Duration;

use crate::capture::CaptureConfig;

/// Initial delay after an unsuccessful reconnect attempt.
pub const AUDIO_RECONNECT_BASE_DELAY: Duration = Duration::from_secs(2);

/// Retry ceiling while capture remains unavailable.
pub const AUDIO_RECONNECT_MAX_DELAY: Duration = Duration::from_secs(30);

/// Continuous live time required before resetting the retry ladder.
pub const AUDIO_RECONNECT_SETTLE: Duration = AUDIO_RECONNECT_BASE_DELAY;

/// Failed attempts before alternating a loopback preference with plain input.
pub const AUDIO_FALLBACK_AFTER_FAILURES: u32 = 2;

/// How long to wait before the next reconnect attempt, given how many consecutive
/// attempts have already failed to produce *sustained* capture.
///
/// Zero for the first attempt, then `2 s`, `4 s`, `8 s`, `16 s`, capped at
/// [`AUDIO_RECONNECT_MAX_DELAY`]. Saturating throughout: a set that runs for hours
/// with no device must not overflow the exponent or the `Duration`.
pub fn audio_reconnect_delay(failed_attempts: u32) -> Duration {
    let Some(doublings) = failed_attempts.checked_sub(1) else {
        // Nothing has failed yet — this is the first attempt after the loss.
        return Duration::ZERO;
    };
    // `checked_pow` rather than a shift: `1u32 << 32` panics in debug and wraps in
    // release, and a session retrying for hours can push the count past 31.
    let delay = 2u32
        .checked_pow(doublings)
        .and_then(|factor| AUDIO_RECONNECT_BASE_DELAY.checked_mul(factor))
        .unwrap_or(AUDIO_RECONNECT_MAX_DELAY);
    delay.min(AUDIO_RECONNECT_MAX_DELAY)
}

/// Pure retry gate: has the current rung's wait elapsed?
pub fn should_try_audio_reconnect(failed_attempts: u32, since_last_attempt: Duration) -> bool {
    // `>=`, not `>`: a frame landing exactly on the boundary must fire, or the
    // wait silently becomes a whole extra period on a fixed-cadence loop.
    since_last_attempt >= audio_reconnect_delay(failed_attempts)
}

/// Alternate failed loopback requests with default-input attempts. A plain
/// default-input request is preserved at every retry.
pub fn next_capture_config(requested: CaptureConfig, failed_attempts: u32) -> CaptureConfig {
    if !requested.prefer_loopback || failed_attempts < AUDIO_FALLBACK_AFTER_FAILURES {
        return requested;
    }
    if (failed_attempts - AUDIO_FALLBACK_AFTER_FAILURES).is_multiple_of(2) {
        CaptureConfig {
            prefer_loopback: false,
        }
    } else {
        requested
    }
}

/// What the frame loop should do about capture this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureDecision {
    /// Capture is live — nothing to do.
    Idle,
    /// Capture is down but the backoff has not elapsed. Keep waiting.
    Wait,
    /// Backoff elapsed — drop any dead engine and open a new one with this config.
    Reconnect(CaptureConfig),
}

/// Capture status suitable for a host's diagnostics or user interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureHealth {
    /// A stream is open and has not reported an error.
    Live,
    /// Down, with a reconnect scheduled.
    Reconnecting {
        /// Consecutive attempts that have not yet produced *sustained* capture.
        failed_attempts: u32,
        /// Wait that applies before the next attempt.
        next_attempt_in: Duration,
        /// Why the last attempt failed, when there has been one.
        last_error: Option<String>,
    },
    /// Deliberately stopped by the host (muted, synthetic audio, headless).
    /// Never reached by the policy itself — a host sets it when it takes capture
    /// down on purpose, so "off" is distinguishable from "broken".
    Suspended,
}

/// The reconnect bookkeeping, kept as a tiny pure state machine so a whole
/// open/die/open/die sequence can be replayed in a unit test.
///
/// Times are monotonic durations since an arbitrary caller-owned epoch rather than
/// `Instant`s, so a test can replay an entire set's worth of frames without a
/// clock.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReconnectState {
    /// Attempts that have not yet produced *sustained* live capture.
    failures: u32,
    /// When the last authorised attempt was made.
    last_attempt: Duration,
    /// When the current unbroken run of observed-live frames began. `None`
    /// whenever capture is down, so any single dead frame restarts the run.
    live_since: Option<Duration>,
    /// Why the last attempt failed. Cleared once capture settles.
    last_error: Option<String>,
}

impl ReconnectState {
    /// Start with a given failure count. `ReconnectState::default()` is the usual
    /// entry point; this exists for tests and for a host resuming a known state.
    pub fn with_failures(failures: u32) -> Self {
        Self {
            failures,
            ..Self::default()
        }
    }

    /// Decide what to do about capture this frame, folding the decision into the
    /// counter.
    ///
    /// Two things deliberately do **not** reset the backoff ladder:
    ///
    /// 1. **A successful open.** A flapping device — opens cleanly, then its
    ///    worker dies before the next frame — constructs successfully every single
    ///    time. Counting construction as recovery pins the delay at zero and
    ///    reopens the stream at frame rate forever.
    /// 2. **A single live frame.** [`crate::AudioEngine`]'s liveness flag is
    ///    initialised `true` at construction, *before* the stream is confirmed, and
    ///    is only cleared later from the cpal error callback. A device that errors
    ///    within ~5–20 ms therefore reports live for the one frame between the two
    ///    — shorter than a 16.7 ms frame period — which is enough to zero a
    ///    one-poll reset and put the ladder straight back at the bottom.
    ///
    /// Only liveness **sustained** for [`AUDIO_RECONNECT_SETTLE`] counts as
    /// recovery.
    pub fn poll(&mut self, live: bool, now: Duration, requested: CaptureConfig) -> CaptureDecision {
        if live {
            let run_began = *self.live_since.get_or_insert(now);
            if now.saturating_sub(run_began) >= AUDIO_RECONNECT_SETTLE {
                self.failures = 0;
                self.last_error = None;
            }
            return CaptureDecision::Idle;
        }
        // Any dead frame breaks the run, so a blip can never accumulate.
        self.live_since = None;
        if !should_try_audio_reconnect(self.failures, now.saturating_sub(self.last_attempt)) {
            return CaptureDecision::Wait;
        }
        let cfg = next_capture_config(requested, self.failures);
        // Count the ATTEMPT, not its construction result.
        self.last_attempt = now;
        self.failures = self.failures.saturating_add(1);
        CaptureDecision::Reconnect(cfg)
    }

    /// Record why the attempt this frame authorised failed to open.
    pub fn note_open_error(&mut self, error: impl Into<String>) {
        self.last_error = Some(error.into());
    }

    /// The wait that now applies before the next attempt.
    pub fn delay(&self) -> Duration {
        audio_reconnect_delay(self.failures)
    }

    /// Consecutive attempts that have not yet produced sustained capture.
    pub fn failures(&self) -> u32 {
        self.failures
    }

    /// Current health, for hosts that surface it. `live` is the same fact passed
    /// to [`Self::poll`].
    pub fn health(&self, live: bool) -> CaptureHealth {
        if live {
            return CaptureHealth::Live;
        }
        CaptureHealth::Reconnecting {
            failed_attempts: self.failures,
            next_attempt_in: self.delay(),
            last_error: self.last_error.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOOPBACK: CaptureConfig = CaptureConfig {
        prefer_loopback: true,
    };
    const MIC: CaptureConfig = CaptureConfig {
        prefer_loopback: false,
    };

    #[test]
    fn ladder_is_immediate_then_escalates_and_saturates() {
        assert_eq!(audio_reconnect_delay(0), Duration::ZERO);
        assert_eq!(audio_reconnect_delay(1), AUDIO_RECONNECT_BASE_DELAY);
        assert_eq!(audio_reconnect_delay(2), Duration::from_secs(4));
        assert_eq!(audio_reconnect_delay(3), Duration::from_secs(8));
        assert_eq!(audio_reconnect_delay(4), Duration::from_secs(16));
        assert_eq!(audio_reconnect_delay(5), AUDIO_RECONNECT_MAX_DELAY);
        assert_eq!(audio_reconnect_delay(64), AUDIO_RECONNECT_MAX_DELAY);
        assert_eq!(audio_reconnect_delay(u32::MAX), AUDIO_RECONNECT_MAX_DELAY);
        // Boundary is exact: fires ON the deadline, not one frame later.
        assert!(should_try_audio_reconnect(1, AUDIO_RECONNECT_BASE_DELAY));
        assert!(!should_try_audio_reconnect(
            1,
            AUDIO_RECONNECT_BASE_DELAY - Duration::from_millis(1)
        ));
    }

    #[test]
    fn a_dead_stream_produces_a_bounded_escalating_restart_ladder() {
        let mut state = ReconnectState::default();
        // First dead frame reconnects immediately — the loss may already be over.
        assert_eq!(
            state.poll(false, Duration::ZERO, LOOPBACK),
            CaptureDecision::Reconnect(LOOPBACK)
        );
        // 60 fps for 40 s, never live.
        let mut attempts = vec![Duration::ZERO];
        for frame in 1..=2400u64 {
            let now = Duration::from_micros(frame * 16_667);
            if let CaptureDecision::Reconnect(_) = state.poll(false, now, LOOPBACK) {
                attempts.push(now);
            }
        }
        // 0 s, +2, +4, +8, +16 → 5 attempts inside 40 s, then the 30 s cap.
        assert_eq!(
            attempts.len(),
            5,
            "expected a bounded ladder, got {attempts:?}"
        );
        let gaps: Vec<Duration> = attempts.windows(2).map(|w| w[1] - w[0]).collect();
        for (gap, want) in gaps.iter().zip([
            AUDIO_RECONNECT_BASE_DELAY,
            Duration::from_secs(4),
            Duration::from_secs(8),
            Duration::from_secs(16),
        ]) {
            assert!(
                *gap >= want && *gap < want + Duration::from_millis(20),
                "gap {gap:?} not ~{want:?} (all {gaps:?})"
            );
        }
        assert_eq!(state.delay(), AUDIO_RECONNECT_MAX_DELAY);
    }

    /// The flapping-device replay. A device that opens cleanly and reports live for
    /// exactly one frame before dying must NOT reset the ladder — the defect two
    /// rounds of review found in the OjoDrop original, which reopened the stream
    /// 300 times in 600 frames.
    #[test]
    fn a_single_live_frame_never_credits_recovery() {
        let mut state = ReconnectState::default();
        let mut reconnects = 0;
        for frame in 0..600u64 {
            let now = Duration::from_micros(frame * 16_667);
            // Live for exactly one frame out of every two: the flap.
            let live = frame % 2 == 1;
            if let CaptureDecision::Reconnect(_) = state.poll(live, now, LOOPBACK) {
                reconnects += 1;
            }
        }
        // 600 frames is 10 s, so the ladder (0 s, +2, +4, +8) permits exactly 3
        // attempts. The pre-fix behaviour a one-poll reset produced was 300.
        assert_eq!(
            reconnects, 3,
            "flapping device reopened the stream {reconnects} times in 600 frames"
        );
        assert!(
            state.delay() > AUDIO_RECONNECT_BASE_DELAY,
            "the ladder must have escalated past the base rung, delay {:?}",
            state.delay()
        );
    }

    #[test]
    fn only_sustained_liveness_resets_the_ladder() {
        let mut state = ReconnectState::with_failures(3);
        state.note_open_error("device busy");
        let t0 = Duration::from_secs(100);
        assert_eq!(state.poll(true, t0, LOOPBACK), CaptureDecision::Idle);
        assert_eq!(state.failures(), 3, "one live frame is not recovery");
        let almost = t0 + AUDIO_RECONNECT_SETTLE - Duration::from_millis(1);
        assert_eq!(state.poll(true, almost, LOOPBACK), CaptureDecision::Idle);
        assert_eq!(state.failures(), 3);
        // A dead frame breaks the run; credit must restart from zero. That poll may
        // also authorise an attempt (the ladder's clock has long since elapsed), so
        // compare against the count it leaves behind rather than a literal.
        state.poll(false, almost, LOOPBACK);
        let after_dead_frame = state.failures();
        assert!(after_dead_frame >= 3);
        let resumed = almost + Duration::from_millis(1);
        assert_eq!(state.poll(true, resumed, LOOPBACK), CaptureDecision::Idle);
        assert_eq!(
            state.failures(),
            after_dead_frame,
            "the settle run must have restarted, not carried its old credit"
        );
        assert_eq!(
            state.poll(true, resumed + AUDIO_RECONNECT_SETTLE, LOOPBACK),
            CaptureDecision::Idle
        );
        assert_eq!(state.failures(), 0, "sustained liveness resets the ladder");
        assert_eq!(state.health(true), CaptureHealth::Live);
    }

    /// Device fallback: a persistently-failing loopback preference must not starve
    /// the plain mic, and must not permanently give up on loopback either.
    #[test]
    fn device_fallback_alternates_once_loopback_keeps_failing() {
        // Below the threshold, the request is honoured verbatim.
        for failures in 0..AUDIO_FALLBACK_AFTER_FAILURES {
            assert_eq!(next_capture_config(LOOPBACK, failures), LOOPBACK);
        }
        // From the threshold on, alternate — mic, loopback, mic, loopback…
        assert_eq!(next_capture_config(LOOPBACK, 2), MIC);
        assert_eq!(next_capture_config(LOOPBACK, 3), LOOPBACK);
        assert_eq!(next_capture_config(LOOPBACK, 4), MIC);
        assert_eq!(next_capture_config(LOOPBACK, 5), LOOPBACK);
        // A host that never asked for loopback has no alternate to fall back to.
        for failures in [0, 1, 2, 3, 9, u32::MAX] {
            assert_eq!(next_capture_config(MIC, failures), MIC);
        }
    }

    /// The fallback rule reaches the frame loop, not just the helper: replay a
    /// never-succeeding loopback device and confirm the decisions alternate.
    #[test]
    fn the_frame_loop_actually_receives_the_fallback_config() {
        let mut state = ReconnectState::default();
        let mut chosen = Vec::new();
        for frame in 0..6000u64 {
            let now = Duration::from_micros(frame * 16_667);
            if let CaptureDecision::Reconnect(cfg) = state.poll(false, now, LOOPBACK) {
                chosen.push(cfg.prefer_loopback);
            }
        }
        assert!(chosen.len() >= 5, "too few attempts: {chosen:?}");
        assert_eq!(
            &chosen[..5],
            &[true, true, false, true, false],
            "expected two honoured attempts then alternation"
        );
    }

    #[test]
    fn health_exposes_the_failure_state_a_host_needs() {
        let mut state = ReconnectState::default();
        assert_eq!(state.health(true), CaptureHealth::Live);
        state.poll(false, Duration::ZERO, LOOPBACK);
        state.note_open_error("no default input device");
        match state.health(false) {
            CaptureHealth::Reconnecting {
                failed_attempts,
                next_attempt_in,
                last_error,
            } => {
                assert_eq!(failed_attempts, 1);
                assert_eq!(next_attempt_in, AUDIO_RECONNECT_BASE_DELAY);
                assert_eq!(last_error.as_deref(), Some("no default input device"));
            }
            other => panic!("expected Reconnecting, got {other:?}"),
        }
        // Settling clears the recorded error so a stale cause is never shown.
        state.poll(true, Duration::from_secs(10), LOOPBACK);
        state.poll(true, Duration::from_secs(20), LOOPBACK);
        assert_eq!(state.health(true), CaptureHealth::Live);
        assert_eq!(state.failures(), 0);
    }
}
