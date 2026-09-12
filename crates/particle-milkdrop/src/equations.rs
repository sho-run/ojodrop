// EEL2 per-frame equation evaluator for MilkDrop presets.
//
// Targets Butterchurn's ns-eel2 semantics (milkdrop-eel-parser grammar +
// presetBase.js runtime). All values are JS doubles → we use f64.
//
// Supports the subset actually used by presets:
//   - Variable assignment:   var = expr
//   - Compound assignment:   var += -= *= /= %= expr
//   - Buffer access/assign:  megabuf(i), gmegabuf(i), megabuf(i) = v, ...
//   - Arithmetic:            + - * / % ^ (^ = power, not xor)
//   - Comparison:            < > <= >= == !=  (return 0.0 or 1.0, EPSILON for ==/!=)
//   - Logical:               && ||  !
//   - Control flow:          if(c,t,f), exec2/exec3, loop(n,..), while(..)
//   - Functions:             above, below, equal, bnot, band, bor, abs, sin, cos,
//                            tan, asin, acos, atan, atan2, sqrt, sqr, pow, log,
//                            log10, exp, min, max, floor, ceil, int, sign, clamp,
//                            lerp, rand, randint, bitor, bitand, sigmoid,
//                            megabuf, gmegabuf
//   - Comments:              // to end of line

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// EEL variable environment.
///
/// Interns variable names to dense slot indices so per-vertex / per-point / per-
/// shape evaluation does **no** heap allocation: once a name is first seen it maps
/// to a stable slot in a reused `Vec<f64>`, and [`Env::clear`] resets presence via
/// a generation stamp without dropping any key or freeing any buffer. This replaces
/// the previous per-frame `HashMap<String, f64>` whose every `insert`, `clear`, and
/// per-vertex reset reallocated key strings — O(vertices) allocations per frame.
pub struct Env {
    /// Stable identity used by compiled programs to cache name -> slot bindings.
    /// A clone receives a fresh identity because it can subsequently intern names
    /// independently of its source environment.
    id: u64,
    /// Variable name → dense slot index. Grows monotonically; a key string is
    /// allocated only the first time a given name is ever inserted.
    ids: HashMap<String, u32>,
    /// Slot values, indexed by the interned id.
    vals: Vec<f64>,
    /// Per-slot generation stamp. A slot is "present" iff `stamp[id] == gen`.
    stamp: Vec<u32>,
    /// Current generation. `clear()` bumps it so all slots read as absent again
    /// without touching the buffers.
    gen: u32,
}

/// Opaque dense slot in an [`Env`]. Callers that repeatedly seed/read a known
/// variable set can intern the names once and use these handles without touching
/// the string table on the hot path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EnvSlot(u32);

/// Dense, allocation-stable snapshot of a selected set of [`Env`] slots.
///
/// MilkDrop's per-pixel runner deliberately restores only its ten authored warp
/// controls between mesh vertices; user temporaries are allowed to carry to the
/// next vertex. Keeping the selected slot list here preserves that behavior while
/// avoiding name lookups and allocations in the vertex loop.
#[derive(Default)]
pub(crate) struct EnvSnapshot {
    env_id: u64,
    slots: Vec<EnvSlot>,
    values: Vec<f64>,
}

static NEXT_ENV_ID: AtomicU64 = AtomicU64::new(1);

impl Clone for Env {
    fn clone(&self) -> Self {
        Self {
            id: NEXT_ENV_ID.fetch_add(1, Ordering::Relaxed),
            ids: self.ids.clone(),
            vals: self.vals.clone(),
            stamp: self.stamp.clone(),
            gen: self.gen,
        }
    }
}

impl Default for Env {
    fn default() -> Self {
        // `gen` starts at 1 so a freshly pushed slot (stamp 0) reads as absent.
        Self {
            id: NEXT_ENV_ID.fetch_add(1, Ordering::Relaxed),
            ids: HashMap::new(),
            vals: Vec::new(),
            stamp: Vec::new(),
            gen: 1,
        }
    }
}

impl Env {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern `name`, returning its dense slot index. Allocates a key string only
    /// the first time a name is ever seen.
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            id
        } else {
            let id = self.vals.len() as u32;
            self.vals.push(0.0);
            self.stamp.push(0); // < any real generation (gen starts at 1)
            self.ids.insert(name.to_string(), id);
            id
        }
    }

    #[inline]
    fn get_slot(&self, id: u32) -> f64 {
        let id = id as usize;
        if self.stamp.get(id).copied() == Some(self.gen) {
            self.vals[id]
        } else {
            0.0
        }
    }

    #[inline]
    fn set_slot(&mut self, id: u32, value: f64) {
        let id = id as usize;
        self.vals[id] = value;
        self.stamp[id] = self.gen;
    }

    /// Intern a variable once for a caller-managed hot path.
    pub(crate) fn intern_slot(&mut self, name: &str) -> EnvSlot {
        EnvSlot(self.intern(name))
    }

    /// Write a previously interned slot without a string/hash lookup.
    #[inline]
    pub(crate) fn set_slot_value(&mut self, slot: EnvSlot, value: f64) {
        self.set_slot(slot.0, value);
    }

    /// Read a previously interned slot without a string/hash lookup.
    #[inline]
    pub(crate) fn slot_value(&self, slot: EnvSlot) -> f64 {
        self.get_slot(slot.0)
    }

    /// Copy a cached variable set from another environment without repeating
    /// string interning or hash-table lookups. `source_slots` and
    /// `destination_slots` must have been interned in their corresponding
    /// environments in the same name order.
    pub(crate) fn copy_slot_values_from(
        &mut self,
        destination_slots: &[EnvSlot],
        source: &Env,
        source_slots: &[EnvSlot],
    ) {
        debug_assert_eq!(destination_slots.len(), source_slots.len());
        for (&destination, &source_slot) in destination_slots.iter().zip(source_slots) {
            self.set_slot_value(destination, source.slot_value(source_slot));
        }
    }

    /// Set `name` to `v`. After a name's first insert this reuses its slot with no
    /// allocation (the hot per-vertex path).
    pub fn insert(&mut self, name: &str, v: f64) {
        let id = self.intern(name) as usize;
        self.vals[id] = v;
        self.stamp[id] = self.gen;
    }

    /// Read `name`, or `None` if unset in the current generation. Returns `&f64`
    /// (not `f64`) to keep the existing `.copied()` call sites unchanged.
    pub fn get(&self, name: &str) -> Option<&f64> {
        let id = *self.ids.get(name)? as usize;
        if self.stamp[id] == self.gen {
            Some(&self.vals[id])
        } else {
            None
        }
    }

    /// Reset every variable to absent, O(1), without dropping keys or freeing the
    /// value buffer (so capacity is retained across per-vertex reuse).
    pub fn clear(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            // Wraparound: retire every stamp so a stale value cannot alias gen 0.
            for s in &mut self.stamp {
                *s = 0;
            }
            self.gen = 1;
        }
    }

    /// Replace the current variables with the active values from another
    /// environment. This crosses two independently interned slot tables, so it
    /// performs name lookups once at the frame boundary; selected control slots
    /// can then be snapshotted/restored densely inside the vertex loop.
    pub(crate) fn copy_present_from(&mut self, source: &Env) {
        self.clear();
        for (name, &id) in &source.ids {
            let id = id as usize;
            if source.stamp[id] == source.gen {
                self.insert(name, source.vals[id]);
            }
        }
    }

    /// Capture selected slots into reusable dense storage. Existing snapshot
    /// capacity is retained across frames.
    pub(crate) fn capture_slots_into(&self, slots: &[EnvSlot], snapshot: &mut EnvSnapshot) {
        snapshot.env_id = self.id;
        snapshot.slots.clear();
        snapshot.values.clear();
        if snapshot.slots.capacity() < slots.len() {
            snapshot.slots.reserve(slots.len() - snapshot.slots.len());
        }
        if snapshot.values.capacity() < slots.len() {
            snapshot.values.reserve(slots.len() - snapshot.values.len());
        }
        for &slot in slots {
            snapshot.slots.push(slot);
            snapshot.values.push(self.slot_value(slot));
        }
    }

    /// Restore only the selected slots captured from this environment. Other
    /// variables remain untouched so per-pixel user state carries between mesh
    /// vertices exactly as it does in MilkDrop/Butterchurn.
    pub(crate) fn restore_slots(&mut self, snapshot: &EnvSnapshot) {
        debug_assert_eq!(snapshot.env_id, self.id);
        for (&slot, &value) in snapshot.slots.iter().zip(&snapshot.values) {
            self.set_slot_value(slot, value);
        }
    }

    /// Snapshot the present `(name, value)` pairs. Used once per frame at the
    /// per-frame → per-pixel env boundary (not on the per-vertex hot path).
    pub fn into_pairs(self) -> Vec<(String, f64)> {
        let Env {
            ids,
            vals,
            stamp,
            gen,
            ..
        } = self;
        ids.into_iter()
            .filter(|&(_, id)| stamp[id as usize] == gen)
            .map(|(k, id)| (k, vals[id as usize]))
            .collect()
    }

    /// Capacity of the dense value buffer (test hook: proves per-vertex reuse does
    /// not reallocate).
    #[cfg(test)]
    pub fn value_capacity(&self) -> usize {
        self.vals.capacity()
    }

    /// Number of interned names (test hook: proves no new names are interned once
    /// the variable set is warm).
    #[cfg(test)]
    pub fn interned_len(&self) -> usize {
        self.ids.len()
    }
}

impl std::ops::Index<&str> for Env {
    type Output = f64;
    fn index(&self, name: &str) -> &f64 {
        self.get(name).expect("no entry found for key")
    }
}

/// EEL value comparison epsilon (ns-eel2 uses 1e-5 for ==/!=/if/bnot etc.).
const EPS: f64 = 1e-5;
/// Maximum addressable megabuf/gmegabuf index (Butterchurn pre-fills 1<<20).
const MEGABUF_MAX: i64 = 1_048_576;
/// Per-loop cap plus per-program cumulative budget for loop/while. Butterchurn's
/// 1<<20 guard is too large for a render-thread evaluator; normal MilkDrop loops
/// are tiny, and hostile/buggy presets should yield quickly.
const LOOP_CAP: u64 = 16_384;
const LOOP_ITERATION_BUDGET: u64 = 16_384;
#[cfg(test)]
const EVAL_DEPTH_CAP: u32 = 256;
const PARSE_DEPTH_CAP: u32 = 256;
/// Maximum number of AST nodes a single statement may contain. [`PARSE_DEPTH_CAP`]
/// only bounds *recursive* descent (nested parens/calls/assignments). A flat,
/// left-associative chain — `1+1+1+…` — parses iteratively yet builds a tree whose
/// depth equals the operator count, and that tree is later walked/dropped
/// *recursively* (`Drop`, `eval`). An attacker-controlled chain therefore overflows
/// the host stack even though parsing itself does not. Capping the per-statement
/// node count bounds that downstream recursion. A real preset statement uses only a
/// handful of nodes, so this cap sits far above any legitimate input yet well below
/// the recursive-walk overflow threshold.
const MAX_PARSE_NODES: u32 = 4096;

/// Typed parse failure surfaced by [`EelProgram::try_parse`]. The infallible
/// [`EelProgram::parse`] maps it to an inert (empty) program instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// A statement exceeded [`MAX_PARSE_NODES`] AST nodes and was rejected before
    /// it could overflow the host stack when dropped/walked recursively.
    ExpressionTooLarge { limit: u32 },
}

/// Number of contiguous doubles stored in one lazily allocated megabuf page.
///
/// MilkDrop programs commonly walk contiguous ranges (particle arrays, point
/// histories, etc.). A `HashMap<i64, f64>` made every one of those accesses hash
/// an integer even though the address space is fixed and dense. Paging retains
/// the sparse/zero-filled behavior without eagerly allocating the full 8 MiB
/// buffer, while turning hot reads and writes into two indexed loads.
const MEGABUF_PAGE_LEN: usize = 4096;

/// Sparse, paged backing store for megabuf / gmegabuf.
#[derive(Default)]
pub struct MegaBuf {
    pages: Vec<Option<Box<[f64]>>>,
}

impl MegaBuf {
    #[inline]
    fn address(idx: f64) -> Option<(usize, usize)> {
        let i = idx.floor() as i64;
        if i < 0 || i >= MEGABUF_MAX {
            return None;
        }
        let i = i as usize;
        Some((i / MEGABUF_PAGE_LEN, i % MEGABUF_PAGE_LEN))
    }

    #[inline]
    fn read(&self, idx: f64) -> f64 {
        let Some((page_index, offset)) = Self::address(idx) else {
            return 0.0;
        };
        self.pages
            .get(page_index)
            .and_then(Option::as_deref)
            .map_or(0.0, |page| page[offset])
    }

    #[inline]
    fn write(&mut self, idx: f64, v: f64) -> f64 {
        let Some((page_index, offset)) = Self::address(idx) else {
            return v;
        };
        if self.pages.len() <= page_index {
            self.pages.resize_with(page_index + 1, || None);
        }
        let page = self.pages[page_index]
            .get_or_insert_with(|| vec![0.0; MEGABUF_PAGE_LEN].into_boxed_slice());
        page[offset] = v;
        v
    }

    #[cfg(test)]
    fn allocated_pages(&self) -> usize {
        self.pages.iter().filter(|page| page.is_some()).count()
    }
}

/// Default seed used by standalone/test EEL states. Renderers should create one
/// [`EelRng`] from a preset-derived seed and share it across every pool belonging
/// to that preset.
pub(crate) const DEFAULT_EEL_RNG_SEED: u64 = 0x1234_5678_9abc_def0;

/// Preset-owned EEL random stream.
///
/// EEL `rand()` is stateful. A process-global generator makes one preset's
/// output depend on which presets/tests happened to run before it, while a
/// generator per equation pool incorrectly restarts the sequence for shapes,
/// waves, and per-pixel equations. Sharing this handle gives every pool in one
/// renderer a single deterministic stream without coupling separate renderers.
#[derive(Debug)]
pub(crate) struct EelRng {
    state: AtomicU64,
}

impl EelRng {
    pub(crate) fn shared(seed: u64) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU64::new(seed),
        })
    }

    /// Reset the shared stream. Since every pool holds the same `Arc`, reseeding
    /// from the renderer resets the complete preset rather than one pool.
    pub(crate) fn reseed(&self, seed: u64) {
        self.state.store(seed, Ordering::Relaxed);
    }

    /// Advance the preset stream and return a uniform value in `[0, 1)`. The
    /// renderer may use this for MilkDrop's per-frame shader random vectors so
    /// they share the same preset-owned lifecycle as EEL `rand()`.
    pub(crate) fn next_unit(&self) -> f64 {
        // Atomic update keeps the shared stream data-race free. Custom-wave
        // programs that use rand() are classified as parallel-unsafe by the
        // compiler, so renderer scheduling still gives deterministic call order.
        let mut old = self.state.load(Ordering::Relaxed);
        loop {
            let new = old
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            match self
                .state
                .compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return (new >> 11) as f64 / (1u64 << 53) as f64,
                Err(observed) => old = observed,
            }
        }
    }
}

/// Runtime state threaded through an EelProgram run:
///   - `megabuf` is PER-POOL (each per-frame / shape / wave context has its own).
///   - `gmegabuf` is SHARED across the whole preset. The synchronized handle also
///     lets independent, buffer-free custom-wave pools run on the render worker pool.
///   - `rng` is SHARED across the whole preset so EEL rand() has one deterministic
///     lifecycle without leaking state between renderers.
pub struct EelState {
    pub megabuf: MegaBuf,
    pub gmegabuf: Arc<Mutex<MegaBuf>>,
    rng: Arc<EelRng>,
    /// Reusable argument stack for function-call evaluation. Persisting it on the
    /// (pool-lived) state means per-vertex `Expr::Call` evaluation reuses one
    /// growable buffer instead of allocating a fresh `Vec` per call per vertex.
    #[cfg(test)]
    args: Vec<f64>,
    /// Reused operand stack for the compiled slot-based evaluator.
    values: Vec<f64>,
}

impl EelState {
    /// New per-pool state with a private gmegabuf (use [`with_gmegabuf`] to share).
    pub fn new() -> Self {
        Self {
            megabuf: MegaBuf::default(),
            gmegabuf: Arc::new(Mutex::new(MegaBuf::default())),
            rng: EelRng::shared(DEFAULT_EEL_RNG_SEED),
            #[cfg(test)]
            args: Vec::new(),
            values: Vec::new(),
        }
    }
    /// New per-pool state sharing the given preset-wide gmegabuf.
    pub fn with_gmegabuf(gmegabuf: Arc<Mutex<MegaBuf>>) -> Self {
        Self::with_shared(gmegabuf, EelRng::shared(DEFAULT_EEL_RNG_SEED))
    }

    /// New per-pool state sharing both preset-wide resources. The renderer must
    /// use this constructor for its per-frame, per-pixel, shape, and wave pools.
    pub(crate) fn with_shared(gmegabuf: Arc<Mutex<MegaBuf>>, rng: Arc<EelRng>) -> Self {
        Self {
            megabuf: MegaBuf::default(),
            gmegabuf,
            rng,
            #[cfg(test)]
            args: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Capacity of the reusable call-argument stack (test hook).
    #[cfg(test)]
    pub fn arg_capacity(&self) -> usize {
        self.args.capacity()
    }

    #[cfg(test)]
    pub fn value_capacity(&self) -> usize {
        self.values.capacity()
    }
}

impl Default for EelState {
    fn default() -> Self {
        Self::new()
    }
}

// ── AST ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Expr {
    Num(f64),
    Var(String),
    Assign(String, Box<Expr>),
    /// megabuf/gmegabuf write: (is_global, index, value). Returns value.
    BufAssign(bool, Box<Expr>, Box<Expr>),
    BinOp(BinKind, Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Call(String, Vec<Expr>),
}

#[derive(Debug, Clone, Copy)]
enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

impl BinKind {
    fn from_assign_op(op: &str) -> Option<BinKind> {
        match op {
            "+=" => Some(BinKind::Add),
            "-=" => Some(BinKind::Sub),
            "*=" => Some(BinKind::Mul),
            "/=" => Some(BinKind::Div),
            "%=" => Some(BinKind::Mod),
            _ => None,
        }
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

pub struct EelProgram {
    #[cfg(test)]
    stmts: Vec<Expr>,
    compiled: CompiledProgram,
    bindings: RefCell<Bindings>,
}

#[derive(Default)]
struct Bindings {
    env_id: u64,
    slots: Vec<u32>,
}

impl EelProgram {
    /// Infallible constructor: a pathological / over-budget source yields an inert
    /// program (no statements) rather than panicking or overflowing the stack.
    /// Callers that need to distinguish a rejection use [`EelProgram::try_parse`].
    pub fn parse(src: &str) -> Self {
        Self::try_parse(src).unwrap_or_else(|_| Self::from_stmts(Vec::new()))
    }

    /// Parse, surfacing a typed [`ParseError`] when a statement exceeds the node
    /// budget (rather than building a tree that overflows the stack when later
    /// dropped or evaluated recursively).
    pub fn try_parse(src: &str) -> Result<Self, ParseError> {
        let mut parser = Parser::new(src);
        let stmts = parser.parse_program();
        match parser.node_error {
            Some(err) => Err(err),
            None => Ok(Self::from_stmts(stmts)),
        }
    }

    fn from_stmts(stmts: Vec<Expr>) -> Self {
        let compiled = CompiledProgram::compile(&stmts);
        Self {
            #[cfg(test)]
            stmts,
            compiled,
            bindings: RefCell::new(Bindings::default()),
        }
    }

    /// Run with a throwaway per-call megabuf/gmegabuf (back-compat path: presets
    /// that don't use buffers behave identically; buffer state is not persisted).
    /// Used by unit tests; the renderer always uses [`run_with`].
    #[allow(dead_code)]
    pub fn run(&self, env: &mut Env) {
        let mut state = EelState::new();
        self.run_with(env, &mut state);
    }

    /// Run with explicit, caller-owned buffer state (megabuf persists across
    /// frames for a pool; gmegabuf is shared across pools).
    pub fn run_with(&self, env: &mut Env, state: &mut EelState) {
        // A program can access gmegabuf many thousands of times inside loops.
        // Lock once for the complete execution, while leaving buffer-free programs
        // entirely unsynchronized so independent custom waves can still run in
        // parallel.
        let gmegabuf_handle = self
            .compiled
            .effects
            .uses_gmegabuf()
            .then(|| state.gmegabuf.clone());
        let mut gmegabuf_guard = gmegabuf_handle.as_ref().map(|handle| {
            handle
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        });
        self.run_with_optional_gmegabuf(env, state, gmegabuf_guard.as_deref_mut());
    }

    /// Run while the caller holds the preset-wide gmegabuf lock. Custom-wave
    /// point programs call this for an authored point batch, amortizing one lock
    /// over hundreds of point evaluations without changing their serial order.
    pub(crate) fn run_with_prelocked_gmegabuf(
        &self,
        env: &mut Env,
        state: &mut EelState,
        gmegabuf: &mut MegaBuf,
    ) {
        debug_assert!(self.compiled.effects.uses_gmegabuf());
        self.run_with_optional_gmegabuf(env, state, Some(gmegabuf));
    }

    fn run_with_optional_gmegabuf(
        &self,
        env: &mut Env,
        state: &mut EelState,
        mut gmegabuf: Option<&mut MegaBuf>,
    ) {
        let mut budget = EvalBudget::default();
        state.values.clear();

        let needs_bind = {
            let b = self.bindings.borrow();
            b.env_id != env.id || b.slots.len() != self.compiled.symbols.len()
        };
        if needs_bind {
            let slots = self
                .compiled
                .symbols
                .iter()
                .map(|name| env.intern(name))
                .collect();
            *self.bindings.borrow_mut() = Bindings {
                env_id: env.id,
                slots,
            };
        }
        let bindings = self.bindings.borrow();

        run_code(
            &self.compiled.code,
            &bindings.slots,
            env,
            state,
            &mut budget,
            &mut gmegabuf,
        );
        state.values.clear();
    }

    /// Approximate compiled operation count, used to identify expensive custom
    /// per-point programs for the optional adaptive sample LOD.
    pub(crate) fn operation_count(&self) -> usize {
        self.compiled.op_count
    }

    /// Whether the compiled program can read or write `name`. Renderer pools use
    /// this once at activation to seed only the q/t/reg globals their bytecode
    /// can observe, instead of copying all 140 globals for every shape instance.
    pub(crate) fn references_symbol(&self, name: &str) -> bool {
        self.compiled.symbols.iter().any(|symbol| symbol == name)
    }

    /// The full compile-time [`EffectSummary`] for this program. The renderer
    /// and later governors read the explicit fields (RNG, loops, private vs
    /// global buffers, output-write set, point/shape carry) to decide parallel
    /// execution, adaptive LOD, and shape-instance stratified sampling.
    #[allow(dead_code)] // Staged MILK-02 Rescue lever + tests.
    pub(crate) fn effect_summary(&self) -> EffectSummary {
        self.compiled.effects
    }

    pub(crate) fn uses_gmegabuf(&self) -> bool {
        self.compiled.effects.uses_gmegabuf()
    }

    /// Whether this program can run concurrently with a different custom-wave
    /// pool. Private env/megabuf state, loops, and point-to-point carry are safe:
    /// only preset-wide state such as gmegabuf and the shared RNG serialize waves.
    pub(crate) fn custom_wave_parallel_safe(&self) -> bool {
        self.compiled.effects.custom_wave_parallel_safe()
    }

    /// Whether adaptive custom-wave point LOD may skip evaluations without
    /// changing authored semantics. This is deliberately stricter than parallel
    /// safety: loops, buffers, random calls, and non-output assignments can all
    /// carry state between authored points and therefore retain full density.
    pub(crate) fn custom_wave_lod_safe(&self) -> bool {
        self.compiled.effects.custom_wave_lod_safe()
    }

    /// Whether the later MILK-02 shape-instance stratified-sampling lever may
    /// decimate this per-shape program without dropping carried state. Broader
    /// than [`custom_wave_lod_safe`](Self::custom_wave_lod_safe): the renderer
    /// reseeds regs/`t`/`q` and every shape output between instances, so only
    /// buffers, RNG, loops, and writes to custom (non-reset) names make a shape
    /// stateful across instances.
    #[allow(dead_code)] // Staged MILK-02 Rescue lever + tests.
    pub(crate) fn shape_instance_carry_safe(&self) -> bool {
        self.compiled.effects.shape_instance_carry_safe()
    }
}

// ── Slot-based bytecode -----------------------------------------------------

#[derive(Clone, Copy)]
enum Builtin {
    Above,
    Below,
    Equal,
    Div,
    Mod,
    Bnot,
    Band,
    Bor,
    Abs,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sqrt,
    InvSqrt,
    Sqr,
    Pow,
    Exp,
    Log,
    Log10,
    Min,
    Max,
    Floor,
    Ceil,
    Int,
    Sign,
    Clamp,
    Lerp,
    Sigmoid,
    BitOr,
    BitAnd,
    Rand,
    RandInt,
    Unknown,
}

impl Builtin {
    fn from_name(name: &str) -> Self {
        match name {
            "above" => Self::Above,
            "below" => Self::Below,
            "equal" => Self::Equal,
            "div" => Self::Div,
            "mod" => Self::Mod,
            "bnot" => Self::Bnot,
            "band" => Self::Band,
            "bor" => Self::Bor,
            "abs" => Self::Abs,
            "sin" => Self::Sin,
            "cos" => Self::Cos,
            "tan" => Self::Tan,
            "asin" => Self::Asin,
            "acos" => Self::Acos,
            "atan" => Self::Atan,
            "atan2" => Self::Atan2,
            "sqrt" => Self::Sqrt,
            "invsqrt" => Self::InvSqrt,
            "sqr" => Self::Sqr,
            "pow" => Self::Pow,
            "exp" => Self::Exp,
            "log" => Self::Log,
            "log10" => Self::Log10,
            "min" => Self::Min,
            "max" => Self::Max,
            "floor" => Self::Floor,
            "ceil" => Self::Ceil,
            "int" => Self::Int,
            "sign" => Self::Sign,
            "clamp" => Self::Clamp,
            "lerp" => Self::Lerp,
            "sigmoid" => Self::Sigmoid,
            "bitor" => Self::BitOr,
            "bitand" => Self::BitAnd,
            "rand" => Self::Rand,
            "randint" => Self::RandInt,
            _ => Self::Unknown,
        }
    }
}

enum Instr {
    Const(f64),
    Load(u32),
    Store(u32),
    Bin(BinKind),
    Neg,
    Not,
    Call(Builtin, usize),
    BufRead(bool),
    BufWrite(bool),
    If(Box<Code>, Box<Code>),
    Loop(Box<Code>),
    While(Box<Code>),
    Pop,
}

#[derive(Default)]
struct Code {
    ops: Vec<Instr>,
}

/// Bitmask over the six per-invocation reset outputs shared by per-pixel,
/// custom-wave, and custom-shape programs (`x y r g b a`). The renderer reseeds
/// these before every invocation, so assigning them never carries state forward.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct OutputWrites(u8);

impl OutputWrites {
    const X: u8 = 1 << 0;
    const Y: u8 = 1 << 1;
    const R: u8 = 1 << 2;
    const G: u8 = 1 << 3;
    const B: u8 = 1 << 4;
    const A: u8 = 1 << 5;

    /// The bit for a reset-output name, or `None` for any other symbol.
    fn bit_for(name: &str) -> Option<u8> {
        Some(match name {
            "x" => Self::X,
            "y" => Self::Y,
            "r" => Self::R,
            "g" => Self::G,
            "b" => Self::B,
            "a" => Self::A,
            _ => return None,
        })
    }

    fn set(&mut self, bit: u8) {
        self.0 |= bit;
    }

    /// Whether the program assigns reset output `name` (one of `x y r g b a`).
    #[allow(dead_code)] // Staged MILK-02 Rescue lever + tests.
    pub(crate) fn writes(&self, name: &str) -> bool {
        Self::bit_for(name).is_some_and(|bit| self.0 & bit != 0)
    }

    /// Whether the program assigns any of the six reset outputs.
    #[allow(dead_code)] // Staged MILK-02 Rescue lever + tests.
    pub(crate) fn any(&self) -> bool {
        self.0 != 0
    }

    /// Raw bitmask (`X|Y|R|G|B|A`); exposed for tests and downstream tooling.
    #[allow(dead_code)] // Staged MILK-02 Rescue lever + tests.
    pub(crate) fn bits(&self) -> u8 {
        self.0
    }
}

/// Whether writing `name` is reset by the renderer before the next custom-shape
/// instance runs, so the write cannot carry across instances. Mirrors the shape
/// per-instance reseed in `renderer.rs` (the per-instance loop reseeds
/// `reg00..reg99`, `t1..t8`, `q1..q32`, `instance`/`num_inst`, and every shape
/// output var). Any other (custom) name persists in the shape's env and does
/// carry. Frame globals are treated conservatively as non-reset here: shape code
/// virtually never assigns them, and flagging such a write as carry only makes
/// the shape-instance analysis stricter, never looser.
fn is_shape_reset_var(name: &str) -> bool {
    // Shape per-instance outputs plus the `instance`/`num_inst` indices, all
    // reseeded from base before each instance runs (`renderer.rs`
    // `ShapeEnvSlots` / `seed_shape_base_env` / per-instance loop).
    const SHAPE_OUTPUTS: [&str; 24] = [
        "x",
        "y",
        "r",
        "g",
        "b",
        "a",
        "r2",
        "g2",
        "b2",
        "a2",
        "rad",
        "ang",
        "sides",
        "border_r",
        "border_g",
        "border_b",
        "border_a",
        "thickoutline",
        "textured",
        "tex_ang",
        "tex_zoom",
        "additive",
        "instance",
        "num_inst",
    ];
    if SHAPE_OUTPUTS.contains(&name) {
        return true;
    }
    // `reg00..reg99` — three chars `reg` + exactly two ASCII digits.
    if let Some(digits) = name.strip_prefix("reg") {
        return digits.len() == 2 && digits.bytes().all(|byte| byte.is_ascii_digit());
    }
    // `q1..q32` and `t1..t8` — reseeded from the preset-global snapshot.
    if let Some(digits) = name.strip_prefix('q') {
        return matches!(digits.parse::<u32>(), Ok(n) if (1..=32).contains(&n));
    }
    if let Some(digits) = name.strip_prefix('t') {
        return matches!(digits.parse::<u32>(), Ok(n) if (1..=8).contains(&n));
    }
    false
}

/// Compile-time effect summary for one EEL program. Pure metadata computed once
/// at compile and consumed by the renderer to schedule parallel custom waves,
/// gate adaptive custom-wave point LOD, decide gmegabuf locking, and (later)
/// gate shape-instance stratified sampling. Nothing here changes evaluation or
/// rendering: the derived [`uses_gmegabuf`](Self::uses_gmegabuf),
/// [`custom_wave_parallel_safe`](Self::custom_wave_parallel_safe), and
/// [`custom_wave_lod_safe`](Self::custom_wave_lod_safe) values are byte-identical
/// to the three booleans they formalize.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct EffectSummary {
    /// Calls `rand` / `randint`; each call advances the shared RNG stream, so
    /// skipping or reordering invocations changes results.
    pub(crate) uses_rng: bool,
    /// Contains a `loop(...)` or `while(...)` construct.
    pub(crate) has_loops: bool,
    /// Reads the pool-private `megabuf`.
    pub(crate) reads_private_megabuf: bool,
    /// Writes the pool-private `megabuf`.
    pub(crate) writes_private_megabuf: bool,
    /// Reads the preset-global `gmegabuf`.
    pub(crate) reads_global_gmegabuf: bool,
    /// Writes the preset-global `gmegabuf`.
    pub(crate) writes_global_gmegabuf: bool,
    /// Which of the six reset outputs (`x y r g b a`) the program assigns.
    // Consumed by tests + the staged MILK-02 Rescue lever, not production yet.
    #[allow(dead_code)]
    pub(crate) output_writes: OutputWrites,
    /// Assigns at least one symbol outside the custom-wave reset outputs
    /// (`x y r g b a`); such a write persists to the next authored point.
    pub(crate) writes_non_wave_output: bool,
    /// Assigns at least one symbol outside the custom-shape reset set (see
    /// [`is_shape_reset_var`]); such a write persists to the next shape instance.
    // Read only through `shape_instance_carry_safe` (staged MILK-02 Rescue lever).
    #[allow(dead_code)]
    pub(crate) writes_non_shape_output: bool,
}

impl EffectSummary {
    /// Reads or writes the pool-private `megabuf`.
    pub(crate) fn uses_private_megabuf(&self) -> bool {
        self.reads_private_megabuf || self.writes_private_megabuf
    }

    /// Reads or writes the preset-global `gmegabuf`.
    pub(crate) fn uses_global_gmegabuf(&self) -> bool {
        self.reads_global_gmegabuf || self.writes_global_gmegabuf
    }

    /// Touches either buffer (private or global) in any direction.
    pub(crate) fn uses_any_megabuf(&self) -> bool {
        self.uses_private_megabuf() || self.uses_global_gmegabuf()
    }

    /// Formalizes today's `uses_gmegabuf`: the program reads or writes the
    /// preset-wide global buffer (private `megabuf` does not count).
    pub(crate) fn uses_gmegabuf(&self) -> bool {
        self.uses_global_gmegabuf()
    }

    /// Formalizes today's `custom_wave_parallel_safe`: distinct wave pools may
    /// run concurrently unless they share preset-wide state (`gmegabuf`) or the
    /// shared RNG stream. Private env/megabuf/point-carry stay pool-local.
    pub(crate) fn custom_wave_parallel_safe(&self) -> bool {
        !(self.uses_global_gmegabuf() || self.uses_rng)
    }

    /// Formalizes today's `custom_wave_lod_safe`: adaptive point LOD may skip
    /// invocations only for pure, output-only programs. Buffers, RNG, loops, and
    /// non-output assignments all carry state between authored points.
    pub(crate) fn custom_wave_lod_safe(&self) -> bool {
        !(self.uses_rng || self.has_loops || self.uses_any_megabuf() || self.writes_non_wave_output)
    }

    /// Whether per-point / per-instance state persists across invocations
    /// through a non-output register/variable write or any megabuf/gmegabuf
    /// access. (RNG carry is reported separately via [`uses_rng`](Self::uses_rng).)
    #[allow(dead_code)] // Staged MILK-02 Rescue lever + tests.
    pub(crate) fn point_carry(&self) -> bool {
        self.writes_non_wave_output || self.uses_any_megabuf()
    }

    /// Whether the later MILK-02 shape-instance stratified-sampling lever may
    /// decimate this per-shape program. Analogous to
    /// [`custom_wave_lod_safe`](Self::custom_wave_lod_safe), but the renderer
    /// reseeds regs/`t`/`q` and every shape output between instances, so those
    /// writes do not carry — only buffers, RNG, loops, and writes to custom
    /// (non-reset) names make a shape stateful across instances.
    #[allow(dead_code)] // Staged MILK-02 Rescue lever + tests.
    pub(crate) fn shape_instance_carry_safe(&self) -> bool {
        !(self.uses_rng
            || self.has_loops
            || self.uses_any_megabuf()
            || self.writes_non_shape_output)
    }
}

struct CompiledProgram {
    code: Code,
    symbols: Vec<String>,
    op_count: usize,
    effects: EffectSummary,
}

struct Compiler {
    symbols: Vec<String>,
    symbol_ids: HashMap<String, u32>,
    op_count: usize,
    effects: EffectSummary,
}

impl CompiledProgram {
    fn compile(stmts: &[Expr]) -> Self {
        let mut compiler = Compiler {
            symbols: Vec::new(),
            symbol_ids: HashMap::new(),
            op_count: 0,
            effects: EffectSummary::default(),
        };
        let mut code = Code::default();
        for stmt in stmts {
            compiler.expr(stmt, &mut code);
            compiler.push(&mut code, Instr::Pop);
        }
        Self {
            code,
            symbols: compiler.symbols,
            op_count: compiler.op_count,
            effects: compiler.effects,
        }
    }
}

impl Compiler {
    fn symbol(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.symbol_ids.get(name) {
            id
        } else {
            let id = self.symbols.len() as u32;
            self.symbols.push(name.to_owned());
            self.symbol_ids.insert(name.to_owned(), id);
            id
        }
    }

    fn push(&mut self, code: &mut Code, op: Instr) {
        self.op_count += 1;
        code.ops.push(op);
    }

    fn value_block(&mut self, exprs: &[Expr]) -> Code {
        let mut code = Code::default();
        if exprs.is_empty() {
            self.push(&mut code, Instr::Const(0.0));
            return code;
        }
        for (i, expr) in exprs.iter().enumerate() {
            self.expr(expr, &mut code);
            if i + 1 != exprs.len() {
                self.push(&mut code, Instr::Pop);
            }
        }
        code
    }

    fn expr(&mut self, expr: &Expr, code: &mut Code) {
        match expr {
            Expr::Num(value) => self.push(code, Instr::Const(*value)),
            Expr::Var(name) => {
                let id = self.symbol(name);
                self.push(code, Instr::Load(id));
            }
            Expr::Assign(name, rhs) => {
                // The renderer resets the six custom-wave outputs for every
                // point. Any other assignment may intentionally carry into the
                // next authored point, so downsampling would change semantics.
                let name = name.as_str();
                if let Some(bit) = OutputWrites::bit_for(name) {
                    self.effects.output_writes.set(bit);
                } else {
                    self.effects.writes_non_wave_output = true;
                }
                // The shape per-instance reseed is broader (regs, t*, q*, and
                // every shape output), so only custom names carry between shape
                // instances.
                if !is_shape_reset_var(name) {
                    self.effects.writes_non_shape_output = true;
                }
                self.expr(rhs, code);
                let id = self.symbol(name);
                self.push(code, Instr::Store(id));
            }
            Expr::BufAssign(global, index, value) => {
                if *global {
                    self.effects.writes_global_gmegabuf = true;
                } else {
                    self.effects.writes_private_megabuf = true;
                }
                self.expr(index, code);
                self.expr(value, code);
                self.push(code, Instr::BufWrite(*global));
            }
            Expr::BinOp(kind, lhs, rhs) => {
                self.expr(lhs, code);
                self.expr(rhs, code);
                self.push(code, Instr::Bin(*kind));
            }
            Expr::Neg(value) => {
                self.expr(value, code);
                self.push(code, Instr::Neg);
            }
            Expr::Not(value) => {
                self.expr(value, code);
                self.push(code, Instr::Not);
            }
            Expr::Call(name, args) => match name.as_str() {
                "if" | "If" | "IF" => {
                    if let Some(cond) = args.first() {
                        self.expr(cond, code);
                    } else {
                        self.push(code, Instr::Const(0.0));
                    }
                    let then_code =
                        self.value_block(args.get(1).map(std::slice::from_ref).unwrap_or(&[]));
                    let else_code =
                        self.value_block(args.get(2).map(std::slice::from_ref).unwrap_or(&[]));
                    self.push(code, Instr::If(Box::new(then_code), Box::new(else_code)));
                }
                "exec2" | "exec3" => {
                    let block = self.value_block(args);
                    for op in block.ops {
                        code.ops.push(op);
                    }
                }
                "loop" => {
                    self.effects.has_loops = true;
                    if let Some(count) = args.first() {
                        self.expr(count, code);
                    } else {
                        self.push(code, Instr::Const(0.0));
                    }
                    let body = self.value_block(args.get(1..).unwrap_or(&[]));
                    self.push(code, Instr::Loop(Box::new(body)));
                }
                "while" => {
                    self.effects.has_loops = true;
                    let body = self.value_block(args);
                    self.push(code, Instr::While(Box::new(body)));
                }
                "megabuf" | "gmegabuf" => {
                    let global = name == "gmegabuf";
                    if global {
                        self.effects.reads_global_gmegabuf = true;
                    } else {
                        self.effects.reads_private_megabuf = true;
                    }
                    if let Some(index) = args.first() {
                        self.expr(index, code);
                    } else {
                        self.push(code, Instr::Const(0.0));
                    }
                    self.push(code, Instr::BufRead(name == "gmegabuf"));
                }
                _ => {
                    let builtin = Builtin::from_name(name);
                    if matches!(builtin, Builtin::Rand | Builtin::RandInt) {
                        self.effects.uses_rng = true;
                    }
                    for arg in args {
                        self.expr(arg, code);
                    }
                    self.push(code, Instr::Call(builtin, args.len()));
                }
            },
        }
    }
}

#[inline]
fn pop_value(values: &mut Vec<f64>) -> f64 {
    values.pop().unwrap_or(0.0)
}

fn run_value_block(
    code: &Code,
    slots: &[u32],
    env: &mut Env,
    state: &mut EelState,
    budget: &mut EvalBudget,
    gmegabuf: &mut Option<&mut MegaBuf>,
) -> f64 {
    let base = state.values.len();
    run_code(code, slots, env, state, budget, gmegabuf);
    let value = state.values.pop().unwrap_or(0.0);
    state.values.truncate(base);
    value
}

fn run_code(
    code: &Code,
    slots: &[u32],
    env: &mut Env,
    state: &mut EelState,
    budget: &mut EvalBudget,
    gmegabuf: &mut Option<&mut MegaBuf>,
) {
    for op in &code.ops {
        match op {
            Instr::Const(value) => state.values.push(*value),
            Instr::Load(symbol) => state.values.push(env.get_slot(slots[*symbol as usize])),
            Instr::Store(symbol) => {
                let value = state.values.last().copied().unwrap_or(0.0);
                env.set_slot(slots[*symbol as usize], value);
            }
            Instr::Bin(kind) => {
                let rhs = pop_value(&mut state.values);
                let lhs = pop_value(&mut state.values);
                state.values.push(eval_bin(*kind, lhs, rhs));
            }
            Instr::Neg => {
                let value = pop_value(&mut state.values);
                state.values.push(-value);
            }
            Instr::Not => {
                let value = pop_value(&mut state.values);
                state.values.push(if value.abs() > EPS { 0.0 } else { 1.0 });
            }
            Instr::Call(function, argc) => {
                let base = state.values.len().saturating_sub(*argc);
                let value = eval_builtin(*function, &state.values[base..], &state.rng);
                state.values.truncate(base);
                state.values.push(value);
            }
            Instr::BufRead(global) => {
                let index = pop_value(&mut state.values);
                let value = if *global {
                    gmegabuf
                        .as_deref_mut()
                        .expect("compiled gmegabuf access without execution guard")
                        .read(index)
                } else {
                    state.megabuf.read(index)
                };
                state.values.push(value);
            }
            Instr::BufWrite(global) => {
                let value = pop_value(&mut state.values);
                let index = pop_value(&mut state.values);
                let value = if *global {
                    gmegabuf
                        .as_deref_mut()
                        .expect("compiled gmegabuf access without execution guard")
                        .write(index, value)
                } else {
                    state.megabuf.write(index, value)
                };
                state.values.push(value);
            }
            Instr::If(then_code, else_code) => {
                let condition = pop_value(&mut state.values);
                let branch = if condition.abs() > EPS {
                    then_code
                } else {
                    else_code
                };
                let value = run_value_block(branch, slots, env, state, budget, gmegabuf);
                state.values.push(value);
            }
            Instr::Loop(body) => {
                let count = pop_value(&mut state.values);
                let mut last = 0.0;
                let mut i = 0u64;
                while (i as f64) < count && i < LOOP_CAP && budget.spend_loop_iter() {
                    last = run_value_block(body, slots, env, state, budget, gmegabuf);
                    i += 1;
                }
                state.values.push(last);
            }
            Instr::While(body) => {
                let mut last = 0.0;
                let mut count = 0u64;
                loop {
                    if !budget.spend_loop_iter() {
                        break;
                    }
                    last = run_value_block(body, slots, env, state, budget, gmegabuf);
                    count += 1;
                    if last.abs() <= EPS || count >= LOOP_CAP {
                        break;
                    }
                }
                state.values.push(last);
            }
            Instr::Pop => {
                state.values.pop();
            }
        }
    }
}

#[inline]
fn eval_bin(kind: BinKind, lhs: f64, rhs: f64) -> f64 {
    match kind {
        BinKind::Add => lhs + rhs,
        BinKind::Sub => lhs - rhs,
        BinKind::Mul => lhs * rhs,
        BinKind::Div => {
            if rhs == 0.0 {
                0.0
            } else {
                lhs / rhs
            }
        }
        BinKind::Mod => {
            let divisor = rhs.floor();
            if divisor == 0.0 {
                0.0
            } else {
                (lhs.floor() as i64)
                    .checked_rem(divisor as i64)
                    .unwrap_or(0) as f64
            }
        }
        BinKind::Pow => lhs.powf(rhs),
        BinKind::Lt => (lhs < rhs) as i32 as f64,
        BinKind::Gt => (lhs > rhs) as i32 as f64,
        BinKind::Le => (lhs <= rhs) as i32 as f64,
        BinKind::Ge => (lhs >= rhs) as i32 as f64,
        BinKind::Eq => ((lhs - rhs).abs() < EPS) as i32 as f64,
        BinKind::Ne => ((lhs - rhs).abs() >= EPS) as i32 as f64,
        BinKind::And => ((lhs != 0.0) && (rhs != 0.0)) as i32 as f64,
        BinKind::Or => ((lhs != 0.0) || (rhs != 0.0)) as i32 as f64,
    }
}

// ── Evaluator ────────────────────────────────────────────────────────────────

struct EvalBudget {
    #[cfg(test)]
    depth: u32,
    remaining_loop_iters: u64,
}

impl Default for EvalBudget {
    fn default() -> Self {
        Self {
            #[cfg(test)]
            depth: 0,
            remaining_loop_iters: LOOP_ITERATION_BUDGET,
        }
    }
}

impl EvalBudget {
    #[cfg(test)]
    fn enter(&mut self) -> bool {
        if self.depth >= EVAL_DEPTH_CAP {
            return false;
        }
        self.depth += 1;
        true
    }

    #[cfg(test)]
    fn exit(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    fn spend_loop_iter(&mut self) -> bool {
        if self.remaining_loop_iters == 0 {
            return false;
        }
        self.remaining_loop_iters -= 1;
        true
    }
}

#[cfg(test)]
fn eval(e: &Expr, env: &mut Env, st: &mut EelState, budget: &mut EvalBudget) -> f64 {
    if !budget.enter() {
        return 0.0;
    }
    let value = match e {
        Expr::Num(v) => *v,
        Expr::Var(n) => *env.get(n.as_str()).unwrap_or(&0.0),
        Expr::Assign(n, rhs) => {
            let v = eval(rhs, env, st, budget);
            env.insert(n, v);
            v
        }
        Expr::BufAssign(is_global, idx, val) => {
            let i = eval(idx, env, st, budget);
            let v = eval(val, env, st, budget);
            if *is_global {
                st.gmegabuf
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .write(i, v)
            } else {
                st.megabuf.write(i, v)
            }
        }
        Expr::Neg(e) => -eval(e, env, st, budget),
        Expr::Not(e) => {
            if eval(e, env, st, budget).abs() > EPS {
                0.0
            } else {
                1.0
            }
        }
        Expr::BinOp(op, l, r) => {
            let lv = eval(l, env, st, budget);
            let rv = eval(r, env, st, budget);
            match op {
                BinKind::Add => lv + rv,
                BinKind::Sub => lv - rv,
                BinKind::Mul => lv * rv,
                // ns-eel2 div(): y==0 ? 0 : x/y
                BinKind::Div => {
                    if rv == 0.0 {
                        0.0
                    } else {
                        lv / rv
                    }
                }
                // ns-eel2 mod(): INTEGER mod — y==0 ? 0 : floor(x) % floor(y)
                BinKind::Mod => {
                    let d = rv.floor();
                    // checked_rem avoids the `i64::MIN % -1` overflow panic (debug/test builds);
                    // 0 is the correct result for any `x % -1`.
                    if d == 0.0 {
                        0.0
                    } else {
                        (lv.floor() as i64).checked_rem(d as i64).unwrap_or(0) as f64
                    }
                }
                BinKind::Pow => lv.powf(rv),
                BinKind::Lt => (lv < rv) as i32 as f64,
                BinKind::Gt => (lv > rv) as i32 as f64,
                BinKind::Le => (lv <= rv) as i32 as f64,
                BinKind::Ge => (lv >= rv) as i32 as f64,
                // ns-eel2 ==/!= use EPSILON tolerance
                BinKind::Eq => ((lv - rv).abs() < EPS) as i32 as f64,
                BinKind::Ne => ((lv - rv).abs() >= EPS) as i32 as f64,
                BinKind::And => ((lv != 0.0) && (rv != 0.0)) as i32 as f64,
                BinKind::Or => ((lv != 0.0) || (rv != 0.0)) as i32 as f64,
            }
        }
        Expr::Call(name, args) => eval_call_node(name, args, env, st, budget),
    };
    budget.exit();
    value
}

/// Handles control-flow functions that need LAZY / ordered evaluation, then
/// falls back to eager math functions in `eval_call`.
#[cfg(test)]
fn eval_call_node(
    name: &str,
    args: &[Expr],
    env: &mut Env,
    st: &mut EelState,
    budget: &mut EvalBudget,
) -> f64 {
    match name {
        // if(c, t, f): only the taken branch runs (side-effect safety).
        "if" | "If" | "IF" => {
            let c = eval(args.first().unwrap_or(&Expr::Num(0.0)), env, st, budget);
            if c.abs() > EPS {
                eval(args.get(1).unwrap_or(&Expr::Num(0.0)), env, st, budget)
            } else {
                eval(args.get(2).unwrap_or(&Expr::Num(0.0)), env, st, budget)
            }
        }
        // exec2(a, b) → eval in order, return last. exec3(a, b, c) likewise.
        "exec2" | "exec3" => {
            let mut last = 0.0;
            for a in args {
                last = eval(a, env, st, budget);
            }
            last
        }
        // loop(n, body...) → for(i=0; i<n; i++) body. n evaluated ONCE.
        // Multi-statement bodies parse as args[1..].
        "loop" => {
            let n = eval(args.first().unwrap_or(&Expr::Num(0.0)), env, st, budget);
            let mut last = 0.0;
            let mut i = 0u64;
            // Butterchurn compares i<n with float n (loop(3.9,..) → 4 iters).
            while (i as f64) < n && i < LOOP_CAP && budget.spend_loop_iter() {
                for a in &args[1..] {
                    last = eval(a, env, st, budget);
                }
                i += 1;
            }
            last
        }
        // while(body...) → run body; repeat while |last|>1e-5, hard cap.
        "while" => {
            let mut last = 0.0;
            let mut c = 0u64;
            loop {
                if !budget.spend_loop_iter() {
                    break;
                }
                for a in args {
                    last = eval(a, env, st, budget);
                }
                c += 1;
                if last.abs() <= EPS || c >= LOOP_CAP {
                    break;
                }
            }
            last
        }
        // megabuf/gmegabuf READ (write form is handled via BufAssign at parse).
        "megabuf" => {
            let i = eval(args.first().unwrap_or(&Expr::Num(0.0)), env, st, budget);
            st.megabuf.read(i)
        }
        "gmegabuf" => {
            let i = eval(args.first().unwrap_or(&Expr::Num(0.0)), env, st, budget);
            st.gmegabuf
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .read(i)
        }
        // Everything else: eager args. Evaluate onto the state's reusable arg
        // stack (no per-call heap allocation), call, then pop back to `base`.
        // Nested calls push above `base` and truncate themselves, so the slice
        // `[base..]` holds exactly this call's args.
        _ => {
            let base = st.args.len();
            for e in args {
                let v = eval(e, env, st, budget);
                st.args.push(v);
            }
            let result = eval_call(name, &st.args[base..], &st.rng);
            st.args.truncate(base);
            result
        }
    }
}

#[cfg(test)]
fn eval_call(name: &str, a: &[f64], rng: &EelRng) -> f64 {
    eval_builtin(Builtin::from_name(name), a, rng)
}

fn eval_builtin(function: Builtin, a: &[f64], rng: &EelRng) -> f64 {
    let get = |i: usize| a.get(i).copied().unwrap_or(0.0);
    match function {
        Builtin::Above => (get(0) > get(1)) as i32 as f64,
        Builtin::Below => (get(0) < get(1)) as i32 as f64,
        Builtin::Equal => ((get(0) - get(1)).abs() < EPS) as i32 as f64,
        // ns-eel2 div(x,y): y==0 ? 0 : x/y (matches the `/` BinKind::Div semantics).
        // Butterchurn's JS transpiler emits `div(a,b)` for `a/b` (EEL division).
        Builtin::Div => {
            let y = get(1);
            if y == 0.0 {
                0.0
            } else {
                get(0) / y
            }
        }
        // ns-eel2 mod(x,y): INTEGER mod — y==0 ? 0 : floor(x) % floor(y).
        // Mirrors the `%` BinKind::Mod logic exactly.
        Builtin::Mod => {
            let d = get(1).floor();
            // checked_rem avoids the `i64::MIN % -1` overflow panic; 0 is correct for `x % -1`.
            if d == 0.0 {
                0.0
            } else {
                (get(0).floor() as i64).checked_rem(d as i64).unwrap_or(0) as f64
            }
        }
        // ns-eel2 boolean ops (bnot was missing → ORB's edge-detect collapsed q3,
        // killing the warp-feedback tunnel accumulation).
        Builtin::Bnot => (get(0).abs() < EPS) as i32 as f64,
        Builtin::Band => ((get(0) != 0.0) && (get(1) != 0.0)) as i32 as f64,
        Builtin::Bor => ((get(0) != 0.0) || (get(1) != 0.0)) as i32 as f64,
        Builtin::Abs => get(0).abs(),
        Builtin::Sin => get(0).sin(),
        Builtin::Cos => get(0).cos(),
        Builtin::Tan => get(0).tan(),
        Builtin::Asin => get(0).asin(),
        Builtin::Acos => get(0).acos(),
        Builtin::Atan => get(0).atan(),
        Builtin::Atan2 => get(0).atan2(get(1)),
        // ns-eel2 sqrt(): sqrt(abs(x))
        Builtin::Sqrt => get(0).abs().sqrt(),
        Builtin::InvSqrt => {
            let s = get(0).sqrt();
            if s == 0.0 {
                0.0
            } else {
                1.0 / s
            }
        }
        Builtin::Sqr => get(0) * get(0),
        Builtin::Pow => {
            let z = get(0).powf(get(1));
            if z.is_finite() {
                z
            } else {
                0.0
            }
        }
        Builtin::Exp => get(0).exp(),
        Builtin::Log => get(0).ln(),
        Builtin::Log10 => get(0).log10(),
        Builtin::Min => get(0).min(get(1)),
        Builtin::Max => get(0).max(get(1)),
        Builtin::Floor => get(0).floor(),
        Builtin::Ceil => get(0).ceil(),
        Builtin::Int => get(0).trunc(),
        // ns-eel2 sign(): x>0?1 : x<0?-1 : 0  (signum() returns ±1 at 0 — differs)
        Builtin::Sign => {
            let x = get(0);
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        Builtin::Clamp => get(0).clamp(get(1), get(2)),
        Builtin::Lerp => get(0) + (get(1) - get(0)) * get(2),
        // ns-eel2 sigmoid(x,y): t=1+exp(-x*y); |t|>1e-5 ? 1/t : 0
        Builtin::Sigmoid => {
            let t = 1.0 + (-get(0) * get(1)).exp();
            if t.abs() > EPS {
                1.0 / t
            } else {
                0.0
            }
        }
        Builtin::BitOr => ((get(0).floor() as i64) | (get(1).floor() as i64)) as f64,
        Builtin::BitAnd => ((get(0).floor() as i64) & (get(1).floor() as i64)) as f64,
        Builtin::Rand => rand_eel(rng, get(0)),
        Builtin::RandInt => rand_eel(rng, get(0)).floor(),
        Builtin::Unknown => 0.0,
    }
}

/// ns-eel2 rand(x): xf=floor(x); xf<1 ? random() : random()*xf  (in [0, xf)).
fn rand_eel(rng: &EelRng, x: f64) -> f64 {
    let u = rng.next_unit();
    let xf = x.floor();
    if xf < 1.0 {
        u
    } else {
        u * xf
    }
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Op(String), // multi-char operators stored as string
    LParen,
    RParen,
    Comma,
    Semi,
    Eof,
}

struct Lexer {
    chars: Vec<char>,
    pos: usize,
}

impl Lexer {
    fn new(src: &str) -> Self {
        Self {
            chars: src.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> char {
        self.chars.get(self.pos).copied().unwrap_or('\0')
    }
    fn next(&mut self) -> char {
        let c = self.peek();
        self.pos += 1;
        c
    }
    fn eat_if(&mut self, c: char) -> bool {
        if self.peek() == c {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        // Iterative + budgeted: each pass consumes whitespace or a `//` comment,
        // both of which advance `pos`, so the loop is bounded by the remaining
        // input length. The budget is a defensive cap guaranteeing termination
        // even if a future edit introduced a non-advancing branch.
        let mut budget = self.chars.len().saturating_add(1);
        loop {
            if budget == 0 {
                return;
            }
            budget -= 1;
            while self.peek().is_ascii_whitespace() {
                self.next();
            }
            if self.peek() == '/' && self.chars.get(self.pos + 1) == Some(&'/') {
                while self.peek() != '\n' && self.peek() != '\0' {
                    self.next();
                }
            } else {
                break;
            }
        }
    }

    fn next_tok(&mut self) -> Tok {
        // Skip whitespace, comments, and unknown characters iteratively within a
        // bounded loop. A pathological input — a long run of unknown characters or
        // many consecutive comments — must not recurse the host stack (the previous
        // `_ => self.next_tok()` tail-recursed once per skipped unknown char) or
        // spin unboundedly. Each loop iteration consumes at least one character, so
        // the loop is bounded by the remaining input length; `budget` is a
        // defensive cap that guarantees termination even if a future edit added a
        // non-advancing branch.
        let mut budget = self.chars.len().saturating_add(1);
        loop {
            if budget == 0 {
                return Tok::Eof;
            }
            budget -= 1;

            self.skip_whitespace_and_comments();
            let c = self.peek();
            if c == '\0' {
                return Tok::Eof;
            }

            // Number
            if c.is_ascii_digit()
                || (c == '.'
                    && self
                        .chars
                        .get(self.pos + 1)
                        .map(|x| x.is_ascii_digit())
                        .unwrap_or(false))
            {
                let start = self.pos;
                while self.peek().is_ascii_digit() || self.peek() == '.' {
                    self.next();
                }
                if self.peek() == 'e' || self.peek() == 'E' {
                    self.next();
                    if self.peek() == '+' || self.peek() == '-' {
                        self.next();
                    }
                    while self.peek().is_ascii_digit() {
                        self.next();
                    }
                }
                let s: String = self.chars[start..self.pos].iter().collect();
                return Tok::Num(s.parse().unwrap_or(0.0));
            }

            // Identifier
            if c.is_ascii_alphabetic() || c == '_' {
                let start = self.pos;
                while self.peek().is_ascii_alphanumeric() || self.peek() == '_' {
                    self.next();
                }
                let s: String = self.chars[start..self.pos].iter().collect();
                return Tok::Ident(s);
            }

            self.next(); // consume c
            let tok = match c {
                '(' => Tok::LParen,
                ')' => Tok::RParen,
                ',' => Tok::Comma,
                ';' => Tok::Semi,
                // Compound-assignment operators desugar in the parser.
                '+' => {
                    if self.eat_if('=') {
                        Tok::Op("+=".into())
                    } else {
                        Tok::Op("+".into())
                    }
                }
                '-' => {
                    if self.eat_if('=') {
                        Tok::Op("-=".into())
                    } else {
                        Tok::Op("-".into())
                    }
                }
                '*' => {
                    if self.eat_if('=') {
                        Tok::Op("*=".into())
                    } else {
                        Tok::Op("*".into())
                    }
                }
                '/' => {
                    if self.eat_if('=') {
                        Tok::Op("/=".into())
                    } else {
                        Tok::Op("/".into())
                    }
                }
                '%' => {
                    if self.eat_if('=') {
                        Tok::Op("%=".into())
                    } else {
                        Tok::Op("%".into())
                    }
                }
                '^' => Tok::Op("^".into()),
                '!' => {
                    if self.eat_if('=') {
                        Tok::Op("!=".into())
                    } else {
                        Tok::Op("!".into())
                    }
                }
                '<' => {
                    if self.eat_if('=') {
                        Tok::Op("<=".into())
                    } else {
                        Tok::Op("<".into())
                    }
                }
                '>' => {
                    if self.eat_if('=') {
                        Tok::Op(">=".into())
                    } else {
                        Tok::Op(">".into())
                    }
                }
                '=' => {
                    if self.eat_if('=') {
                        Tok::Op("==".into())
                    } else {
                        Tok::Op("=".into())
                    }
                }
                '&' => {
                    if self.eat_if('&') {
                        Tok::Op("&&".into())
                    } else {
                        Tok::Op("&".into())
                    }
                }
                '|' => {
                    if self.eat_if('|') {
                        Tok::Op("||".into())
                    } else {
                        Tok::Op("|".into())
                    }
                }
                // Ternary punctuation (handled by parse_ternary).
                '?' => Tok::Op("?".into()),
                ':' => Tok::Op(":".into()),
                // Ignore unknown chars silently: rescan iteratively (never recurse).
                _ => continue,
            };
            return tok;
        }
    }
}

// ── Parser ───────────────────────────────────────────────────────────────────

struct Parser {
    tokens: Vec<Tok>,
    pos: usize,
    parse_depth: u32,
    /// AST nodes built for the current statement; bounded by [`MAX_PARSE_NODES`].
    /// Reset at the start of each statement in [`Parser::parse_program`].
    nodes: u32,
    /// First over-budget rejection, recorded while unwinding so the whole program
    /// can be rejected with a precise typed error.
    node_error: Option<ParseError>,
}

fn is_assign_op(s: &str) -> bool {
    matches!(s, "=" | "+=" | "-=" | "*=" | "/=" | "%=")
}

impl Parser {
    fn new(src: &str) -> Self {
        let mut lex = Lexer::new(src);
        let mut tokens = Vec::new();
        loop {
            let t = lex.next_tok();
            let done = t == Tok::Eof;
            tokens.push(t);
            if done {
                break;
            }
        }
        Self {
            tokens,
            pos: 0,
            parse_depth: 0,
            nodes: 0,
            node_error: None,
        }
    }

    /// Account for one freshly built AST node against [`MAX_PARSE_NODES`]. Returns
    /// `false` — recording an [`ParseError::ExpressionTooLarge`] on first breach —
    /// once the budget is exceeded, so the caller stops growing the tree (breaking
    /// out of flat operator loops / not recursing deeper) instead of overflowing
    /// the stack. Call it BEFORE recursing or before extending a flat chain.
    fn bump_node(&mut self) -> bool {
        self.nodes = self.nodes.saturating_add(1);
        if self.nodes > MAX_PARSE_NODES {
            if self.node_error.is_none() {
                self.node_error = Some(ParseError::ExpressionTooLarge {
                    limit: MAX_PARSE_NODES,
                });
            }
            return false;
        }
        true
    }

    fn over_budget(&self) -> bool {
        self.node_error.is_some()
    }

    fn peek(&self) -> &Tok {
        self.tokens.get(self.pos).unwrap_or(&Tok::Eof)
    }
    fn peek2(&self) -> &Tok {
        self.tokens.get(self.pos + 1).unwrap_or(&Tok::Eof)
    }

    fn eat(&mut self) -> Tok {
        let t = self.tokens.get(self.pos).cloned().unwrap_or(Tok::Eof);
        self.pos += 1;
        t
    }

    fn eat_semi(&mut self) {
        while matches!(self.peek(), Tok::Semi) {
            self.eat();
        }
    }

    fn enter_parse(&mut self) -> bool {
        if self.parse_depth >= PARSE_DEPTH_CAP {
            return false;
        }
        self.parse_depth += 1;
        true
    }

    fn exit_parse(&mut self) {
        self.parse_depth = self.parse_depth.saturating_sub(1);
    }

    fn skip_depth_limited_expr(&mut self) {
        let mut depth = 0i32;
        while !matches!(self.peek(), Tok::Eof) {
            match self.peek() {
                Tok::Comma | Tok::Semi if depth == 0 => break,
                Tok::RParen if depth == 0 => break,
                Tok::LParen => {
                    depth += 1;
                    self.eat();
                }
                Tok::RParen => {
                    depth -= 1;
                    self.eat();
                }
                _ => {
                    self.eat();
                }
            }
        }
    }

    fn parse_program(&mut self) -> Vec<Expr> {
        let mut stmts = Vec::new();
        while !matches!(self.peek(), Tok::Eof) {
            self.eat_semi();
            if matches!(self.peek(), Tok::Eof) {
                break;
            }
            // The node budget is per-statement: the recursive walk/drop it guards
            // happens per expression.
            self.nodes = 0;
            let e = self.parse_expr();
            stmts.push(e);
            // A statement blew the node budget → stop; `try_parse` rejects the
            // whole program. Bailing here also guarantees forward progress: an
            // exhausted budget makes descent methods return without consuming
            // tokens, which would otherwise spin the statement loop.
            if self.over_budget() {
                break;
            }
            self.eat_semi();
        }
        stmts
    }

    // Highest-level: assignment (right-assoc). Handles `=` and compound `+= …`.
    fn parse_expr(&mut self) -> Expr {
        if !self.enter_parse() {
            self.skip_depth_limited_expr();
            return Expr::Num(0.0);
        }
        let expr = self.parse_expr_inner();
        self.exit_parse();
        expr
    }

    fn parse_expr_inner(&mut self) -> Expr {
        // Look-ahead: `IDENT <assign-op> ...`  (plain var assignment).
        if let Tok::Ident(name) = self.peek().clone() {
            if let Tok::Op(op) = self.peek2().clone() {
                if is_assign_op(&op) {
                    let name = name.clone();
                    self.eat(); // ident
                    self.eat(); // assign-op
                    let rhs = self.parse_expr(); // right-associative
                    return match BinKind::from_assign_op(&op) {
                        // x += rhs  →  x = x <op> rhs
                        Some(k) => Expr::Assign(
                            name.clone(),
                            Box::new(Expr::BinOp(k, Box::new(Expr::Var(name)), Box::new(rhs))),
                        ),
                        None => Expr::Assign(name, Box::new(rhs)), // plain '='
                    };
                }
            }
        }

        // Look-ahead: buffer assignment `megabuf(idx) <assign-op> rhs`.
        if let Tok::Ident(name) = self.peek().clone() {
            if (name == "megabuf" || name == "gmegabuf") && matches!(self.peek2(), Tok::LParen) {
                if let Some(close) = self.matching_paren(self.pos + 1) {
                    if let Tok::Op(op) = self.tokens.get(close + 1).cloned().unwrap_or(Tok::Eof) {
                        if is_assign_op(&op) {
                            let is_global = name == "gmegabuf";
                            self.eat(); // ident
                            self.eat(); // '('
                            let idx = self.parse_expr();
                            if matches!(self.peek(), Tok::RParen) {
                                self.eat();
                            }
                            self.eat(); // assign-op
                            let rhs = self.parse_expr();
                            return match BinKind::from_assign_op(&op) {
                                Some(k) => {
                                    // buf(i) op= rhs → buf(i) = buf(i) op rhs
                                    let read = Expr::Call(name.clone(), vec![idx.clone()]);
                                    Expr::BufAssign(
                                        is_global,
                                        Box::new(idx),
                                        Box::new(Expr::BinOp(k, Box::new(read), Box::new(rhs))),
                                    )
                                }
                                None => Expr::BufAssign(is_global, Box::new(idx), Box::new(rhs)),
                            };
                        }
                    }
                }
            }
        }

        self.parse_ternary()
    }

    /// C-style ternary `cond ? a : b`, looser than `||` (the lowest binary level).
    /// Right-associative so `a ? b : c ? d : e` parses as `a ? b : (c ? d : e)`.
    /// Desugars to the existing lazy `if(cond, a, b)` so ONLY the taken branch
    /// evaluates (same side-effect safety as `if`). The JS-transpiled Butterchurn
    /// equations use ternaries heavily (e.g. `.00001<abs(above(d,r))?0:sin(...)`).
    fn parse_ternary(&mut self) -> Expr {
        let cond = self.parse_or();
        if matches!(self.peek(), Tok::Op(s) if s == "?") {
            self.eat(); // '?'
                        // Branches can themselves contain assignments/ternaries.
            let then_branch = self.parse_expr();
            if matches!(self.peek(), Tok::Op(s) if s == ":") {
                self.eat(); // ':'
            }
            let else_branch = self.parse_expr();
            Expr::Call("if".to_string(), vec![cond, then_branch, else_branch])
        } else {
            cond
        }
    }

    /// Given index of an LParen, return index of the matching RParen.
    fn matching_paren(&self, lparen: usize) -> Option<usize> {
        let mut depth = 0i32;
        let mut i = lparen;
        while i < self.tokens.len() {
            match self.tokens[i] {
                Tok::LParen => depth += 1,
                Tok::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                Tok::Eof => return None,
                _ => {}
            }
            i += 1;
        }
        None
    }

    fn parse_or(&mut self) -> Expr {
        let mut lhs = self.parse_and();
        while matches!(self.peek(), Tok::Op(s) if s == "||") {
            // Cap the flat chain length so the left-deep tree cannot overflow the
            // stack when dropped/evaluated recursively.
            if !self.bump_node() {
                break;
            }
            self.eat();
            let rhs = self.parse_and();
            lhs = Expr::BinOp(BinKind::Or, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_and(&mut self) -> Expr {
        let mut lhs = self.parse_cmp();
        while matches!(self.peek(), Tok::Op(s) if s == "&&") {
            if !self.bump_node() {
                break;
            }
            self.eat();
            let rhs = self.parse_cmp();
            lhs = Expr::BinOp(BinKind::And, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_cmp(&mut self) -> Expr {
        let mut lhs = self.parse_add();
        loop {
            let op = match self.peek() {
                Tok::Op(s) if s == "<" => BinKind::Lt,
                Tok::Op(s) if s == ">" => BinKind::Gt,
                Tok::Op(s) if s == "<=" => BinKind::Le,
                Tok::Op(s) if s == ">=" => BinKind::Ge,
                Tok::Op(s) if s == "==" => BinKind::Eq,
                Tok::Op(s) if s == "!=" => BinKind::Ne,
                _ => break,
            };
            if !self.bump_node() {
                break;
            }
            self.eat();
            let rhs = self.parse_add();
            lhs = Expr::BinOp(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_add(&mut self) -> Expr {
        let mut lhs = self.parse_mul();
        loop {
            let op = match self.peek() {
                Tok::Op(s) if s == "+" => BinKind::Add,
                Tok::Op(s) if s == "-" => BinKind::Sub,
                _ => break,
            };
            if !self.bump_node() {
                break;
            }
            self.eat();
            let rhs = self.parse_mul();
            lhs = Expr::BinOp(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_mul(&mut self) -> Expr {
        let mut lhs = self.parse_unary();
        loop {
            let op = match self.peek() {
                Tok::Op(s) if s == "*" => BinKind::Mul,
                Tok::Op(s) if s == "/" => BinKind::Div,
                Tok::Op(s) if s == "%" => BinKind::Mod,
                _ => break,
            };
            if !self.bump_node() {
                break;
            }
            self.eat();
            let rhs = self.parse_unary();
            lhs = Expr::BinOp(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_unary(&mut self) -> Expr {
        match self.peek() {
            // A long `----…x` / `!!!!…x` run recurses parse_unary once per prefix
            // op (unguarded by PARSE_DEPTH_CAP, which only counts parse_expr). Bump
            // BEFORE recursing so the descent — and the resulting tree — stays
            // bounded rather than overflowing the stack.
            Tok::Op(s) if s == "-" => {
                if !self.bump_node() {
                    return Expr::Num(0.0);
                }
                self.eat();
                Expr::Neg(Box::new(self.parse_unary()))
            }
            Tok::Op(s) if s == "!" => {
                if !self.bump_node() {
                    return Expr::Num(0.0);
                }
                self.eat();
                Expr::Not(Box::new(self.parse_unary()))
            }
            _ => self.parse_pow(),
        }
    }

    fn parse_pow(&mut self) -> Expr {
        let base = self.parse_atom();
        if matches!(self.peek(), Tok::Op(s) if s == "^") {
            // `2^2^2^…` right-recurses parse_pow via parse_unary (also unguarded by
            // PARSE_DEPTH_CAP). Bump before recursing into the exponent.
            if !self.bump_node() {
                return base;
            }
            self.eat();
            let exp = self.parse_unary(); // right-associative
            Expr::BinOp(BinKind::Pow, Box::new(base), Box::new(exp))
        } else {
            base
        }
    }

    fn parse_atom(&mut self) -> Expr {
        match self.peek().clone() {
            Tok::Num(v) => {
                self.eat();
                Expr::Num(v)
            }
            Tok::LParen => {
                self.eat();
                let e = self.parse_expr();
                if matches!(self.peek(), Tok::RParen) {
                    self.eat();
                }
                e
            }
            Tok::Ident(name) => {
                self.eat();
                if matches!(self.peek(), Tok::LParen) {
                    self.eat(); // consume '('
                    let mut args = Vec::new();
                    while !matches!(self.peek(), Tok::RParen | Tok::Eof) {
                        // Once the node budget is blown, parse_expr returns without
                        // consuming a token — break so this arg loop cannot spin.
                        if self.over_budget() {
                            break;
                        }
                        args.push(self.parse_expr());
                        if matches!(self.peek(), Tok::Comma) {
                            self.eat();
                        }
                    }
                    if matches!(self.peek(), Tok::RParen) {
                        self.eat();
                    }
                    Expr::Call(name, args)
                } else {
                    Expr::Var(name)
                }
            }
            _ => {
                self.eat(); // skip unexpected token
                Expr::Num(0.0)
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str) -> Env {
        let prog = EelProgram::parse(src);
        let mut env = Env::new();
        prog.run(&mut env);
        env
    }

    fn run_st(src: &str, st: &mut EelState) -> Env {
        let prog = EelProgram::parse(src);
        let mut env = Env::new();
        prog.run_with(&mut env, st);
        env
    }

    fn run_reference_ast(program: &EelProgram, env: &mut Env, state: &mut EelState) {
        let mut budget = EvalBudget::default();
        state.args.clear();
        for stmt in &program.stmts {
            eval(stmt, env, state, &mut budget);
        }
    }

    #[test]
    fn compiled_vm_matches_reference_ast_semantics() {
        let cases = [
            "x=2; y=x*3+sin(.25); z=if(y>1, exec2(x+=4, x/y), 99);",
            "a=0; b=0; loop(7, a+=1, b+=a); while(exec2(b-=3, above(b, 20)));",
            "megabuf(4)=9; megabuf(4)+=2; x=megabuf(4); y=0?17:42;",
            "x=sqrt(-9)+mod(7.8,3.2); y=band(1,bnot(0)); z=clamp(x,0,3);",
            "x=0/0; y=!x; z=if(x,1,2);",
        ];
        for source in cases {
            let program = EelProgram::parse(source);
            let mut compiled_env = Env::new();
            let mut reference_env = Env::new();
            let mut compiled_state = EelState::new();
            let mut reference_state = EelState::new();
            program.run_with(&mut compiled_env, &mut compiled_state);
            run_reference_ast(&program, &mut reference_env, &mut reference_state);
            for name in ["x", "y", "z", "a", "b"] {
                let got = compiled_env.get(name).copied().unwrap_or(0.0);
                let expected = reference_env.get(name).copied().unwrap_or(0.0);
                assert!(
                    (got == expected) || (got.is_nan() && expected.is_nan()),
                    "compiled/reference mismatch for {name} in {source}: {got:?} != {expected:?}"
                );
            }
        }
    }

    #[test]
    fn compiled_vm_reuses_operand_stack() {
        let pure = EelProgram::parse("x=sin(sample)+value1; y=cos(sample)+value2;");
        assert!(pure.operation_count() > 0);

        let mut env = Env::new();
        let mut state = EelState::new();
        pure.run_with(&mut env, &mut state);
        let capacity = state.value_capacity();
        for i in 0..4096 {
            env.insert("sample", i as f64 / 4095.0);
            env.insert("value1", 0.25);
            env.insert("value2", -0.25);
            pure.run_with(&mut env, &mut state);
            assert_eq!(state.value_capacity(), capacity);
        }
    }

    #[test]
    fn compiled_program_reports_only_referenced_symbols() {
        let program = EelProgram::parse("x=q7+reg03; t2=x; y=sample;");
        for name in ["x", "y", "q7", "reg03", "t2", "sample"] {
            assert!(program.references_symbol(name), "missing {name}");
        }
        for name in ["q1", "q8", "reg00", "reg04", "t1", "value1"] {
            assert!(!program.references_symbol(name), "unexpected {name}");
        }
    }

    #[test]
    fn custom_wave_parallel_and_lod_safety_have_distinct_state_rules() {
        let pure = EelProgram::parse("x=sin(sample)+value1; y=cos(sample)+value2;");
        assert!(pure.custom_wave_parallel_safe());
        assert!(pure.custom_wave_lod_safe());

        // These programs are stateful across authored points, but all of that
        // state belongs to one wave pool, so distinct waves may still run in
        // parallel while adaptive point LOD stays disabled.
        for source in [
            "carry+=1; x=carry;",
            "x=megabuf(sample);",
            "megabuf(sample)=x;",
            "loop(2,x+=1);",
            "while(x-=1);",
        ] {
            let program = EelProgram::parse(source);
            assert!(program.custom_wave_parallel_safe(), "{source}");
            assert!(!program.custom_wave_lod_safe(), "{source}");
        }

        // Preset-wide state requires deterministic serial wave scheduling and
        // also prevents skipping authored point evaluations.
        for source in ["x=rand(10);", "x=gmegabuf(sample);", "gmegabuf(sample)=x;"] {
            let program = EelProgram::parse(source);
            assert!(!program.custom_wave_parallel_safe(), "{source}");
            assert!(!program.custom_wave_lod_safe(), "{source}");
        }
    }

    #[test]
    fn effect_summary_reports_explicit_flags_for_representative_programs() {
        // Pure output-only: nothing carries, everything is decimation-safe.
        let pure =
            EelProgram::parse("x=sin(sample)+value1; y=cos(sample)+value2;").effect_summary();
        assert!(!pure.uses_rng);
        assert!(!pure.has_loops);
        assert!(!pure.uses_private_megabuf());
        assert!(!pure.uses_global_gmegabuf());
        assert!(!pure.uses_gmegabuf());
        assert!(!pure.point_carry());
        assert!(pure.custom_wave_parallel_safe());
        assert!(pure.custom_wave_lod_safe());
        assert!(pure.shape_instance_carry_safe());
        assert!(pure.output_writes.any());
        assert!(pure.output_writes.writes("x"));
        assert!(pure.output_writes.writes("y"));
        assert!(!pure.output_writes.writes("r"));
        assert_eq!(pure.output_writes.bits(), OutputWrites::X | OutputWrites::Y);

        // Stateful RNG: the shared stream advances, so it is neither parallel nor
        // decimation safe, yet it carries no register/buffer state itself.
        let rng = EelProgram::parse("x=rand(10);").effect_summary();
        assert!(rng.uses_rng);
        assert!(
            !rng.point_carry(),
            "rand carry is reported via uses_rng only"
        );
        assert!(!rng.custom_wave_parallel_safe());
        assert!(!rng.custom_wave_lod_safe());
        assert!(!rng.shape_instance_carry_safe());

        // Private megabuf: pool-local, so waves may still run in parallel, but the
        // buffer carries across invocations (no LOD, no shape decimation).
        let private = EelProgram::parse("megabuf(1)+=x; x=megabuf(1);").effect_summary();
        assert!(private.uses_private_megabuf());
        assert!(!private.uses_global_gmegabuf());
        assert!(!private.uses_gmegabuf());
        assert!(private.point_carry());
        assert!(private.custom_wave_parallel_safe());
        assert!(!private.custom_wave_lod_safe());
        assert!(!private.shape_instance_carry_safe());

        // Global gmegabuf read and write: preset-wide state, serial scheduling.
        for source in ["x=gmegabuf(3);", "gmegabuf(3)=x;"] {
            let global = EelProgram::parse(source).effect_summary();
            assert!(global.uses_global_gmegabuf(), "{source}");
            assert!(global.uses_gmegabuf(), "{source}");
            assert!(global.point_carry(), "{source}");
            assert!(!global.custom_wave_parallel_safe(), "{source}");
            assert!(!global.custom_wave_lod_safe(), "{source}");
            assert!(!global.shape_instance_carry_safe(), "{source}");
        }

        // Looping: iteration is not decimation-safe, but a loop over reset
        // outputs alone still parallelizes and carries no register/buffer state.
        let looping = EelProgram::parse("loop(4,x=x+1);").effect_summary();
        assert!(looping.has_loops);
        assert!(!looping.point_carry());
        assert!(looping.custom_wave_parallel_safe());
        assert!(!looping.custom_wave_lod_safe());
        assert!(!looping.shape_instance_carry_safe());
    }

    #[test]
    fn shape_instance_carry_distinguishes_reset_globals_from_custom_state() {
        // Regs, t*, q*, and every shape output are reseeded between instances, so
        // a shape that only mutates those is decimation-safe ACROSS INSTANCES even
        // though the same code is NOT custom-wave-LOD-safe (a wave never resets
        // regs/q between points).
        let reset_only =
            EelProgram::parse("rad=rad*1.1; q1=q1+0.1; reg50=bass; ang=ang+t8;").effect_summary();
        assert!(!reset_only.writes_non_shape_output);
        assert!(reset_only.writes_non_wave_output);
        assert!(reset_only.shape_instance_carry_safe());
        assert!(!reset_only.custom_wave_lod_safe());
        assert!(reset_only.custom_wave_parallel_safe());

        // A custom (non-reset) variable persists in the shape env across
        // instances, so decimating would drop carried state.
        let custom_carry = EelProgram::parse("myvar=myvar+1; x=myvar;").effect_summary();
        assert!(custom_carry.writes_non_shape_output);
        assert!(!custom_carry.shape_instance_carry_safe());

        // The EelProgram wrapper delegates to the summary.
        assert!(EelProgram::parse("rad=rad*1.1; q1=q1+0.1;").shape_instance_carry_safe());
        assert!(!EelProgram::parse("myvar=1; x=myvar;").shape_instance_carry_safe());

        // Buffers and RNG carry across shape instances regardless of which vars
        // they touch.
        for source in ["rad=megabuf(1);", "rad=gmegabuf(1);", "r=rand(1); g=r;"] {
            let program = EelProgram::parse(source).effect_summary();
            assert!(
                !program.shape_instance_carry_safe(),
                "{source} must stay stateful"
            );
        }

        // Reset-var name classifier boundaries.
        for name in [
            "x", "rad", "border_a", "reg00", "reg99", "q1", "q32", "t1", "t8",
        ] {
            assert!(is_shape_reset_var(name), "{name} should be a reset var");
        }
        for name in [
            "reg100", "reg0", "q0", "q33", "t0", "t9", "myvar", "time", "bass",
        ] {
            assert!(
                !is_shape_reset_var(name),
                "{name} should not be a reset var"
            );
        }
    }

    /// Focused CPU benchmark for the custom-wave workload. Ignored in normal
    /// tests; run with `cargo test --release eel_vm_custom_wave_benchmark --
    /// --ignored --nocapture` when changing the evaluator.
    #[test]
    #[ignore]
    fn eel_vm_custom_wave_benchmark() {
        let mut source = String::new();
        for i in 0..120 {
            source.push_str(&format!(
                "t{i}=sin(sample*{})+cos(value1*{})+value2;",
                i + 1,
                i + 2
            ));
        }
        source.push_str("x=.5+value1+t119*.001; y=.5+value2-t118*.001;");
        let program = EelProgram::parse(&source);
        let iterations = 512 * 100;

        let mut compiled_env = Env::new();
        let mut compiled_state = EelState::new();
        let compiled_started = std::time::Instant::now();
        for i in 0..iterations {
            compiled_env.insert("sample", (i % 512) as f64 / 511.0);
            compiled_env.insert("value1", 0.25);
            compiled_env.insert("value2", -0.125);
            program.run_with(&mut compiled_env, &mut compiled_state);
        }
        let compiled = compiled_started.elapsed();

        let mut reference_env = Env::new();
        let mut reference_state = EelState::new();
        let reference_started = std::time::Instant::now();
        for i in 0..iterations {
            reference_env.insert("sample", (i % 512) as f64 / 511.0);
            reference_env.insert("value1", 0.25);
            reference_env.insert("value2", -0.125);
            run_reference_ast(&program, &mut reference_env, &mut reference_state);
        }
        let reference = reference_started.elapsed();
        assert_eq!(compiled_env["x"], reference_env["x"]);
        assert_eq!(compiled_env["y"], reference_env["y"]);
        eprintln!(
            "custom-wave EEL: compiled={compiled:?} reference={reference:?} speedup={:.2}x",
            reference.as_secs_f64() / compiled.as_secs_f64()
        );
    }

    #[test]
    fn basic_assign() {
        let env = run("x = 3.0 + 4.0;");
        assert_eq!(env["x"], 7.0);
    }

    #[test]
    fn chained() {
        let env = run("x = 2; y = x * 3;");
        assert_eq!(env["y"], 6.0);
    }

    #[test]
    fn power() {
        let env = run("x = 2^10;");
        assert_eq!(env["x"], 1024.0);
    }

    #[test]
    fn if_fn() {
        let e1 = run("x = if(1, 42, 0);");
        let e2 = run("x = if(0, 42, 99);");
        assert_eq!(e1["x"], 42.0);
        assert_eq!(e2["x"], 99.0);
    }

    #[test]
    fn above_below() {
        let env = run("a = above(5, 3); b = below(5, 3);");
        assert_eq!(env["a"], 1.0);
        assert_eq!(env["b"], 0.0);
    }

    #[test]
    fn comments() {
        let env = run("x = 1; // this is a comment\ny = 2;");
        assert_eq!(env["x"], 1.0);
        assert_eq!(env["y"], 2.0);
    }

    #[test]
    fn neg_unary() {
        let env = run("x = -3; y = -x;");
        assert_eq!(env["x"], -3.0);
        assert_eq!(env["y"], 3.0);
    }

    #[test]
    fn spring_step() {
        // Simulate one step of jelly spring
        let src = "
            spring = 18; resist = 5; dt = 0.0003; grav = 1;
            x1 = 0.5; x2 = 0.4; x3 = 0.6; x4 = 0.5;
            y1 = 0.5; y2 = 0.4; y3 = 0.6; y4 = 0.5;
            vx2 = 0; vy2 = 0;
            vx2 = vx2*(1-resist*dt) + dt*((x1+x3-2*x2)*spring);
            vy2 = vy2*(1-resist*dt) + dt*((y1+y3-2*y2)*spring-grav);
            x2 = x2 + vx2;
            y2 = y2 + vy2;
        ";
        let env = run(src);
        assert!(env["vx2"] > 0.0, "vx2={}", env["vx2"]);
        assert!(env["vy2"].is_finite());
    }

    // ── New-feature tests ────────────────────────────────────────────────────

    #[test]
    fn compound_assign() {
        let env = run("x = 5; x += 3;");
        assert_eq!(env["x"], 8.0);
        let env = run("x = 5; x -= 2;");
        assert_eq!(env["x"], 3.0);
        let env = run("x = 5; x *= 4;");
        assert_eq!(env["x"], 20.0);
        let env = run("x = 20; x /= 4;");
        assert_eq!(env["x"], 5.0);
        let env = run("x = 17; x %= 5;");
        assert_eq!(env["x"], 2.0);
    }

    #[test]
    fn compound_assign_accumulate() {
        // The 82%-of-corpus pattern: an accumulator updated each statement.
        let env = run("t = 0; t += 1; t += 1; t += 0.5;");
        assert_eq!(env["t"], 2.5);
    }

    #[test]
    fn compound_assign_does_not_drop() {
        // Prior bug: `x += 1` was parsed as `x` then `+= 1` dropped → x stayed 0.
        let env = run("x = 10; x += 1;");
        assert_eq!(env["x"], 11.0, "compound-assign was dropped");
    }

    #[test]
    fn if_laziness_skips_untaken_branch() {
        // Only the taken branch should execute its side effects.
        let env = run(
            "taken = 0; skipped = 0; r = if(1, exec2(taken = 1, 100), exec2(skipped = 1, 200));",
        );
        assert_eq!(env["r"], 100.0);
        assert_eq!(env["taken"], 1.0);
        assert_eq!(env["skipped"], 0.0, "untaken branch ran its side effect");

        let env = run(
            "taken = 0; skipped = 0; r = if(0, exec2(taken = 1, 100), exec2(skipped = 1, 200));",
        );
        assert_eq!(env["r"], 200.0);
        assert_eq!(env["taken"], 0.0, "untaken branch ran its side effect");
        assert_eq!(env["skipped"], 1.0);
    }

    #[test]
    fn megabuf_round_trip() {
        let mut st = EelState::new();
        let env = run_st("megabuf(5) = 42; x = megabuf(5);", &mut st);
        assert_eq!(env["x"], 42.0);
        // Out-of-range reads → 0.
        let env = run_st("x = megabuf(-1); y = megabuf(2000000);", &mut st);
        assert_eq!(env["x"], 0.0);
        assert_eq!(env["y"], 0.0);
    }

    #[test]
    fn megabuf_pages_are_lazy_zero_filled_and_cross_boundaries() {
        let mut buf = MegaBuf::default();
        assert_eq!(buf.allocated_pages(), 0);
        assert_eq!(buf.read(0.0), 0.0);
        assert_eq!(buf.read((MEGABUF_MAX - 1) as f64), 0.0);
        assert_eq!(buf.allocated_pages(), 0, "reads must not allocate pages");

        let boundary = MEGABUF_PAGE_LEN as f64;
        buf.write(boundary - 1.0, 11.0);
        buf.write(boundary, 22.0);
        buf.write((MEGABUF_MAX - 1) as f64, 33.0);
        assert_eq!(buf.allocated_pages(), 3);
        assert_eq!(buf.read(boundary - 1.0), 11.0);
        assert_eq!(buf.read(boundary), 22.0);
        assert_eq!(buf.read((MEGABUF_MAX - 1) as f64), 33.0);
        assert_eq!(buf.read(boundary + 1.0), 0.0);

        // MilkDrop floors fractional addresses and ignores out-of-range writes.
        buf.write(boundary + 1.9, 44.0);
        assert_eq!(buf.read(boundary + 1.0), 44.0);
        buf.write(-1.0, 55.0);
        buf.write(MEGABUF_MAX as f64, 66.0);
        assert_eq!(buf.allocated_pages(), 3);
        assert_eq!(buf.read(-1.0), 0.0);
        assert_eq!(buf.read(MEGABUF_MAX as f64), 0.0);
    }

    #[test]
    fn megabuf_persists_across_runs() {
        let mut st = EelState::new();
        run_st("megabuf(3) = 7;", &mut st);
        let env = run_st("x = megabuf(3);", &mut st);
        assert_eq!(env["x"], 7.0);
    }

    #[test]
    fn megabuf_compound_assign() {
        let mut st = EelState::new();
        let env = run_st("megabuf(2) = 10; megabuf(2) += 5; x = megabuf(2);", &mut st);
        assert_eq!(env["x"], 15.0);
    }

    #[test]
    fn gmegabuf_shared() {
        let g = Arc::new(Mutex::new(MegaBuf::default()));
        let mut st1 = EelState::with_gmegabuf(g.clone());
        let mut st2 = EelState::with_gmegabuf(g.clone());
        run_st("gmegabuf(1) = 99;", &mut st1);
        let env = run_st("x = gmegabuf(1);", &mut st2);
        assert_eq!(env["x"], 99.0, "gmegabuf not shared across pools");
    }

    #[test]
    fn compiled_program_classifies_and_reuses_one_gmegabuf_guard() {
        let local = EelProgram::parse("megabuf(1)=7; x=megabuf(1);");
        let global = EelProgram::parse("gmegabuf(1)=0; loop(4096, gmegabuf(1)+=1); x=gmegabuf(1);");
        assert!(!local.compiled.effects.uses_gmegabuf());
        assert!(global.compiled.effects.uses_gmegabuf());

        let mut env = Env::new();
        let mut state = EelState::new();
        global.run_with(&mut env, &mut state);
        assert_eq!(env["x"], 4096.0);

        // Custom-wave point batches may hold the same lock across many program
        // executions. The prelocked path must preserve state and results without
        // attempting to recursively acquire the mutex.
        let point = EelProgram::parse("gmegabuf(7)+=1; x=gmegabuf(7);");
        assert!(point.uses_gmegabuf());
        let handle = state.gmegabuf.clone();
        let mut guard = handle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for _ in 0..512 {
            point.run_with_prelocked_gmegabuf(&mut env, &mut state, &mut guard);
        }
        drop(guard);
        assert_eq!(env["x"], 512.0);
    }

    #[test]
    fn eel_rng_is_deterministic_isolated_and_shareable() {
        let program = EelProgram::parse("x=rand(1000); y=randint(17);");

        // Standalone states no longer depend on a process-global stream.
        let mut env_a = Env::new();
        let mut env_b = Env::new();
        let mut state_a = EelState::new();
        let mut state_b = EelState::new();
        program.run_with(&mut env_a, &mut state_a);
        program.run_with(&mut env_b, &mut state_b);
        assert_eq!(env_a["x"], env_b["x"]);
        assert_eq!(env_a["y"], env_b["y"]);

        // Two equation pools sharing a preset RNG consume one ordered stream.
        let seed = 0xdecaf_bad5eed;
        let reference_rng = EelRng::shared(seed);
        let shared_rng = EelRng::shared(seed);
        let reference_g = Arc::new(Mutex::new(MegaBuf::default()));
        let shared_g = Arc::new(Mutex::new(MegaBuf::default()));
        let mut reference = EelState::with_shared(reference_g, reference_rng);
        let mut pool_a = EelState::with_shared(shared_g.clone(), shared_rng.clone());
        let mut pool_b = EelState::with_shared(shared_g, shared_rng.clone());

        let mut reference_env = Env::new();
        program.run_with(&mut reference_env, &mut reference);
        let first = (reference_env["x"], reference_env["y"]);
        program.run_with(&mut reference_env, &mut reference);
        let second = (reference_env["x"], reference_env["y"]);

        let mut pool_a_env = Env::new();
        let mut pool_b_env = Env::new();
        program.run_with(&mut pool_a_env, &mut pool_a);
        program.run_with(&mut pool_b_env, &mut pool_b);
        assert_eq!((pool_a_env["x"], pool_a_env["y"]), first);
        assert_eq!((pool_b_env["x"], pool_b_env["y"]), second);

        // Resetting through the shared handle resets every pool for replay.
        shared_rng.reseed(seed);
        program.run_with(&mut pool_b_env, &mut pool_b);
        assert_eq!((pool_b_env["x"], pool_b_env["y"]), first);
        assert!(pool_b_env["y"] >= 0.0 && pool_b_env["y"] < 17.0);
    }

    #[test]
    fn cached_slot_sets_copy_without_name_lookups() {
        let names = ["reg00", "reg01", "reg99"];
        let mut source = Env::new();
        let mut destination = Env::new();
        let source_slots: Vec<_> = names.iter().map(|name| source.intern_slot(name)).collect();
        let destination_slots: Vec<_> = names
            .iter()
            .map(|name| destination.intern_slot(name))
            .collect();
        for (slot, value) in source_slots.iter().zip([1.0, -2.0, 99.0]) {
            source.set_slot_value(*slot, value);
        }

        destination.copy_slot_values_from(&destination_slots, &source, &source_slots);
        assert_eq!(destination.slot_value(destination_slots[0]), 1.0);
        assert_eq!(destination.slot_value(destination_slots[1]), -2.0);
        assert_eq!(destination.slot_value(destination_slots[2]), 99.0);
    }

    #[test]
    fn while_loop_counts() {
        // Decrement i until 0; n counts iterations. while repeats while |last|>1e-5.
        let env = run("i = 5; n = 0; while(exec2(n = n + 1, i = i - 1));");
        assert_eq!(env["i"], 0.0);
        assert_eq!(env["n"], 5.0);
    }

    #[test]
    fn loop_runs_n_times() {
        let env = run("n = 0; loop(4, n = n + 1);");
        assert_eq!(env["n"], 4.0);
        // loop(3.9,..) runs 4 times (i<n with float n).
        let env = run("n = 0; loop(3.9, n = n + 1);");
        assert_eq!(env["n"], 4.0);
    }

    #[test]
    fn loop_multi_statement_body() {
        let env = run("a = 0; b = 0; loop(3, a = a + 1, b = b + 2);");
        assert_eq!(env["a"], 3.0);
        assert_eq!(env["b"], 6.0);
    }

    #[test]
    fn loops_are_capped_by_cumulative_render_budget() {
        let env = run("n = 0; loop(1048576, n = n + 1);");
        assert_eq!(env["n"], LOOP_ITERATION_BUDGET as f64);

        let env = run("n = 0; loop(1048576, loop(1048576, n = n + 1));");
        assert_eq!(
            env["n"],
            (LOOP_ITERATION_BUDGET - 1) as f64,
            "outer loop consumes one iteration budget entry before inner body runs"
        );
    }

    #[test]
    fn deeply_nested_parse_and_eval_are_bounded() {
        let mut src = String::from("x = ");
        for _ in 0..(PARSE_DEPTH_CAP + 64) {
            src.push_str("if(1,");
        }
        src.push('7');
        for _ in 0..(PARSE_DEPTH_CAP + 64) {
            src.push_str(",0)");
        }
        src.push(';');
        let env = run(&src);
        assert!(env.get("x").copied().unwrap_or(0.0).is_finite());
    }

    #[test]
    fn exec2_exec3_return_last() {
        let env = run("x = exec2(10, 20);");
        assert_eq!(env["x"], 20.0);
        let env = run("x = exec3(1, 2, 3);");
        assert_eq!(env["x"], 3.0);
    }

    #[test]
    fn sigmoid_fn() {
        // sigmoid(0,y) = 1/(1+e^0) = 0.5
        let env = run("x = sigmoid(0, 1);");
        assert!((env["x"] - 0.5).abs() < 1e-9, "sigmoid(0,1)={}", env["x"]);
        // Large positive x*y → near 1.
        let env = run("x = sigmoid(10, 1);");
        assert!(env["x"] > 0.999);
    }

    #[test]
    fn integer_mod() {
        // ns-eel2 mod is integer: floor(x) % floor(y).
        let env = run("x = 7.8 % 3.2;");
        assert_eq!(env["x"], 1.0); // 7 % 3 = 1
    }

    #[test]
    fn sqrt_of_negative() {
        // ns-eel2 sqrt(abs(x)).
        let env = run("x = sqrt(-9);");
        assert_eq!(env["x"], 3.0);
    }

    #[test]
    fn bitops() {
        let env = run("a = bitor(5, 2); b = bitand(6, 3);");
        assert_eq!(env["a"], 7.0); // 101 | 010 = 111
        assert_eq!(env["b"], 2.0); // 110 & 011 = 010
    }

    // ── Ternary + div/mod (Butterchurn JS-transpiled equation support) ──────────

    #[test]
    fn ternary_basic() {
        let env = run("x = 1 ? 42 : 99; y = 0 ? 42 : 99;");
        assert_eq!(env["x"], 42.0);
        assert_eq!(env["y"], 99.0);
    }

    #[test]
    fn ternary_with_condition_expr() {
        // The exact shape from sherwin_maxawow's pixel_eqs.
        let env = run("d = 5; r = 3; x1 = .00001 < abs(above(d, r)) ? 0 : 7;");
        // above(5,3)=1, abs(1)=1, .00001<1 → true → 0.
        assert_eq!(env["x1"], 0.0);
        let env = run("d = 2; r = 3; x1 = .00001 < abs(above(d, r)) ? 0 : 7;");
        // above(2,3)=0, abs(0)=0, .00001<0 → false → 7.
        assert_eq!(env["x1"], 7.0);
    }

    #[test]
    fn ternary_nested_right_assoc() {
        // a ? b : c ? d : e  parses as  a ? b : (c ? d : e)
        let env = run("x = 0 ? 1 : 1 ? 2 : 3;");
        assert_eq!(env["x"], 2.0);
        let env = run("x = 0 ? 1 : 0 ? 2 : 3;");
        assert_eq!(env["x"], 3.0);
        let env = run("x = 1 ? 5 : 0 ? 2 : 3;");
        assert_eq!(env["x"], 5.0);
    }

    #[test]
    fn ternary_is_lazy() {
        // Only the taken branch should run its side effects.
        let env =
            run("taken = 0; skipped = 0; r = 1 ? exec2(taken = 1, 100) : exec2(skipped = 1, 200);");
        assert_eq!(env["r"], 100.0);
        assert_eq!(env["taken"], 1.0);
        assert_eq!(
            env["skipped"], 0.0,
            "untaken ternary branch ran its side effect"
        );
    }

    #[test]
    fn ternary_looser_than_comparison() {
        // `1 < 2 ? 10 : 20` must parse as `(1 < 2) ? 10 : 20`, not `1 < (2 ? 10 : 20)`.
        let env = run("x = 1 < 2 ? 10 : 20;");
        assert_eq!(env["x"], 10.0);
    }

    #[test]
    fn div_fn() {
        let env = run("x = div(10, 4);");
        assert_eq!(env["x"], 2.5);
        // div by zero → 0 (ns-eel2 / BinKind::Div semantics).
        let env = run("x = div(10, 0);");
        assert_eq!(env["x"], 0.0);
    }

    #[test]
    fn mod_fn() {
        // mod(x,y) is integer mod: floor(x) % floor(y).
        let env = run("x = mod(7.8, 3.2);");
        assert_eq!(env["x"], 1.0); // 7 % 3 = 1
                                   // mod by zero → 0.
        let env = run("x = mod(7, 0);");
        assert_eq!(env["x"], 0.0);
    }

    // ── P2-VIS-001: iterative, budgeted lexer skipping ───────────────────────

    #[test]
    fn long_unknown_char_run_is_skipped_in_bounded_work() {
        // The old `_ => self.next_tok()` tail-recursed once per skipped unknown
        // character; a long run overflowed the host stack. Skipping is now
        // iterative — a 200k unknown-char run must lex without recursion and yield
        // the trailing real statement.
        let mut src = String::with_capacity(200_020);
        for _ in 0..200_000 {
            src.push('@');
        }
        src.push_str("x = 5;");
        let env = run(&src);
        assert_eq!(env["x"], 5.0);
    }

    #[test]
    fn many_consecutive_comments_are_skipped_in_bounded_work() {
        let mut src = String::with_capacity(1_000_020);
        for _ in 0..200_000 {
            src.push_str("// c\n");
        }
        src.push_str("y = 7;");
        let env = run(&src);
        assert_eq!(env["y"], 7.0);
    }

    // ── P2-VIS-002: bounded flat expression node count ───────────────────────

    #[test]
    fn flat_operator_chain_past_budget_is_rejected() {
        // A flat `1+1+1+…` chain parses iteratively but builds a left-deep tree of
        // depth == operator count; walked/dropped recursively it overflows the
        // stack. Past the node budget it must be rejected with the typed error.
        let mut src = String::from("x = 1");
        for _ in 0..(MAX_PARSE_NODES + 16) {
            src.push_str("+1");
        }
        src.push(';');
        assert!(matches!(
            EelProgram::try_parse(&src),
            Err(ParseError::ExpressionTooLarge { limit }) if limit == MAX_PARSE_NODES
        ));
        // The infallible constructor yields an inert program rather than
        // overflowing when the over-budget tree is dropped / evaluated.
        let prog = EelProgram::parse(&src);
        let mut env = Env::new();
        prog.run(&mut env);
    }

    #[test]
    fn unary_chain_past_budget_is_rejected() {
        // A long `-----…x` run recurses parse_unary (unguarded by PARSE_DEPTH_CAP);
        // past the budget it must be rejected rather than overflow the parse stack.
        let mut src = String::from("x = ");
        for _ in 0..(MAX_PARSE_NODES + 16) {
            src.push('-');
        }
        src.push_str("1;");
        assert!(matches!(
            EelProgram::try_parse(&src),
            Err(ParseError::ExpressionTooLarge { limit }) if limit == MAX_PARSE_NODES
        ));
    }

    // Token/node-count boundary: the existing budget tests probe
    // MAX_PARSE_NODES + 16, which an off-by-one in `bump_node` would still
    // reject. Pin the edge itself: the flip from accepted to typed
    // rejection must be sharp, and the accepted side must still EVALUATE
    // correctly rather than merely parse.
    #[test]
    fn flat_chain_node_budget_edge_is_sharp() {
        // `x = 1+1+…+1` with `terms` addends.
        fn chain(terms: u32) -> String {
            let mut src = String::from("x = 1");
            for _ in 0..terms {
                src.push_str("+1");
            }
            src.push(';');
            src
        }

        // The predicate is monotone in `terms`, so binary-search the edge
        // instead of parsing every length.
        let (mut lo, mut hi) = (0u32, MAX_PARSE_NODES + 64);
        assert!(EelProgram::try_parse(&chain(lo)).is_ok());
        assert!(EelProgram::try_parse(&chain(hi)).is_err());
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if EelProgram::try_parse(&chain(mid)).is_ok() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let last_ok = lo;
        let first_rejected = hi;
        assert_eq!(
            first_rejected,
            last_ok + 1,
            "the edge must be a single step"
        );

        // Where the edge sits, exactly: one counted node per addend, so a
        // chain of MAX_PARSE_NODES addends is the largest accepted and the
        // next one breaches. This ties the observable limit to the
        // constant — a change to either side moves this assertion.
        assert_eq!(
            last_ok, MAX_PARSE_NODES,
            "each addend costs exactly one budgeted node"
        );

        // Accepted side: parses AND evaluates to the arithmetic answer.
        let prog = EelProgram::try_parse(&chain(last_ok)).expect("edge-1 must parse");
        let mut env = Env::new();
        prog.run(&mut env);
        assert_eq!(env["x"], f64::from(last_ok) + 1.0);

        // Rejected side: the typed error naming the cap, not a panic and
        // not a silently truncated program.
        assert!(matches!(
            EelProgram::try_parse(&chain(first_rejected)),
            Err(ParseError::ExpressionTooLarge { limit }) if limit == MAX_PARSE_NODES
        ));
    }

    // Parse-depth boundary: the parse-depth guard has a DIFFERENT
    // policy from the node budget — `enter_parse` refuses to descend and
    // the statement degrades, with no typed error. Pin that distinction
    // so a future change cannot quietly convert one into the other.
    #[test]
    fn parse_depth_cap_degrades_without_typed_error_or_overflow() {
        fn nested(levels: u32) -> String {
            let mut src = String::from("x = ");
            for _ in 0..levels {
                src.push_str("if(1,");
            }
            src.push('7');
            for _ in 0..levels {
                src.push_str(",0)");
            }
            src.push(';');
            src
        }

        // Well inside the cap: the nest evaluates to its innermost value.
        let shallow = EelProgram::try_parse(&nested(8)).expect("shallow nesting must parse");
        let mut env = Env::new();
        shallow.run(&mut env);
        assert_eq!(env["x"], 7.0);

        // At and past the cap the guard stops the descent. Node cost here
        // is ~2 per level, so at these depths the NODE budget cannot be
        // what fires — this isolates the depth guard.
        for levels in [PARSE_DEPTH_CAP, PARSE_DEPTH_CAP + 1, PARSE_DEPTH_CAP * 2] {
            assert!(
                levels * 4 < MAX_PARSE_NODES,
                "depth probe must stay under the node budget"
            );
            let parsed = EelProgram::try_parse(&nested(levels));
            let Ok(parsed) = parsed else {
                panic!(
                    "over-deep nesting must degrade rather than raise a typed error \
                     (that is the node budget's job); {levels} levels was rejected"
                );
            };
            // Bounded work, no stack overflow, and whatever survives is a
            // finite value — never NaN/inf leaking from a partial parse.
            let mut env = Env::new();
            parsed.run(&mut env);
            assert!(
                env.get("x").copied().unwrap_or(0.0).is_finite(),
                "degraded parse must still yield a finite value at {levels} levels"
            );
        }
    }

    #[test]
    fn modest_operator_chain_still_parses() {
        // A legitimately sized chain (well under the budget) is unaffected.
        let mut src = String::from("x = 1");
        for _ in 0..100 {
            src.push_str("+1");
        }
        src.push(';');
        assert!(EelProgram::try_parse(&src).is_ok());
        let env = run(&src);
        assert_eq!(env["x"], 101.0);
    }

    // ── P2-VIS-012: interned dense env, no per-vertex allocation ─────────────

    #[test]
    fn per_vertex_eval_reuses_buffers_without_allocation() {
        // A per_pixel-style program: assignments + function calls (arg stack) +
        // audio/coordinate reads. Simulate a mesh's worth of vertices reusing one
        // Env + EelState. Results must stay numerically exact (equivalence) and the
        // value buffer / interned name set / call-arg stack must NOT grow after
        // warm-up (proving no per-vertex allocation).
        let prog = EelProgram::parse("zoom = 1 + sin(x) * bass; warp = abs(y) + zoom;");
        let mut env = Env::new();
        let mut state = EelState::new();

        let run_vertex = |env: &mut Env, state: &mut EelState, x: f64, y: f64, bass: f64| {
            env.clear();
            env.insert("bass", bass);
            env.insert("x", x);
            env.insert("y", y);
            prog.run_with(env, state);
            (env["zoom"], env["warp"])
        };

        // Warm up: first runs intern names and grow buffers to steady state.
        run_vertex(&mut env, &mut state, 0.1, 0.2, 0.5);
        run_vertex(&mut env, &mut state, 0.3, 0.4, 0.6);
        let val_cap = env.value_capacity();
        let names = env.interned_len();
        let arg_cap = state.arg_capacity();

        for k in 0..4096u32 {
            let x = k as f64 * 0.001;
            let y = k as f64 * 0.002;
            let bass = (k % 7) as f64 * 0.1;
            let (zoom, warp) = run_vertex(&mut env, &mut state, x, y, bass);

            // Equivalence with the reference formula, exactly.
            let exp_zoom = 1.0 + x.sin() * bass;
            let exp_warp = y.abs() + exp_zoom;
            assert!(
                (zoom - exp_zoom).abs() < 1e-12,
                "zoom mismatch at vertex {k}"
            );
            assert!(
                (warp - exp_warp).abs() < 1e-12,
                "warp mismatch at vertex {k}"
            );

            // No per-vertex allocation.
            assert_eq!(
                env.value_capacity(),
                val_cap,
                "env value buffer reallocated"
            );
            assert_eq!(
                env.interned_len(),
                names,
                "a new name was interned per vertex"
            );
            assert_eq!(state.arg_capacity(), arg_cap, "call-arg stack reallocated");
        }
    }

    #[test]
    fn selective_slot_snapshot_restores_controls_and_preserves_temporaries() {
        let mut env = Env::new();
        let warp = env.intern_slot("warp");
        let zoom = env.intern_slot("zoom");
        let temporary = env.intern_slot("temporary");
        let x = env.intern_slot("x");
        env.set_slot_value(warp, 0.25);
        env.set_slot_value(zoom, 1.1);
        env.set_slot_value(temporary, 3.0);
        env.set_slot_value(x, 0.1);

        let mut snapshot = EnvSnapshot::default();
        env.capture_slots_into(&[warp, zoom], &mut snapshot);

        // Simulate one vertex changing both authored controls and user state.
        env.set_slot_value(warp, 9.0);
        env.set_slot_value(zoom, 8.0);
        env.set_slot_value(temporary, 7.0);
        env.set_slot_value(x, 0.9);
        env.restore_slots(&snapshot);

        assert_eq!(env.slot_value(warp), 0.25);
        assert_eq!(env.slot_value(zoom), 1.1);
        assert_eq!(env.slot_value(temporary), 7.0);
        assert_eq!(env.slot_value(x), 0.9);
    }

    #[test]
    fn cleared_env_reports_absent_vars_as_zero() {
        // Generation-based clear must make a var set on a previous "vertex" read as
        // absent (→ 0.0 in eval) unless re-seeded, matching HashMap::clear + reseed.
        let prog = EelProgram::parse("out = leftover + 1;");
        let mut env = Env::new();
        let mut state = EelState::new();
        env.insert("leftover", 41.0);
        prog.run_with(&mut env, &mut state);
        assert_eq!(env["out"], 42.0);
        // Next vertex: clear, do NOT re-seed `leftover` → it reads as 0.
        env.clear();
        prog.run_with(&mut env, &mut state);
        assert_eq!(env["out"], 1.0);
        assert!(env.get("leftover").is_none());
    }
}
