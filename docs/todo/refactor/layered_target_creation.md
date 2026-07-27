# Layered target creation

Status: CUDA implementation proposal.

## 1. Scope

This file specifies the sole production path:

```text
Leaf completion
  -> initial work frontier
  -> one ordered CUDA kernel per topology layer
  -> SigmaTarget + ActionTarget
  -> atomic publication
```

The mathematical reference is `BranchedValue` in `docs/theory/core_math.tex`.
Sequential execution is the $N=N_{\rm tgt}=\widetilde N=1$ specialization.
Replica topologies are independent; layer $\ell$ concatenates work from every replica that has layer $\ell$.

No Python loop launches work per node or action. No layer requires a host read or device synchronization.
CUDA graph capture is not part of the contract.

## 2. Mode constants

One validated mode row supplies:

```text
training_regime   self_play | best_response
rollout_mode      sequential | branched.fixed | branched.KKT
private_estimator sampled | rao_blackwell

P_train           P for self_play; 1 for best_response
has_mass_moment   private_estimator == rao_blackwell
has_trace_cost    rollout_mode != sequential
N_max             1 for sequential; ceil(beta_max) for branched
```

One shared kernel template is specialized from the mode schema. There is no runtime target algorithm selector.

`owner_slot` is the physical seat in self-play, `0` at exploiter nodes in best response, and `-1` for structural rows.

## 3. Layer plan

CPU topology planning assigns each retained structural node a reverse layer:

$$
L(s)=1+\max_{u\in\operatorname{child}(s)}L(u),
\qquad L(\mathtt{Leaf})=0.
$$

Shorter child paths are carried until their parent layer. For layer $\ell$:

```text
LayerPlan
  layer_ptr <layer_count + 1>
  carry_ptr <layer_count + 1>

  node <node_count>
    replica
    child_begin
    child_count
    destination_work_row
    input_type
    output_type
    owner_slot                    # work-channel index; -1 for structural
    N
    N_tgt
    N_tilde
    sigma_head_row                # -1 for structural
    sigma_target_row              # -1 for structural
    action_begin                  # -1 for structural

  child <edge_count>
    source_work_row
    source_output_type
    path_length
    cost_multiplier

  carry <carry_count>
    replica
    source_work_row
    destination_work_row
    output_type
```

Invariants:

```text
child_count == N
1 <= N_tgt <= N <= N_max
target children are child[child_begin : child_begin + N_tgt]
every node's child span is in logical-child order
every action target span uses the same logical-child order
sigma_head_row == -1 iff owner_slot == -1
all source rows are live in input arena parity
all destination rows are distinct in output arena parity
every child source and node destination has the node's replica
every carry source and destination has the carry's replica
leaf child cost_multiplier == 0
```

`cost_multiplier` is the nominal multiplier of the retained child endpoint. For a compressed unary path, the edge
cost contribution is:

$$
\mathtt{path\_length}
+\mathtt{cost\_multiplier}\,C_{\rm child}.
$$

The planner must prove this summary is sufficient for every omitted node in the selected mode.

## 4. Work arenas

Two fixed-capacity SoA arenas alternate by layer. Logical domains are `all`, `pre`, and `post`.

### 4.1 Sampled estimator

```text
SampledWork[domain]
  return_value <P_train, size[domain]>
  trace_vector <P_train>                   # branched only
  cost                                     # branched only
  delta                                    # branched only
```

`return_value` is the lambda-return carrier. `trace_vector` is the unbootstrapped terminal-value vector used by the
trace-variance recurrence. They are distinct fields.

### 4.2 Rao--Blackwell estimator

```text
RBWork[domain]
  mass <P_train, size[domain]>
  moment <P_train, size[domain]>
  trace_vector <P_train>                   # branched only
  cost                                     # branched only
  delta                                    # branched only
```

`moment = mass * conditional_value`. `trace_vector` is never overwritten by value bootstrapping.

Sequential schemas omit `trace_vector`, `cost`, and `delta`.

## 5. Completion epilogue

Completion writes Leaf-order rows. One bounded epilogue:

```text
sampled:
  return_value <- selected payoff components
  trace_vector <- selected payoff components             # branched only

rao_blackwell:
  mass <- p_hat
  moment <- p_hat * v_hat
  trace_vector <- completed vector value                 # branched only

branched:
  cost <- 0
  delta <- 0
```

For a `pre` row whose evaluator result is `post`-wide, aggregate mass and moment before conditionalization:

$$
w_k=\sum_{h:c(h)=k}w_h,
\qquad
m_k=\sum_{h:c(h)=k}m_h.
$$

The initial movement plan permutes completed rows into layer-zero work addresses. Completion order is otherwise
unchanged.

## 6. CUDA launch

For each layer $\ell$:

```text
src = arena[ell & 1]
dst = arena[(ell + 1) & 1]

propagate_layer<Mode><<<node_count[ell] + carry_count[ell], block_threads, shared_bytes, stream>>>(
    mode_constants,
    layer_plan,
    src,
    dst,
    SigmaHead,
    SigmaTarget,
    ActionTarget,
    DeviceFailure)
```

Counts are summed across replicas; `replica` selects lane-local arenas, heads, targets, and failures. All layer
launches are enqueued in order from C++/CUDA orchestration without an intervening host synchronization.

CTA assignment:

```text
blockIdx.x < node_count[ell]       one CTA owns one parent
otherwise                         one CTA owns one carry row
```

All mode and row-kind branches are uniform within a CTA. `block_threads` is one hardware-selected constant for the
production kernel.

## 7. Child conversion

For each child field, `Convert(child_output_type -> parent_output_type)` is:

```text
same domain       identity
post -> pre       class sum
all -> all        identity
```

For RB, convert `mass` and `moment` separately, then conditionalize.
For sampled work, convert `return_value`.
`trace_vector`, `cost`, and `delta` are domain-independent.

Unsupported conversion edges fail schema construction.

## 8. Parent reduction

Let $\mathcal E_s^{\rm tgt}$ be the first `N_tgt` logical children.
All sums use logical-child order.

### 8.1 Common scalars

```text
trace_mean = Avg_tgt(child.trace_vector)
delta_bar  = Avg_tgt(child.delta)                   # branched only
cost_raw   = Avg_tgt(path_length
                     + cost_multiplier * child.cost) # branched only
```

For a structural row:

```text
dst.trace_vector = trace_mean
dst.cost          = cost_raw                        # branched only
dst.delta         = delta_bar / N_tgt               # branched only
```

### 8.2 Sampled payload

Maintain one carrier $z_s^{(q)}$ for every trained player $q$:

$$
z_s^{(q)}=\operatorname{Avg}_{\rm tgt}(z_{\rm child}^{(q)}).
$$

For a trained node whose work-channel slot is $p$ and action child $a$:

$$
Q_a=z_a^{(p)},
$$

$$
B_a=
\begin{cases}
\lambda\operatorname{Avg}_{b\ne a}(z_b^{(p)})+(1-\lambda)V_{\rm old},
  &N_{\rm tgt}>1,\\
V_{\rm old},&N_{\rm tgt}=1,
\end{cases}
$$

$$
A_a=Q_a-B_a.
$$

Write:

```text
SigmaTarget.v_lambda_hat <- z_s[owner_slot]
ActionTarget.a_lambda_hat[a] <- A_a
```

Propagate:

$$
z_s^{(p)}\leftarrow\lambda z_s^{(p)}+(1-\lambda)V_{\rm old}
$$

Non-owner carriers are unchanged.

### 8.3 Rao--Blackwell payload

Maintain mass and moment for every trained player $q$ and output component:

$$
w_s^{(q)}=\operatorname{Avg}_{\rm tgt}(w_{\rm child}^{(q)}),
\qquad
m_s^{(q)}=\operatorname{Avg}_{\rm tgt}(m_{\rm child}^{(q)}).
$$

For a trained node whose work-channel slot is $p$ and action child $a$:

$$
Q_a=\operatorname{Cond}(m_a^{(p)},w_a^{(p)}),
$$

$$
B_a=
\begin{cases}
\lambda\operatorname{Cond}
\left(\sum_{b\ne a}m_b^{(p)},\sum_{b\ne a}w_b^{(p)}\right)
+(1-\lambda)V_{\rm old},
  &N_{\rm tgt}>1,\\
V_{\rm old},&N_{\rm tgt}=1,
\end{cases}
$$

$$
A_a=Q_a-B_a.
$$

Write:

```text
SigmaTarget.v_lambda_hat <- Cond(m_s[owner_slot], w_s[owner_slot])
ActionTarget.p_hat[a] <- w_a[owner_slot]
ActionTarget.a_lambda_hat[a] <- A_a
```

Propagate for the trained owner:

$$
m_s^{(p)}\leftarrow\lambda m_s^{(p)}+(1-\lambda)w_s^{(p)}V_{\rm old}.
$$

Other players' moments are unchanged.

## 9. Branched trace and cost targets

For a trained row with `N_tgt >= 2`:

$$
T_{\rm exec}
=\frac{1}{N_{\rm tgt}-1}
\sum_{a\in\mathcal E_s^{\rm tgt}}
\lVert\mathtt{trace\_vector}_a-\mathtt{trace\_mean}\rVert_2^2,
$$

$$
\widehat T=\max(0,T_{\rm exec}-\overline\Delta),
\qquad T^*=\widehat T.
$$

For `N_tgt == 1`:

$$
\widehat T=\bot,
\qquad T^*=T_{\rm old}.
$$

Then:

$$
\Delta_s
=\frac{T^*+\overline\Delta}{N_{\rm tgt}}
-\frac{T^*}{\widetilde N}.
$$

Write:

```text
SigmaTarget.t_hat <- T_hat                  # validity: N_tgt >= 2
SigmaTarget.c_lambda_hat <- cost_raw
```

Propagate:

$$
C_s\leftarrow\lambda C_{\rm raw}+(1-\lambda)C_{\rm old}.
$$

Frozen best-response opponents are structural rows: they write no heads or targets and apply no value, trace, or
cost-head bootstrap. Branched frozen rows with multiple children remain structural reductions; unary frozen rows
may be compressed into `path_length`.

## 10. CTA algorithm

For one parent CTA:

```text
1. load the node descriptor;
2. reduce trace_vector, cost, and delta in lane order;
3. for component tiles:
     a. load and convert every target child in logical order;
     b. accumulate child totals;
     c. write parent work;
     d. if trained, write sigma and per-child action targets;
     e. if trained, apply owner bootstrap to propagated work;
4. compute trace target and propagated delta;
5. publish no per-row flag.
```

For leave-one-out values, compute `total - child` in fp32. A validation experiment must compare this with direct
sibling summation for concentrated posterior rows. If accuracy is insufficient, the direct sibling sum replaces it
as the sole production implementation.

Carry CTAs copy exactly the active logical fields for their output domain.

## 11. Failure and publication

Kernels validate:

```text
finite work and targets
mass >= 0
moment == 0 when mass == 0, within tolerance
valid row/span bounds
valid N, N_tgt, N_tilde
supported conversion
```

The minimum failing logical row per replica writes its `DeviceFailure`. Later layers skip that replica without
writes. A replica's targets become learner-visible only after its final layer and validation/publication kernel
succeed; failure discards only its partial targets.

## 12. Memory and traffic counters

Record:

```text
layer_count
node_count per layer
carry_count and bytes per layer
child edges read
mass/moment bytes read and written
target bytes written
kernel launches
CUDA execution time
```

If carry bytes are material, replace physical carries with a last-consumer slot plan. That replacement must preserve
the schemas and recurrences above and delete the carry implementation.

## 13. Acceptance

- Exact agreement with a direct `BranchedValue` reference for every supported mode.
- Logical-child-order deterministic reductions.
- Correct `N_tgt` protected-root prefix.
- Correct $N_{\rm tgt}=1$ fallback to frozen heads.
- Correct `post -> pre` mass/moment aggregation before conditionalization.
- No frozen-opponent head or target access.
- No Python node/action loop.
- No host synchronization between layers.
- One production target path per mode.
- CUDA coverage for empty strata, unary chains, duplicate labels, concentrated/zero posterior mass, maximum degree,
  maximum depth, and deliberately unbalanced trees.
- At least current sequential target throughput and a substantial reduction from the audited branched propagation
  time.

## 14. Deferred experiments

- Carry arenas versus last-consumer slots.
- Direct leave-one-out sum versus totals-minus-child.
- CTA thread count and component tile width.
- Optional graph capture after the launch sequence is measured.

An experiment that wins replaces the production choice and deletes the losing implementation.
