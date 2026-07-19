1. Fix/strengthen BR implementation: see "docs/todo/BR_probe.md" and "docs/todo/branching_BR_probe.md"
2. Finish implementing/testing branching and dynamic $\beta$. See the stale "docs/todo/control_branching_size.md" 
3. Launch a bigger wave of runs to see how good our ideas are
4. Add a measure of "number of hands seen in rb training" with entropy-based estimation of number of hands in a posterior.
5. Calibrate the MC sample number: how much can we increase it without hurting training time? Maybe 10k would be better? Or even 100k? We should get an idea of how fast it converges, and how much it reduces variance of the gradient.
6. Experiment with IS in minibatches: can we select more high-variance spots? This could help learn later streets better. See "docs/todo/importance_minibatching.md"
7. Tune parameters for a first Quest run.