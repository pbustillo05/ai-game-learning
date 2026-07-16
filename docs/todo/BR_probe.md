NOT YET ADDRESSED - TO KEEP

We need to implement a modification of the training algorithm to train a BR probe to measure exploitability. In all likeliness, we need:
- to use Rao-Blackwell sequential
- at the start of each trajectory, we randomly pick an "exploiter seat" uniformly: this seat plays the weights we're training, the other seats all play the frozen weights we want to exploit
- we can initialize the exploiter weights with the frozen ones
- we do normal pre-terminal rollouts EXCEPT that:
    * we stop whenever the exploiter folds
    * we only care about the exploiter's equity and posterior: so technically the MC api only needs to return these numbers
    * after running a frozen model forward pass during rollout, we only need to store the new range of that player, everything else can be discarded (in particular, no need to remember the \pi(\cdot \mid x_p, \sigma) distributions)
    * backward estimation of A^ and V^ only needs to be done on the exploiter
    * minibatching and gradient descent is also only done on the expploiter obviously
- Technically, it's true that there always exists a deterministic BR probe (no need to have a mixing strategy). We might still want mixing output head to facilitate exploration. Maybe a compromise is to use less raise modes (maybe just 1 raising mode is enough).
- Practically speaking, it will be difficult to "fully train" the BR probe to convergence. A good proxy for exploitability might be "how fast is the expected leak growing?"
- We probably need to rollout P times more trajectories in the start to reach the same number of nodes to use in minibatches.