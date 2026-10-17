## PETS Reacher Evaluation Code

This directory contains code adapted from the [publicly released PETS implementation](https://github.com/kchua/handful-of-trials) associated with Chua et al., "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models," NeurIPS 2018.

The original PETS code was released under the MIT License; the original license notice is preserved in `LICENSE`.

This adaptation is used for the Reacher evaluation in our paper. PETS is used as an evaluation framework, and the only experimental modification is to restrict the input coordinates supplied to the learned dynamics model according to the discovered graphs.