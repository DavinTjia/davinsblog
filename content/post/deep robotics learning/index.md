---
title: Deep Robotics Learning (1) - Introduction and Policy Gradient
description: A note on deep robotics learning (UW CSE 599 G1)
slug: drl
date: 2023-03-07 00:00:00+0000
image: cover.jpg
categories:
- Lecture Notes Expansion
tags:
- Reinforcement Learning
- Deep Learning
- CSE599 G1 Note
---

In this blog post, I would be going over some topics on deep reinforcement learning with focus on robotics learning. I would 
expect this to be spanned across a series of posts. The content of these posts is my takeaway and additional exploration 
on the topics taught in [CSE599 G1](https://courses.cs.washington.edu/courses/cse599g/23wi/) at University of Washington
by [Abishek Gupta](https://homes.cs.washington.edu/~abhgupta/). Abishek is such a great teacher and this is such a great 
class that I think it deserves documentation of some form. That is the motivations of this blog posts series. 

**Disclaimer**: if there is any error in this blog posts, it is entirely my mistakes and my misunderstanding, but not 
the results of the slides of the lectures or Abhishek Gupta's delivery. Please contact me if you have any concerns or 
questions about the content of this post. Thank you!

## Introduction

To start, I would like to discuss some motivation on why reinforcement learning and why deep learning. 

Although the concepts of learning is not new, the field robotics have much longer history in the classical methods. However, without learning, robotics is a much more complicated and specific tasks and requires multiple 
layers of engineering across different discipline, For example, robots that have been built and coded to navigate one 
environment might fail entirely at another (although similar) environment. This is because hard coded functionality, 
perception, and planning is much less generalizable in exchange for precision and reliability. However, such domain-specific
design has many draw backs. It is much less economically justifiable (than, says, deploy a real human) and less likely
to have a wide-spread deployment. This is where the motivation of learning comes in. Instead of building a finite state
machine that transitions based on tasks, it is much better to train a model that would adapt based on the diversity in 
the data. Realistically speaking, the most likely solutions would fall in somewhere in between: we would like to 
modularize robotic engineering where each module perhaps consists of some learned models that are good at a tasks but 
are more generalization than state machine.

Hopefully, this would convince you a bit on maybe we should explore the possibilities of using learning in robotics. 
Assuming that you are buying the concepts of learning, you might want to ask why reinforcement learning (RL)? Supervised
learning is much more extensively studied and seems to work phenomenonly well based on recent progress on CV and NLP.
For supervised learning to perform well, some typical requires include 1. the data much be collected iid from the 
distribution 2. a lot of data, like a lot of data, as much as possible 3. well labeled data. These requirements 
do not fit well with the nature of robotic tasks. For 1., robotics tasks is usually sequential, that means it is not iid. 
For 2, it is exponentially more difficult to collect data. Different labs use different robots, under different 
environments, and with different calibration. There is lack of a general real-world data set that most researchers 
generally agree on like ImageNet in CV in early 2010s. For 3., it is hard to label the data. For example, a state 
might be good in one sequence of action but might be horrible in another sequence of action. The label of a particular 
data points need to take in multiple factors for considerations such as current goal, current state and available actions, 
and current trajectory. This make supervised learning generally an unsuitable way for robot to learn novel tasks. 

Reinforcement learning on the other hand is a much more natural way to learn a sequence of action for a goal. In 
psychology there are two main category of conditioning: classical and operant. Classical conditioning is very well
known thanks to Pavlov's dog. Operant conditioning, in simple term, refers to using rewards or punishments to incentive
good behaviors. B.F. Skinner, one of the main figure in behavior psychology, believes that all behavior is learned and 
novel reinforcement shapes novel behaviors. Now, if only there is a way that we can use rewards or punishments to 
reinforce the robots to behave. Reinforcement learning perfectly fits that description. (I would assume
most readers have basic knowledge in RL from now on).

If I successfully (kinda) convince you that learning, specifically reinforcement learning, might be 
part of the solution to robotics, the next step is to get you believe that we should also combine
deep learning with RL. One most obvious example is [AlphaGo](https://www.nature.com/articles/nature16961).
The main ideas of AlphaGo is combining deep Q-learning with Monte Carlo Tree Search (MCTS). A 
non-neural way to do Q-learning is to uses a dictionary (or some other cleverer data structure) to 
store the values of all the state-action pairs. However, this quickly become intractable when the 
possible combinations exceed the number of atoms in the universe like the game of Go. It would be 
much better if we have a parameterized function that could takes in state-action pairs and spits out
their (estimated) values. Similarly, hand-crafting the agents and models are difficult and expensive.
Using a neural network to represent the models would be much simpler to do (sometimes it is as 
easy as importing an out-of-the-box ResNet from the library!). 

Now at the end of introduction, I would also like to raise some counterarguments of why we shouldn't
always be using deep reinforcement learning. Typical problems with deep learning such as low 
interpretability, data inefficiency, and sensitivity to hyperparameters still apply. In addition, 
deep reinforcement learning (or RL in general) are not suitable for tasks that requires high precisions.
The values from the output is often just an estimation (albeit a good one) and many other factors 
including sampling strategy can all affect the final trajectories. In industry such as manufacturing
where a robot only needs to master one tasks well and efficiently or medicine where safety and interpretability is critical
are not (yet) good areas where DRL should be prioritized. 

## Policy Gradient

To use deep learning on reinforcement learning, identifying objective function for optimization is the first crucial step. Recall that in RL we have a policy $\pi$ that takes a state $s$ as input and output an action $a$. The hope is that the trajectories $\tau$'s (a sequence of states and actions) generated according $\pi$ would optimize the return $r$. This gives us
$$
\max\_{\theta} \mathbb{E}\_{\tau \sim \pi\_{\theta}} [\sum^T\_{t = 0 } r (s\_t, a\_t)]
$$
There are many techniques that can be used to optimize over this objective functions. Different techniques requires different modifications to the objectives and results in different algorithms. Here, we would be focusing on using gradient *asscent*, which in the setting of RL is often known as **policy gradient**. Equivalently, the continuous version of the sum above is given by 
$$
    J(\theta) = \int p\_{\theta } (\tau ) R(\tau )d\tau 
$$
This induce the REINFORCE algorithm where we estimate the policy gradient with likelihood ratio
$$\begin{aligned}
       \nabla \_{\theta } J(\theta) &= \nabla\_{\theta} \int p\_{\theta } (\tau ) R(\tau )d\tau \\\\\\ 
       &=  \int \nabla\_{\theta} p\_{\theta } (\tau ) R(\tau )d\tau \\\\\\
       &=   \int \frac{p \_{\theta}(\tau)}{p\_\theta (\tau) } \nabla\_{\theta} p\_{\theta } (\tau ) R(\tau )d\tau \\\\\\
       &=   \int \frac{p \_{\theta}(\tau)}{ p\_\theta (\tau) } \nabla\_{\theta} p\_{\theta } (\tau ) R(\tau )d\tau \\\\\\
       &=  \int p\_\theta (\tau) \nabla\_{\theta} \log p\_{\theta } (\tau ) R(\tau )d\tau \quad \text{REINFORCE trick} \\\\\\
       &= \mathbb{E } \_{p\_{\theta }(\tau )} [ \nabla\_{\theta} \log p\_{\theta }(\tau )R(\tau)]
\end{aligned}$$
Now we unroll the definition of $p\_{\theta }(\tau )$
$$
p\_{\theta }(\tau ) = p(s\_0) \prod^{T- 1 }\_{t= 0 }p(s\_{t+ 1} | s\_t, a\_t) \pi(a\_t |s\_t)
$$
where $p$ is the environment dynamics (an abuse of notation) and $\pi$ is the policy. Using property of log, we have 
$$\begin{aligned}
\log p\_{\theta }(\tau ) &= \log p(s\_0) +  \sum^{T- 1 }\_{t= 0 } (\log p(s\_{t+ 1} | s\_t, a\_t) + \log \pi(a\_t |s\_t)) \\\
 \nabla\_{\theta} \log p\_{\theta }(\tau ) &=  \nabla\_{\theta} \log p(s\_0) +  \sum^{T- 1 }\_{t= 0 } ( \nabla\_{\theta} \log p(s\_{t+ 1} | s\_t, a\_t) +  \nabla\_{\theta} \log \pi(a\_t |s\_t)) 
\end{aligned}$$
Now, we assume a *model free* learning, that is, only the policy is parameterized. This means the dynamic $p$ is independent of $\theta$, which leaves us  
$$\begin{aligned}
 \nabla\_{\theta} \log p\_{\theta }(\tau ) &=  \nabla\_{\theta} \log p(s\_0) +  \sum^{T- 1 }\_{t= 0 } ( \nabla\_{\theta} \log p(s\_{t+ 1} | s\_t, a\_t) +  \nabla\_{\theta} \log \pi(a\_t |s\_t))  \\\\\\
 &= 0 +  \sum^{T- 1 }\_{t= 0 } (0 +  \nabla\_{\theta} \log \pi(a\_t |s\_t)) \\\\\\ 
 &= \sum^{T-1 }\_{t =0}   \nabla\_{\theta} \log \pi(a\_t |s\_t)
\end{aligned}$$
We can then rewrite our objectives as 
$$
 \mathbb{E } \_{p\_{\theta }(\tau )} [ \nabla\_{\theta} \log p\_{\theta }(\tau )R(\tau)] =  \mathbb{E } \_{p\_{\theta }(\tau )} [  \sum^{T-1 }\_{t =0}   \nabla\_{\theta} \log \pi(a\_t |s\_t) \sum^{T}\_{t=0}r(s\_t, a\_t)]
$$
where the reward $R$ collected along the trajectory $\tau$ is given by $r(s\_t, a\_t)$ at each time stamp from $t = 0$ to the horizon $t=T$. Although the later expectation seems intractable to compute,  we can approximate it with sampling !
$$
  \mathbb{E } \_{p\_{\theta }(\tau )} [  \sum^{T-1 }\_{t =0}   \nabla\_{\theta} \log \pi(a\_t |s\_t) \sum^{T}\_{t=0}r(s\_t, a\_t)] \approx \frac{1 }{N }\sum^N\_{i = 0 }\sum^T\_{t = 0 }  \nabla\_{\theta} \log \pi(a\_t^i |s\_t^i) \sum^{T}\_{t'=0}r(s\_{t'}^i, a\_{t'}^i)
$$
This is essentially gives us the REINFORCE algorithm: you sample trajectories $\tau^i$ from current policy $\pi\_{\theta} (a\_t | s\_t)$ and then estimate the gradient $ \nabla\_{\theta}J(\theta)$ and then we update the current parameter $\theta \leftarrow \theta + \nabla\_{\theta}J(\theta)$ with gradient ascent:
```
def REINFORCE:
  sample tau from pi_theta 
  gradient <- estimated by sampling
  theta <- current theta + lr * gradient
```
In plain word, we want to select good data and then increase the likelihood of selecting good data. While policy gradient is unbiased, it is nontheless a *high variance* estimator. Because at one time we use a single sample (one trajectory) estimate but we really want an averaged return (averages across many trajectories) estimate. 

### Variance Reduction for PG
The most simple way to address the problem of high variance is taken "causaility" in mind. Notice in the sampling equation, 
$$
 \frac{1 }{N }\sum^N\_{i = 0 }\sum^T\_{t = 0 }  \nabla\_{\theta} \log \pi(a\_t^i |s\_t^i) \sum^{T}\_{t'=0}r(s\_{t'}^i, a\_{t'}^i)
$$
the return is summing accross all $t' \in [0, T]$ at each time stamp. This means the trajectory depends on the past and the future, but at a given moment $t'$ what has been done has been done and we only care about the return we get in the future. Therefore, we can consider **return to go** by ignoring past term and update our sampling equation to 
$$
 \frac{1 }{N }\sum^N\_{i = 0 }\sum^T\_{t = 0 }  \nabla\_{\theta} \log \pi(a\_t^i |s\_t^i) \sum^{T}\_{t'=t}r(s\_{t'}^i, a\_{t'}^i)
$$
where we excluding past term and now $t' \in [t, T]$. This method, however, doesn't solve the problem of "arbitrary centering". When some part of the trajectories returns negative rewards and other part return positive rewards, it is clear where to center the distribution around actions where the reward is positive. In the scenario where all the rewards are positive (which is more common), every actions in the distribution would be pushed up. To select good actions, we need to be very precise about pushing up the good ones more than the other, which is difficult and has high variance. Now, what if instead of pushing up all actions, we push down the actions that are bad and pushed up the actions that are good even though they all receive positive rewards? Can we reduce the variance further. The idea is to introduce a current state dependent function, what we called **baseline** $b(s\_t)$, and subtracting it from the return sum in policy gradient. 
$$
 \frac{1 }{N }\sum^N\_{i = 0 }\sum^T\_{t = 0 }  \nabla\_{\theta} \log \pi(a\_t^i |s\_t^i) \sum^{T}\_{t'=t}[r(s\_{t'}^i, a\_{t'}^i) - b(s\_t)]
$$
Baseline allows us to center the return (at current state) to reduce the variance. But do we sacrifice lower variance with higher bias by introducing the baseline term? Actually, no. 
$$
\begin{aligned}
  \int p\left(\tau \right) \nabla\_\theta \log \pi\_\theta\left(\tau \right)\left[\sum\_{t^{\prime}=t}^T r\left(\tau\right)-b\left(s\_t\right)\right] d\tau = &\int\_{\mathcal{S}} \int\_{\mathcal{A}} p\left(s\_t, a\_t\right) \nabla\_\theta \log \pi\_\theta\left(a\_t \mid s\_t\right)\left[\sum\_{t^{\prime}=t}^T r\left(s\_{t^{\prime}}, a\_{t^{\prime}}\right)-b\left(s\_t\right)\right] d s\_t d a\_t \\\ 
  = &\int\_{\mathcal{S}} \int\_{\mathcal{A}} p\left(s\_t, a\_t\right) \nabla\_\theta \log \pi\_\theta\left(a\_t \mid s\_t\right)\left[\sum\_{t^{\prime}=t}^T r\left(s\_{t^{\prime}}, a\_{t^{\prime}}\right)\right] d s\_t d a\_t - \\\
  &\int\_{\mathcal{S}} \int\_{\mathcal{A}} p\left(s\_t, a\_t\right) \nabla\_\theta \log \pi\_\theta\left(a\_t \mid s\_t\right) b\left(s\_t\right) d s\_t d a\_t
\end{aligned}
$$ 
This means that if we can show the integral with baseline as the integrand is 0, it would mean this is still an unbiased estimator.
$$
\begin{aligned}
\iint p\left(s\_t, a\_t\right) \nabla\_\theta \log \pi\_\theta\left(a\_t \mid s\_t\right)\left[b\left(s\_t\right)\right] d s\_t d a\_t &=\iint p\left(s\_t\right) \pi\_\theta\left(a\_t \mid s\_t\right) \nabla\_\theta \log \pi\_\theta\left(a\_t \mid s\_t\right)\left[b\left(s\_t\right)\right] d s\_t d a\_t \\\
& =\int p\left(s\_t\right) b\left(s\_t\right) \int \pi\_\theta\left(a\_t \mid s\_t\right) \nabla\_\theta \log \pi\_\theta\left(a\_t \mid s\_t\right) d a\_t d s\_t \\\
& =\int p\left(s\_t\right) b\left(s\_t\right) \int \nabla\_\theta \pi\_\theta\left(a\_t \mid s\_t\right) d a\_t d s\_t \\\ 
&=\int p\left(s\_t\right) b\left(s\_t\right) \nabla\_\theta \int \pi\_\theta\left(a\_t \mid s\_t\right) d a\_t d s\_t \\\
&=\int p\left(s\_t\right) b\left(s\_t\right) \nabla\_\theta(1) d s\_t \\\
&=0
\end{aligned}
$$
Indeed, this is a rare day in machine learning where adding the baseline term reduces the variance without trading off more bias.