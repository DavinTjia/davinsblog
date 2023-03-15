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
E_{\tau \sim \pi_{\theta}}
$$
no