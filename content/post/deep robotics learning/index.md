---
title: Deep Robotics Learning
description: A note on deep robotics learning
slug: drl
date: 2023-03-07 00:00:00+0000
image: cover.jpg
categories:
- Lecture Notes Expansion
tags:
- Reinforcement Learning
- Deep Learning
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

Although the concepts of learning is not new, the field robotics have much longer history in the classical methods 
such. However, without learning, robotics is a much more complicated and specific tasks and requires multiple 
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
reinforce the robots to behave. Reinforcement learning perfectly fits that description. The main idea of RL is exactly for 
the robot learns a policy of which it acts on to maximize the reward.
