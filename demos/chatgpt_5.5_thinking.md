我先给结论：**你的手写方案的核心方向是可行的，但需要把“概率直接相加/平均”改成“log-density / score / energy / vector field 的组合”。** 你提出的两个问题里，第一个关于 score function，答案是：**你现在写的 log 概率拆分形式还不是严格的 score function，但它可以自然导出 score function。** 第二个关于用 KL 判断不同阶段/不同条件下的分布差异，答案是：**可以用，但直接比较边缘概率还不够，最好比较条件互信息、条件 KL、以及 flow/diffusion vector field 的 cross-arm coupling。**

你前两个 Markdown 的核心已经建立了三种双臂建模方案：joint policy、conditional/leader-follower policy、independent marginal policy，并且你已经把它们和双臂耦合强弱、任务难度、tradeoff 联系起来了。第一个文件里你明确把 residual/error 理解为 joint policy 与 local policy 之间缺失的 coupling information，并提出用 energy residual 或 log-density residual 表示更合理；第二个文件里你进一步把 A/B/C 三种方案对应为 joint、conditional、independent，并讨论了用条件互信息、KL、泛化误差、coupling cost 来刻画 tradeoff。这个方向是统一三种方案的正确起点。 

---

## 1. 你的手写方案里最关键的正确点

你手写里想做的是：

[
p(a_L,a_R\mid O_L,O_R,O_g)
]

可以同时写成：

[
p(a_L\mid O_L,O_R,O_g),
p(a_R\mid a_L,O_L,O_R,O_g)
]

也可以写成：

[
p(a_R\mid O_L,O_R,O_g),
p(a_L\mid a_R,O_L,O_R,O_g)
]

这一步是严格成立的，来自概率链式法则：

[
p(a_L,a_R\mid o)=p(a_L\mid o)p(a_R\mid a_L,o)
]

[
p(a_L,a_R\mid o)=p(a_R\mid o)p(a_L\mid a_R,o)
]

其中我把：

[
o=(O_L,O_R,O_g)
]

简写为整体观测。

所以你写的“① × 2 = ② + ③”这个直觉，本质上是在说：

> 同一个 joint distribution 可以从左主导或右主导两个方向分解。
> 如果左右两种分解都成立，那么可以把它们统一起来。

这个方向是对的。

但后面你写：

[
p(a_L,a_R\mid o)
================

\frac12[
p(a_L\mid o)p(a_R\mid a_L,o)
+
p(a_R\mid o)p(a_L\mid a_R,o)
]
]

这里需要小心。

如果这两个乘积都是**真实分布**，那么：

[
p(a_L\mid o)p(a_R\mid a_L,o)
============================

# p(a_R\mid o)p(a_L\mid a_R,o)

p(a_L,a_R\mid o)
]

所以它们的平均仍然等于真实 joint distribution：

[
\frac12[p^\star+p^\star]=p^\star
]

这在数学上没问题。

但在模型学习中，问题变成：

[
q_{LR}(a_L,a_R\mid o)
=====================

q_L(a_L\mid o)q_R(a_R\mid a_L,o)
]

[
q_{RL}(a_L,a_R\mid o)
=====================

q_R(a_R\mid o)q_L(a_L\mid a_R,o)
]

这两个模型一般不会完全相等。因此你不能简单假设：

[
q_{LR}=q_{RL}
]

更合理的统一形式是 mixture：

[
q(a_L,a_R\mid o)
================

\alpha(o)q_{LR}(a_L,a_R\mid o)
+
(1-\alpha(o))q_{RL}(a_L,a_R\mid o)
]

进一步扩展到 A/B/C：

[
q(a_L,a_R\mid o)
================

\pi_A(o)q_A(a_L,a_R\mid o)
+
\pi_{LR}(o)q_{LR}(a_L,a_R\mid o)
+
\pi_{RL}(o)q_{RL}(a_L,a_R\mid o)
+
\pi_C(o)q_C(a_L,a_R\mid o)
]

其中：

[
q_C(a_L,a_R\mid o)
==================

q_L(a_L\mid o_L,o_g)q_R(a_R\mid o_R,o_g)
]

这就是你想要的统一 A/B/C 的数学形式。

---

## 2. 你第一个问题：这是不是 score function？

你手写里写了类似：

[
\log p(a_L,a_R\mid o)
=====================

\log \frac12
+
\log p(a_L\mid o)
+
\log p(a_R\mid a_L,o)
+
\log \frac12
+
\log p(a_R\mid o)
+
\log p(a_L\mid a_R,o)
]

这里有一个数学问题：如果你使用的是**概率混合**：

[
p = \frac12 p_{LR}+\frac12 p_{RL}
]

那么：

[
\log p
======

\log\left(
\frac12 p_{LR}+\frac12 p_{RL}
\right)
]

而不是：

[
\log p
======

\log\frac12+\log p_{LR}
+
\log\frac12+\log p_{RL}
]

也就是说：

[
\log(x+y)\neq \log x+\log y
]

所以你写的这个 log 展开不严格。

但是，如果你想做的是 **product-of-experts / energy product**，那么 log 相加是成立的。也就是定义：

[
p(a_L,a_R\mid o)
\propto
p_L(a_L\mid o)
p_R(a_R\mid o)
\exp(R_\theta(a_L,a_R,o))
]

取 log：

[
\log p(a_L,a_R\mid o)
=====================

\log p_L(a_L\mid o)
+
\log p_R(a_R\mid o)
+
R_\theta(a_L,a_R,o)
-------------------

\log Z(o)
]

这里的：

[
R_\theta(a_L,a_R,o)
]

就是你一直说的 residual / error / 信息差 / coupling term。

这时 score function 是：

[
s_\theta(a_L,a_R,o)
===================

\nabla_{a_L,a_R}\log p_\theta(a_L,a_R\mid o)
]

于是：

[
s_\theta
========

\nabla_{a_L,a_R}\log p_L(a_L\mid o)
+
\nabla_{a_L,a_R}\log p_R(a_R\mid o)
+
\nabla_{a_L,a_R}R_\theta(a_L,a_R,o)
]

由于：

[
p_L(a_L\mid o)
]

只依赖 (a_L)，而：

[
p_R(a_R\mid o)
]

只依赖 (a_R)，所以：

[
s_L
===

\nabla_{a_L}\log p_L(a_L\mid o)
+
\nabla_{a_L}R_\theta(a_L,a_R,o)
]

[
s_R
===

\nabla_{a_R}\log p_R(a_R\mid o)
+
\nabla_{a_R}R_\theta(a_L,a_R,o)
]

这就是严格意义上的 score-based 统一形式。

所以对你第一个问题的回答是：

**你现在写的 log 形式不是严格 score function，但只要改成 product-of-experts / energy residual 形式，就可以严格导出 score function。**

---

## 3. 你关于 (p(a_R\mid a_L,o)) 和 (p(a_L\mid a_R,o)) 不好建模的判断是对的

你手写里说：

> (p(a_R\mid a_L,o)) 和 (p(a_L\mid a_R,o)) 不好建模，所以想找中间量。

这个判断非常关键。

原因是这两个条件分布要求模型显式知道：

[
a_L \rightarrow a_R
]

或者：

[
a_R \rightarrow a_L
]

但很多双臂任务不是固定单向因果，而是阶段性变化的。

例如：

阶段 1：两臂独立靠近物体；

阶段 2：左臂固定物体，右臂操作；

阶段 3：双臂同步搬运；

阶段 4：右臂放置，左臂收回。

所以固定建模：

[
p(a_R\mid a_L,o)
]

或者：

[
p(a_L\mid a_R,o)
]

都会引入 direction bias。

你提出用一个中间关系：

[
p(a_R\mid o)
============

w_0 p(a_R\mid a_L,o)
]

[
p(a_L\mid o)
============

w_\phi p(a_L\mid a_R,o)
]

这个想法可以理解为：你想用一个可学习的映射，把边缘分布和条件分布联系起来。

但这里不能把它写成普通的常数乘法，因为概率分布需要归一化。更严谨的写法是 density ratio：

[
\frac{p(a_R\mid a_L,o)}{p(a_R\mid o)}
=====================================

\exp(r_{L\rightarrow R}(a_L,a_R,o))
]

[
\frac{p(a_L\mid a_R,o)}{p(a_L\mid o)}
=====================================

\exp(r_{R\rightarrow L}(a_L,a_R,o))
]

其中：

[
r_{L\rightarrow R}
==================

\log p(a_R\mid a_L,o)-\log p(a_R\mid o)
]

[
r_{R\rightarrow L}
==================

\log p(a_L\mid a_R,o)-\log p(a_L\mid o)
]

这两个量才是你想要的 (w_0,w_\phi) 的严格版本。

它们表示：

> 看到另一只手的动作以后，当前手臂动作分布发生了多大改变。

如果：

[
r_{L\rightarrow R}\approx 0
]

说明：

[
p(a_R\mid a_L,o)\approx p(a_R\mid o)
]

也就是右臂不依赖左臂。

如果：

[
r_{L\rightarrow R}\neq 0
]

说明左臂动作对右臂有信息增益。

这和你第二个 Markdown 里用条件互信息衡量方案 tradeoff 的思想一致：独立方案 C 的不可避免误差可以由 (I(a_L;a_R\mid o)) 刻画，而 conditional / joint 的价值正来自这个 cross-arm dependency。

---

## 4. 你第二个问题：能否用 KL 证明时序差异？

你手写里提出：

[
P_t(a_R\mid O_L,O_R,O_g)
\quad \text{和} \quad
P_{t-h}(a_R\mid O_L,O_R,O_g)
]

之间可以用 KL 来衡量差距，从而判断在 ([t-h,t]) 内左右手关系是否紧密。

这个想法是可行的，但需要稍微改造。

单纯比较：

[
D_{KL}\left(
p_t(a_R\mid o_t)
\Vert
p_{t-h}(a_R\mid o_{t-h})
\right)
]

衡量的是**右臂自身策略随时间变化的幅度**，它不一定等价于**左右臂耦合强度**。

它可以说明：

> 右臂在这段时间是否需要重新规划。

但它不能直接说明：

> 右臂的变化是不是由左臂导致的。

所以你还需要比较 conditional 和 marginal 的差异：

[
D_{KL}\left(
p_t(a_R\mid a_L,o_t)
\Vert
p_t(a_R\mid o_t)
\right)
]

这个量更直接表示：

> 给定左臂动作以后，右臂动作分布改变了多少。

对其取期望：

[
\mathcal{C}_{L\rightarrow R}(t)
===============================

\mathbb{E}_{p(a_L,a_R\mid o_t)}
\left[
\log
\frac{
p(a_R\mid a_L,o_t)
}{
p(a_R\mid o_t)
}
\right]
]

这就是方向性条件互信息：

[
\mathcal{C}_{L\rightarrow R}(t)
===============================

I(a_L;a_R\mid o_t)
]

如果加入时延：

[
\mathcal{C}_{L\rightarrow R}(h,t)
=================================

I(a_{L,t-h};a_{R,t}\mid o_{t-h:t})
]

那么它可以衡量：

> 左臂在 (t-h) 时刻的动作对右臂 (t) 时刻动作有多大预测价值。

同理：

[
\mathcal{C}_{R\rightarrow L}(h,t)
=================================

I(a_{R,t-h};a_{L,t}\mid o_{t-h:t})
]

于是：

* 如果 (\mathcal{C}_{L\rightarrow R}(h,t)) 大，说明左臂领先右臂；
* 如果 (\mathcal{C}_{R\rightarrow L}(h,t)) 大，说明右臂领先左臂；
* 如果 (h=0) 处最大，说明同步 joint coupling；
* 如果所有值都小，说明左右臂近似独立。

所以你原来的 KL 思路可以升级为：

[
\boxed{
\text{用时延条件互信息 / 条件 KL 衡量左右臂耦合方向、强度和频率}
}
]

这比只比较 (p_t(a_R)) 和 (p_{t-h}(a_R)) 更准确。

---

## 5. 用 flow matching 实现你的统一方案

现在把动作轨迹写成：

[
x=a_L,\qquad y=a_R,\qquad z=(x,y)
]

flow matching 的目标是学习一个 vector field：

[
v^\star_t(z_t,o)
]

使得：

[
z_0\sim p_0(z),\qquad z_1\sim p_{\text{data}}(z\mid o)
]

通常可以用线性插值：

[
z_t=(1-t)z_0+t z_1
]

目标速度：

[
u_t=z_1-z_0
]

训练目标是：

[
\mathcal{L}_{FM}
================

\mathbb{E}*{t,z_0,z_1,o}
\left[
\left|
v*\theta(z_t,t,o)-u_t
\right|^2
\right]
]

为了统一 A/B/C，你可以把 vector field 分解成：

[
v_\theta(z_t,t,o)
=================

\pi_A v_A
+
\pi_{LR} v_{LR}
+
\pi_{RL} v_{RL}
+
\pi_C v_C
]

其中：

[
\pi_A+\pi_{LR}+\pi_{RL}+\pi_C=1
]

四个 expert 分别表示：

### A：joint expert

[
v_A=[v_L^A(x_t,y_t,o,t),v_R^A(x_t,y_t,o,t)]
]

左右臂互相看见，适合强耦合。

### B1：left-to-right expert

[
v_{LR}
======

[
v_L(x_t,o,t),
v_R(y_t,x_t,o,t)
]
]

左臂主导右臂。

### B2：right-to-left expert

[
v_{RL}
======

[
v_L(x_t,y_t,o,t),
v_R(y_t,o,t)
]
]

右臂主导左臂。

### C：independent expert

[
v_C=
[
v_L(x_t,o_L,o_g,t),
v_R(y_t,o_R,o_g,t)
]
]

左右臂独立。

然后训练目标为：

[
\mathcal{L}
===========

\mathcal{L}*{FM}
+
\lambda_A \mathbb{E}[\pi_A]
+
\lambda_B \mathbb{E}[\pi*{LR}+\pi_{RL}]
+
\lambda_f\mathcal{R}*{freq}
+
\lambda_s\mathcal{R}*{sparse}
]

其中设置：

[
\lambda_A>\lambda_B>0
]

意思是 joint 模式最贵，conditional 次之，independent 最便宜。

这样模型只有在任务真的需要时才会打开更复杂的耦合结构。

这正好实现你想要的：

> 用机器学习学习 A/B/C 之间的 tradeoff，而不是人工选择。

---

## 6. 用 diffusion / score-based model 实现

如果用 score-based diffusion，你可以建模联合 score：

[
s_\theta(z_t,t,o)
=================

\nabla_{z_t}\log p_t(z_t\mid o)
]

其中：

[
z_t=(a_{L,t},a_{R,t})
]

你可以分解为：

[
s_\theta
========

s_L^{local}
+
s_R^{local}
+
s_{couple}
]

更具体：

[
s_L
===

\nabla_{a_L}\log p_L(a_L\mid o_L,o_g)
+
\nabla_{a_L}R_\theta(a_L,a_R,o)
]

[
s_R
===

\nabla_{a_R}\log p_R(a_R\mid o_R,o_g)
+
\nabla_{a_R}R_\theta(a_L,a_R,o)
]

其中 (R_\theta) 是 coupling residual。

训练目标可以用 denoising score matching：

[
\mathcal{L}_{DSM}
=================

\mathbb{E}*{t,z_0,\epsilon,o}
\left[
\left|
s*\theta(z_t,t,o)
+
\frac{\epsilon}{\sigma_t}
\right|^2
\right]
]

或者常见 diffusion noise prediction loss：

[
\mathcal{L}_{DDPM}
==================

\mathbb{E}*{t,z_0,\epsilon,o}
\left[
\left|
\epsilon-
\epsilon*\theta(z_t,t,o)
\right|^2
\right]
]

为了统一 A/B/C，可以把 noise predictor 写成：

[
\epsilon_\theta
===============

\pi_A\epsilon_A
+
\pi_{LR}\epsilon_{LR}
+
\pi_{RL}\epsilon_{RL}
+
\pi_C\epsilon_C
]

这和 flow matching 的形式完全对应。

---

## 7. 用 energy-based model 实现

EBM 是最贴近你“error / 信息差”直觉的。

定义：

[
p_\theta(a_L,a_R\mid o)
=======================

\frac{
\exp[-E_\theta(a_L,a_R,o)]
}{
Z_\theta(o)
}
]

能量分解为：

[
E_\theta(a_L,a_R,o)
===================

E_L(a_L,o_L,o_g)
+
E_R(a_R,o_R,o_g)
+
E_{couple}(a_L,a_R,o)
]

其中：

[
E_{couple}
]

就是你手写中想表达的“方案 A 和方案 B/C 之间缺失的关系”。

如果：

[
E_{couple}=0
]

则：

[
p(a_L,a_R\mid o)
================

p_L(a_L\mid o_L,o_g)
p_R(a_R\mid o_R,o_g)
]

也就是方案 C。

如果：

[
E_{couple}
==========

E_{LR}(a_R\mid a_L,o)
]

则接近方案 B 的 left-to-right。

如果：

[
E_{couple}
==========

E_{joint}(a_L,a_R,o)
]

则接近方案 A。

训练可以用 score matching：

[
\mathcal{L}_{SM}
================

\mathbb{E}*{p*{data}}
\left[
\frac12
\left|
\nabla_a E_\theta(a,o)
\right|^2
---------

\Delta_a E_\theta(a,o)
\right]
]

实际工程上更常用 contrastive loss：

[
\mathcal{L}_{EBM}
=================

E_\theta(a^+,o)
+
\log
\sum_{a^-}
\exp[-E_\theta(a^-,o)]
]

或者：

[
\mathcal{L}_{NCE}
=================

*

\log
\frac{
\exp[-E_\theta(a^+,o)]
}{
\exp[-E_\theta(a^+,o)]
+
\sum_i \exp[-E_\theta(a_i^-,o)]
}
]

但 EBM 训练和采样成本会比 flow/diffusion 更高。

所以我建议：**理论表述用 EBM 最清晰，工程实现用 flow matching 或 diffusion 更现实。**

---

## 8. 你手写方案中“用 KL 判断 h 大小”的可行版本

你写到可以用：

[
KL\left(
\frac{
P_t(a_L\mid O_L,O_g,O_R)
}{
P_{t+h}(a_L\mid O_LO_g,O_R)
}
\right)
<\epsilon
]

类似方式确定 (h) 大小。

这个思路可以变成一个 adaptive frequency rule：

对每只手定义重规划必要性：

[
\Delta_L(t,h)
=============

D_{KL}
\left[
p_t(a_L\mid o_t)
\Vert
p_{t-h}(a_L\mid o_{t-h})
\right]
]

[
\Delta_R(t,h)
=============

D_{KL}
\left[
p_t(a_R\mid o_t)
\Vert
p_{t-h}(a_R\mid o_{t-h})
\right]
]

如果：

[
\Delta_R(t,h)<\epsilon_R
]

说明右臂策略在这段时间变化小，可以低频更新或 hold。

如果：

[
\Delta_R(t,h)>\epsilon_R
]

说明右臂需要重新规划。

但要判断是不是由左臂导致，还要看：

[
C_{L\rightarrow R}(t,h)
=======================

D_{KL}
\left[
p(a_R\mid a_{L,t-h},o)
\Vert
p(a_R\mid o)
\right]
]

如果：

[
C_{L\rightarrow R}(t,h)>\delta
]

说明右臂更新和左臂有强关系。

所以频率自适应可以定义为：

[
m_R(t)=
\mathbb{1}
[
\Delta_R(t,h)+\beta C_{L\rightarrow R}(t,h)>\epsilon_R
]
]

[
m_L(t)=
\mathbb{1}
[
\Delta_L(t,h)+\beta C_{R\rightarrow L}(t,h)>\epsilon_L
]
]

其中：

[
m_L,m_R\in{0,1}
]

表示左右臂是否在当前时刻更新动作。

这就把你“用 KL 确定 h / 确定是否紧密关系”的想法变成了可执行的数学规则。

---

## 9. 最推荐的统一模型

我建议你的最终模型不要直接写成：

[
p(a_L,a_R\mid o)
================

\frac12[
p(a_L\mid o)p(a_R\mid a_L,o)
+
p(a_R\mid o)p(a_L\mid a_R,o)
]
]

这个可以作为理论 motivation，但不是最优最终形式。

更推荐写成：

[
\boxed{
v_\theta
========

\pi_A v_A
+
\pi_{LR}v_{LR}
+
\pi_{RL}v_{RL}
+
\pi_Cv_C
}
]

其中：

[
\pi_k=\text{softmax}(g_\psi(o,a_L,a_R,t))
]

并加入结构代价：

[
\boxed{
\mathcal{L}
===========

\mathcal{L}*{FM/DDPM}
+
\lambda_c
\mathbb{E}
[
\pi_A\Omega_A+
\pi*{LR}\Omega_B+
\pi_{RL}\Omega_B+
\pi_C\Omega_C
]
+
\lambda_f\mathcal{R}_{freq}
}
]

其中：

[
\Omega_A>\Omega_B>\Omega_C
]

这个目标会自动学习：

* 强耦合时用 A；
* 左主右从时用 (L\rightarrow R)；
* 右主左从时用 (R\rightarrow L)；
* 弱耦合时用 C；
* 不需要每步更新时降低某只手的执行频率。

---

## 10. 数学证明部分可以这样组织

你可以证明三个核心命题。

### 命题 1：方案 C 的不可避免误差等于条件互信息

[
\min_{q_Lq_R}
D_{KL}
\left[
p^\star(a_L,a_R\mid o)
\Vert
q_L(a_L\mid o)q_R(a_R\mid o)
\right]
=======

I(a_L;a_R\mid o)
]

证明方式：最优 (q_L,q_R) 是真实边缘分布，代入 KL 后就是条件互信息定义。

这证明了：

> 如果左右臂条件互信息大，independent policy 必然不足。

---

### 命题 2：方案 B 的误差可以分解为 leader 误差和 follower 误差

[
D_{KL}
[
p^\star(a_L,a_R\mid o)
\Vert
q_L(a_L\mid o)q_R(a_R\mid a_L,o)
]
]

# [

D_{KL}
[
p^\star(a_L\mid o)
\Vert
q_L(a_L\mid o)
]
+
\mathbb{E}*{p^\star(a_L\mid o)}
D*{KL}
[
p^\star(a_R\mid a_L,o)
\Vert
q_R(a_R\mid a_L,o)
]
]

这证明了：

> conditional policy 的误差来自 leader 建模和 follower 条件建模两部分，并且会有 rollout distribution shift。

---

### 命题 3：结构选择是一个带复杂度惩罚的最小化问题

定义：

[
J(M)
====

\mathcal{E}_{approx}(M)
+
\alpha\Omega(M)
]

其中：

[
M\in{A,LR,RL,C}
]

[
\Omega(A)>\Omega(B)>\Omega(C)
]

最优结构为：

[
M^\star
=======

\arg\min_M J(M)
]

模型学习的 gating：

[
\pi_\psi(M\mid o,t)
]

就是在近似这个最优结构选择。

这证明了你要做的不是简单把 A/B/C 混起来，而是：

> 学习一个最小必要耦合结构。

---

## 11. 最后判断：你的方案是否可行？

我的判断是：

**可行，而且很有研究价值。**

但要做如下修正：

第一，不能直接把概率分布相加后再拆 log。
如果是 mixture，要用：

[
\log\left(\sum_k \pi_k p_k\right)
]

如果是 product-of-experts，才能把 log-density 相加。

第二，(w_0,w_\phi) 不应该是普通乘法常数，而应该是 density ratio / log-density residual：

[
r_{L\rightarrow R}
==================

\log
\frac{
p(a_R\mid a_L,o)
}{
p(a_R\mid o)
}
]

第三，用 KL 判断时序差异是可行的，但要区分：

[
D_{KL}(p_t(a_R)\Vert p_{t-h}(a_R))
]

衡量的是右臂自身重规划需求；

而：

[
D_{KL}(p(a_R\mid a_L,o)\Vert p(a_R\mid o))
]

衡量的才是左臂对右臂的耦合影响。

第四，最适合落地的方案是：

[
\boxed{
\text{adaptive mixture of flow/diffusion vector fields}
}
]

而 EBM 更适合做理论解释。

---

## 12. 你可以采用的最终训练目标

如果你用 flow matching，我建议最终写成：

[
z=(a_L,a_R)
]

[
z_t=(1-t)z_0+tz_1
]

[
u_t=z_1-z_0
]

[
v_\theta(z_t,t,o)
=================

\sum_{k\in{A,LR,RL,C}}
\pi_\psi(k\mid z_t,t,o)v_k(z_t,t,o)
]

训练目标：

[
\boxed{
\mathcal{L}
===========

\mathbb{E}
\left[
\left|
v_\theta(z_t,t,o)-u_t
\right|^2
\right]
+
\lambda_c
\mathbb{E}
\left[
\sum_k
\pi_\psi(k\mid z_t,t,o)\Omega_k
\right]
+
\lambda_f
\mathcal{R}*{freq}
+
\lambda_s
\mathcal{R}*{smooth}
}
]

其中：

[
\Omega_A>\Omega_{LR}=\Omega_{RL}>\Omega_C
]

频率正则：

[
\mathcal{R}_{freq}
==================

\mathbb{E}
[
m_L(t)+m_R(t)
]
]

平滑正则：

[
\mathcal{R}_{smooth}
====================

\sum_t
\left(
|a_{L,t+1}-a_{L,t}|^2
+
|a_{R,t+1}-a_{R,t}|^2
\right)
]

这个目标完整表达了你的研究想法：

> 用 imitation/flow loss 保证动作分布正确；
> 用 coupling cost 防止过度 joint；
> 用 frequency cost 学习非对称低频/高频执行；
> 用 gating 让模型自动在 A/B/C 之间切换。

这就是我认为最严谨、最可落地的数学版本。
