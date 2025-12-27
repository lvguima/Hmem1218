\section{Methodology}

\subsection{Problem Formulation}
\label{sec:problem_formulation}

We study \textit{online time series forecasting} (OTSF) in non-stationary environments with delayed feedback. Let a multivariate stream be denoted by $\{\mathbf{x}_t\}_{t=1}^{T}$ with $C$ variables. At each time step $t$, the forecaster observes a look-back window $\mathbf{X}_t=[\mathbf{x}_{t-L+1},\dots,\mathbf{x}_t]\in\mathbb{R}^{L\times C}$ (length $L$) and outputs a multi-horizon forecast $\hat{\mathbf{Y}}_t=[\hat{\mathbf{x}}_{t+1},\dots,\hat{\mathbf{x}}_{t+H}]\in\mathbb{R}^{H\times C}$ for horizon $H$.

Two constraints make OTSF fundamentally different from offline forecasting:
\begin{enumerate}
    \item \textbf{Concept drift.} The data distribution evolves over time, i.e., $P_t(\mathbf{X},\mathbf{Y})\neq P_{t+\Delta}(\mathbf{X},\mathbf{Y})$, so a static model degrades as the environment shifts.
    \item \textbf{Delayed feedback.} The ground truth $\mathbf{Y}_t$ for a horizon-$H$ forecast is only fully revealed after $H$ steps. Thus, when making a prediction at time $t$, the latest fully available supervision corresponds to some past step (e.g., $\mathbf{Y}_{t-H}$).
\end{enumerate}

Our goal is to maintain a causal online forecaster $f_t$ that minimizes the cumulative loss $\sum_{t=1}^{T}\ell(\hat{\mathbf{Y}}_t,\mathbf{Y}_t)$ (e.g., MSE), while ensuring that any update performed when forecasting at time $t$ uses only the supervision that has already arrived.

\subsection{Overview of H-Mem}
\label{sec:overview}

To address drift under delayed feedback, we propose \textbf{H-Mem}, a retrieval-augmented framework that decouples \textit{stable forecasting} from \textit{fast adaptation}. As shown in Fig.\ \ref{fig:framework}, H-Mem consists of:
\begin{enumerate}
    \item \textbf{Frozen backbone (stability).} A pre-trained forecaster $f_{\theta}$ (e.g., iTransformer, PatchTST) produces a base forecast $\hat{\mathbf{Y}}^{\text{base}}_t$. During the online phase, $\theta$ is kept fixed to prevent catastrophic forgetting and to preserve generalizable temporal representations.
    \item \textbf{CHRC corrector (plasticity).} A lightweight module, \textit{Cross-Horizon Retrieval Corrector} (CHRC), maintains an external \textit{error memory} of historical residual trajectories and produces an additive correction for the whole horizon.
    \item \textbf{Delayed evolution (causal memory writing).} H-Mem caches prediction contexts and writes their realized residuals into memory only after the corresponding ground truth becomes available, strictly respecting feedback delay.
\end{enumerate}

\subsection{State Aliasing and Conservative Correction}
\label{sec:state_aliasing}

Retrieval-based correction assumes that similar contexts imply similar forecast errors. In non-stationary systems, however, \textit{context similarity can be aliased}: similar short-term patterns may occur under different latent regimes (seasonality, operating modes, control policies), leading to distinct future residuals. If the corrector blindly applies retrieved residuals, mismatched retrieval may induce \textit{negative transfer}.

H-Mem mitigates this risk via two complementary mechanisms aligned with the final implementation: (i) \textbf{time-aware bucketization} to reduce cross-regime interference, and (ii) \textbf{confidence-controlled correction} that automatically falls back to the frozen backbone when retrieval evidence is weak or unreliable.

\subsection{Cross-Horizon Retrieval Corrector (CHRC)}
\label{sec:chrc}

\subsubsection{Base Forecast}
Given $\mathbf{X}_t$ and optional time features $\mathbf{S}_t$, the frozen backbone outputs:
\begin{equation}
    \hat{\mathbf{Y}}^{\text{base}}_t=f_{\theta}(\mathbf{X}_t,\mathbf{S}_t)\in\mathbb{R}^{H\times C}.
\end{equation}

\subsubsection{POGT (or Proxy) and Retrieval Key}
CHRC conditions on a short \textit{partially observed context snippet} $\mathbf{G}_t\in\mathbb{R}^{P\times C}$, where
\begin{equation}
    P=\max(1,\lfloor rH\rfloor), \quad r\in(0,1].
\end{equation}
Depending on application constraints, $\mathbf{G}_t$ can be instantiated as an early-revealed segment related to the horizon or a proxy extracted from the most recently available measurements. We encode $\mathbf{G}_t$ using a lightweight temporal encoder $e_{\phi}(\cdot)$ with temporal pooling and projection, yielding $\mathbf{z}_t\in\mathbb{R}^{d}$. The retrieval key is the $\ell_2$-normalized embedding:
\begin{equation}
    \mathbf{q}_t=\frac{\mathbf{z}_t}{\|\mathbf{z}_t\|_2}.
\end{equation}

\subsubsection{Error Memory Bank with Temporal Decay}
H-Mem maintains an external memory bank storing key--value pairs $\{(\mathbf{k}_i,\mathbf{E}_i)\}_{i=1}^{N}$, where $\mathbf{k}_i\in\mathbb{R}^{d}$ and $\mathbf{E}_i\in\mathbb{R}^{H\times C}$ is the realized multi-horizon residual trajectory. Given $\mathbf{q}_t$, we compute cosine similarity and apply an age-based decay:
\begin{equation}
    s_{t,i}=\mathbf{q}_t^{\top}\mathbf{k}_i\cdot \gamma^{\,\text{age}_i}, \qquad \gamma\in(0,1].
\end{equation}
We retrieve the top-$K$ entries according to $s_{t,i}$ and aggregate residuals. Our default aggregation uses a Softmax kernel:
\begin{equation}
    w_j=\frac{\exp(s_{t,(j)}/T)}{\sum_{m=1}^{K}\exp(s_{t,(m)}/T)},\qquad
    \bar{\mathbf{E}}_t=\sum_{j=1}^{K} w_j\mathbf{E}_{(j)},
\end{equation}
where $T$ is a temperature and $(j)$ indexes the retrieved neighbors. If the memory is insufficient (cold start) or all similarities are below a minimum threshold, CHRC deactivates retrieval.

\subsubsection{Horizon-Aware Masking}
Residual retrieval is typically more reliable for near-term horizons than for distant ones. We therefore optionally apply a horizon-aware mask $\mathbf{m}\in\mathbb{R}^{H}$ (broadcast to $H\times C$):
\begin{equation}
    \mathbf{m}[h]=
    \begin{cases}
    \delta^{h-1}, & \text{exponential}\\
    1-\frac{h-1}{\max(H-1,1)}, & \text{linear}\\
    \sigma(\mathbf{u}[h]), & \text{learned}
    \end{cases}
\end{equation}
where $\delta\in(0,1)$ is a decay factor (optionally clamped from below) and $\mathbf{u}$ is a learnable vector. The masked residual becomes $\bar{\mathbf{E}}_t \leftarrow \mathbf{m}\odot \bar{\mathbf{E}}_t$.

\subsubsection{Quality Estimation, Refinement, and Confidence Gating}
To prevent negative transfer, CHRC estimates a \textit{retrieval quality} $\rho_t\in(0,1)$ using a small MLP $g_{\eta}$ over the similarity vector and the context embedding:
\begin{equation}
    \rho_t=g_{\eta}([\mathbf{s}_t \,\|\, \mathbf{z}_t]),
\end{equation}
where $\mathbf{s}_t\in\mathbb{R}^{K}$ stacks the retrieved similarities.

Optionally, a refinement network $r_{\xi}(\cdot)$ maps $(\bar{\mathbf{E}}_t,\mathbf{z}_t)$ to a refined correction $\tilde{\mathbf{E}}_t$. The final correction is a convex blend:
\begin{equation}
    \mathbf{C}_t = \rho_t \tilde{\mathbf{E}}_t + (1-\rho_t)\bar{\mathbf{E}}_t.
\end{equation}

Finally, CHRC computes an effective confidence $\alpha_t\in[0,1]$ using three factors:
\begin{enumerate}
    \item \textbf{Similarity gate.} Let $s^{\max}_t=\max_j s_{t,(j)}$. We compute
    \begin{equation}
        g^{\text{sim}}_t=\sigma\big(\kappa(s^{\max}_t-\tau)\big),
    \end{equation}
    where $\tau$ is a trust threshold and $\kappa$ controls steepness.
    \item \textbf{Quality gate.} $\rho_t$ reflects confidence based on the full similarity profile.
    \item \textbf{Compact statistics gate.} To avoid high-dimensional overfitting, we summarize the base forecast and the correction by
    \begin{equation}
        \mathbf{u}_t=\big[\text{mean}(|\hat{\mathbf{Y}}^{\text{base}}_t|),\ \text{std}(\hat{\mathbf{Y}}^{\text{base}}_t),\ \text{mean}(|\mathbf{C}_t|),\ \text{std}(\mathbf{C}_t)\big],
    \end{equation}
    and obtain $g^{\text{conf}}_t=g_{\psi}([\mathbf{z}_t\,\|\,\mathbf{u}_t])\in(0,1)$ from another small MLP.
\end{enumerate}
The final confidence is:
\begin{equation}
    \alpha_t=g^{\text{conf}}_t \cdot \rho_t \cdot g^{\text{sim}}_t \cdot \mathbb{I}[\exists\,\text{valid retrieval}],
\end{equation}
and the final forecast is:
\begin{equation}
    \hat{\mathbf{Y}}^{\text{final}}_t = \hat{\mathbf{Y}}^{\text{base}}_t + \alpha_t \cdot \mathbf{C}_t.
\end{equation}
This design ensures that CHRC is aggressive only when retrieval evidence is strong, and conservative otherwise.

\subsection{Time-Aware Bucketization}
\label{sec:bucket}

To reduce regime mixing, we partition the memory into $B$ buckets, $\mathcal{M}=\{\mathcal{M}^{(b)}\}_{b=1}^{B}$. Given time features $\mathbf{S}_t$ (e.g., hour-of-day, day-of-week, month/season), we compute a bucket index:
\begin{equation}
    b_t = \pi(\mathbf{S}_t)\ \bmod\ B,
\end{equation}
and restrict retrieval and storage to $\mathcal{M}^{(b_t)}$. This period-aware routing both reduces cross-regime interference and lowers retrieval cost by shrinking the candidate set per query.

\subsection{Delayed Feedback Evolution and Memory Maintenance}
\label{sec:evolution}

Due to the horizon $H$, the realized residual $\mathbf{E}_t=\mathbf{Y}_t-\hat{\mathbf{Y}}^{\text{final}}_t$ is only available after $H$ steps. H-Mem implements a causal delayed update protocol by caching the context needed for future writing:
\begin{equation}
    \mathcal{C}_t=(\mathbf{G}_t,\hat{\mathbf{Y}}^{\text{final}}_t,b_t).
\end{equation}
When $\mathbf{Y}_t$ becomes available, we compute $\mathbf{E}_t$ and write $(\text{normalize}(e_{\phi}(\mathbf{G}_t)),\mathbf{E}_t)$ into $\mathcal{M}^{(b_t)}$.

Each bucket has a fixed capacity. When full, we evict the entry with the lowest combined score balancing importance, recency, and access frequency:
\begin{equation}
    \text{score}_i = 0.4\cdot \tilde{\text{imp}}_i + 0.4\cdot \tilde{\text{rec}}_i + 0.2\cdot \tilde{\text{freq}}_i,
\end{equation}
where importance is proportional to residual magnitude, recency is derived from age, and frequency reflects retrieval counts (each term normalized within the bucket). Optionally, we apply active forgetting by decaying importance scores and pruning entries below an importance threshold or above a maximum age.

\begin{algorithm}[tb]
\caption{H-Mem Online Protocol with Bucketed CHRC and Horizon Mask}
\label{alg:hmem_protocol}
\textbf{Input:} frozen backbone $f_\theta$, horizon $H$, POGT length $P$, top-$K$, buckets $B$, horizon mask $\mathbf{m}$ \\
\textbf{Initialize:} bucketed memory $\{\mathcal{M}^{(b)}\}_{b=1}^{B}$, pending buffer $\mathcal{W}\leftarrow\emptyset$
\begin{algorithmic}[1]
\FOR{each time step $t=1,2,\dots$}
    \STATE \textbf{// (A) Delayed write for $t-H$}
    \IF{$t>H$}
        \STATE Pop $(\mathbf{G}_{t-H},\hat{\mathbf{Y}}^{\text{final}}_{t-H},b_{t-H})$ from $\mathcal{W}$
        \STATE Observe $\mathbf{Y}_{t-H}$ and compute $\mathbf{E}_{t-H}\leftarrow \mathbf{Y}_{t-H}-\hat{\mathbf{Y}}^{\text{final}}_{t-H}$
        \STATE Insert $(\text{normalize}(e_{\phi}(\mathbf{G}_{t-H})),\mathbf{E}_{t-H})$ into $\mathcal{M}^{(b_{t-H})}$ with eviction/forgetting
    \ENDIF
    \STATE \textbf{// (B) Forecast at time $t$}
    \STATE Receive $\mathbf{X}_t$ and time features $\mathbf{S}_t$; set $b_t\leftarrow\pi(\mathbf{S}_t)\bmod B$
    \STATE Extract context snippet $\mathbf{G}_t\in\mathbb{R}^{P\times C}$ from available measurements
    \STATE Base forecast $\hat{\mathbf{Y}}^{\text{base}}_t\leftarrow f_\theta(\mathbf{X}_t,\mathbf{S}_t)$
    \STATE Retrieve top-$K$ residuals from $\mathcal{M}^{(b_t)}$, aggregate $\bar{\mathbf{E}}_t$, apply $\bar{\mathbf{E}}_t\leftarrow \mathbf{m}\odot \bar{\mathbf{E}}_t$
    \STATE Compute $\rho_t$ and $\alpha_t$, obtain correction $\mathbf{C}_t$, and output $\hat{\mathbf{Y}}^{\text{final}}_t=\hat{\mathbf{Y}}^{\text{base}}_t+\alpha_t\mathbf{C}_t$
    \STATE Push $(\mathbf{G}_t,\hat{\mathbf{Y}}^{\text{final}}_t,b_t)$ into $\mathcal{W}$
\ENDFOR
\end{algorithmic}
\end{algorithm}

\subsection{Computational Complexity}
\label{sec:complexity}

Let $C$ be the number of variables, $d$ the CHRC feature dimension, and $N_b$ the number of entries in the active bucket.
\paragraph{Time.} In each step, CHRC adds (i) POGT encoding $O(P\cdot C\cdot d)$, (ii) exact similarity computation $O(N_b\cdot d)$, and (iii) residual aggregation $O(K\cdot H\cdot C)$, in addition to one forward pass of the frozen backbone. Bucketization reduces $N_b$ by routing retrieval to a subset of memory entries.
\paragraph{Space.} The memory stores keys and residual trajectories, requiring $O(N_b\cdot(d+H\cdot C))$ per bucket (with fixed total capacity). Compared with gradient-based online adaptation that maintains optimizer states for large backbones, H-Mem offers a compact memory footprint and a practical accuracy--efficiency trade-off.
