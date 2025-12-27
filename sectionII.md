\section{Methodology}

\subsection{Problem Formulation}
\label{sec:problem_formulation}

We study \textit{online time series forecasting} (OTSF) in non-stationary environments with delayed feedback. This setting is common in industrial forecasting and soft-sensing systems, where process conditions drift due to changes in raw materials (e.g., ore grade/mineralogy/hardness), equipment wear, control policy adjustments, and periodic operational schedules (e.g., day/night shifts or seasonal effects). Meanwhile, reliable supervision can be significantly delayed because key quality variables are obtained through downstream measurements or labor-intensive laboratory assays. For instance, in flotation, the concentrate grade (a typical label/quality KPI) often requires manual sampling and chemical analysis and may only be available hours later, which inherently introduces control and learning lag.

Let a multivariate stream be denoted by $\{\mathbf{x}_t\}_{t=1}^{T}$ with $C$ variables. At each time step $t$, the forecaster observes a look-back window $\mathbf{X}_t=[\mathbf{x}_{t-L+1},\dots,\mathbf{x}_t]\in\mathbb{R}^{L\times C}$ (length $L$) and outputs a multi-horizon forecast $\hat{\mathbf{Y}}_t=[\hat{\mathbf{x}}_{t+1},\dots,\hat{\mathbf{x}}_{t+H}]\in\mathbb{R}^{H\times C}$ for horizon $H$. The ground truth horizon $\mathbf{Y}_t=[\mathbf{x}_{t+1},\dots,\mathbf{x}_{t+H}]$ is only fully revealed after $H$ steps.

Our goal is to maintain a causal online forecaster $f_t$ that minimizes the cumulative loss $\sum_{t=1}^{T}\ell(\hat{\mathbf{Y}}_t,\mathbf{Y}_t)$ (e.g., MSE), while ensuring that any update executed when forecasting at time $t$ uses only information available up to that time.

\subsection{Overview of R-mem}
\label{sec:overview}

To address drift under delayed feedback, we propose \textbf{R-mem}, a residual-memory retrieval correction framework that decouples \textit{stable forecasting} from \textit{fast adaptation}. As shown in Fig.\ \ref{fig:framework}, R-mem consists of:
\begin{enumerate}
    \item \textbf{Frozen backbone (stability).} A pre-trained forecaster $f_{\theta}$ (e.g., iTransformer, PatchTST) produces a base forecast $\hat{\mathbf{Y}}^{\text{base}}_t$. During the online phase, $\theta$ is kept fixed to preserve generalizable temporal representations and avoid catastrophic forgetting under noisy short-term drift.
    \item \textbf{RRC corrector (plasticity).} A lightweight module, \textit{Residual Retrieval Corrector} (RRC), maintains an external \textit{residual memory} of historical \emph{residual trajectories} and produces an additive correction for the whole horizon.
    \item \textbf{Delayed evolution (causal memory writing).} R-mem caches prediction contexts and writes their realized residuals into memory only after the corresponding ground truth becomes available, strictly respecting feedback delay.
\end{enumerate}

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{fig1.png}
    \caption{Overall framework of \textbf{R-mem}. A frozen backbone produces the base multi-horizon forecast, while the \textbf{RRC} module retrieves and aggregates historical residual trajectories using a causally available \textbf{Condition Context Snippet} as the query, then applies horizon-aware masking and confidence gating to form an additive correction. Residuals are written into memory only when delayed ground truth arrives, ensuring strict causality.}
    \label{fig:framework}
\end{figure*}

\subsection{Condition Context Snippet as an Industrial ``Fingerprint''}
\label{sec:pogt}

A core question in retrieval-based correction is \emph{what should be used as the retrieval key}. In industrial streams, the same short-term pattern may correspond to different latent regimes (e.g., different operating modes, ore properties, or external conditions), causing \textit{state aliasing}. Therefore, R-mem relies on a short \textit{condition context snippet} (工况上下文片段) that acts as a compact fingerprint of the \emph{current operating condition and bias}, so that the corrector can quickly locate historical situations with similar drift signatures.

We denote this snippet by $\mathbf{G}_t\in\mathbb{R}^{P\times C}$ with length
\begin{equation}
    P=\max(1,\lfloor rH\rfloor), \quad r\in(0,1].
\end{equation}
\paragraph{Meaning.} Intuitively, $\mathbf{G}_t$ captures the most recent evidence about the current process state, which is often the earliest signal of a regime shift (e.g., ore property changes that propagate downstream). Its role is to enable RRC to retrieve from residual memory: \emph{``under similar fingerprints, what systematic forecasting errors did the backbone usually make?''} This is particularly important when the true supervision is delayed (e.g., flotation grade measured by offline assay). In that case, the snippet provides a timely context signal to retrieve and reuse historical residual trajectories for correcting $\hat{\mathbf{Y}}^{\text{base}}_t$ before the delayed label arrives.

\paragraph{How to obtain $\mathbf{G}_t$.} R-mem supports two practically relevant sources:
\begin{itemize}
    \item \textbf{True partially observed future (when available).} In some deployments, a small prefix of the forecasting target becomes available earlier than the full horizon (e.g., fast online analyzers provide early feedback while slow laboratory assays arrive later). Then $\mathbf{G}_t$ can be instantiated as the early-revealed prefix of $\mathbf{Y}_t$.
    \item \textbf{Proxy Condition Context Snippet from latest observations (used in our implementation).} When such early feedback is unavailable, we use a proxy Condition Context Snippet extracted from the most recent measurements. Specifically, in our implementation $\mathbf{G}_t$ is taken as the last $P$ steps of the latest observed sequence (the tail of $\mathbf{X}_t$), which often reflects the onset of drift (e.g., gradual equipment wear or a sudden control adjustment).
\end{itemize}

\paragraph{Industrial example.} Consider flotation and grinding circuits. The flotation concentrate grade (label) is often only available after hours due to manual assays, while the plant continuously observes fast variables (e.g., flows, densities, pressures, currents, and control setpoints). When the ore feed changes (grade/mineralogy/hardness), the immediate sensor trajectories in a short time window may shift noticeably, and this shift is tightly related to the upcoming bias of a pre-trained forecasting model. Using the recent snippet $\mathbf{G}_t$ as a fingerprint enables RRC to retrieve historical cases under similar ore-property-induced drift and reuse their residual trajectories to correct the current multi-horizon forecast, even before the delayed assay result becomes available.

\subsection{Residual-Trajectory Memory and Retrieval}
\label{sec:memory}
Fig.~\ref{fig:rrc} summarizes the internal workflow of RRC, and the following subsections detail its key steps.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{fig2.png}
    \caption{Internal workflow of \textbf{RRC}. The corrector encodes the Condition Context Snippet into a query key, retrieves top-$K$ residual trajectories from a bucketed memory, aggregates them into a candidate correction, and then applies horizon-aware masking and confidence gating to prevent negative transfer under non-stationarity.}
    \label{fig:rrc}
\end{figure*}

\subsubsection{Base Forecast and Residual Trajectory}
Given $\mathbf{X}_t$ and optional time features $\mathbf{S}_t$, the frozen backbone outputs:
\begin{equation}
    \hat{\mathbf{Y}}^{\text{base}}_t=f_{\theta}(\mathbf{X}_t,\mathbf{S}_t)\in\mathbb{R}^{H\times C}.
\end{equation}
When the full ground truth becomes available, we compute the realized residual \emph{trajectory} over the whole horizon:
\begin{equation}
    \mathbf{E}_t=\mathbf{Y}_t-\hat{\mathbf{Y}}^{\text{final}}_t\in\mathbb{R}^{H\times C}.
\end{equation}
We store $\mathbf{E}_t$ (not a scalar) because industrial drift often induces \emph{structured, horizon-dependent} errors (e.g., increasing bias with horizon, phase lag, or feature-coupled deviations). A full residual trajectory provides a reusable correction template.

\paragraph{Practical intuition.} In flotation and grinding, drift is frequently systematic rather than purely stochastic: ore property changes and gradual equipment wear can induce consistent over-/under-estimation patterns that evolve across the prediction horizon. By storing entire residual trajectories, RRC can reuse these structured patterns to correct multi-step forecasts in a coherent, horizon-consistent manner.

\subsubsection{Context Snippet Encoding and Similarity Vector}
We encode $\mathbf{G}_t$ using a lightweight temporal encoder $e_{\phi}(\cdot)$ with pooling and projection, yielding $\mathbf{z}_t\in\mathbb{R}^{d}$. The retrieval key is the $\ell_2$-normalized embedding:
\begin{equation}
    \mathbf{q}_t=\frac{\mathbf{z}_t}{\|\mathbf{z}_t\|_2}.
\end{equation}
The memory stores key--value pairs $\{(\mathbf{k}_i,\mathbf{E}_i)\}_{i=1}^{N}$, where $\mathbf{k}_i\in\mathbb{R}^{d}$ and $\mathbf{E}_i\in\mathbb{R}^{H\times C}$.

Given a query $\mathbf{q}_t$, we compute cosine similarity and apply an age-based decay to down-weight stale memories:
\begin{equation}
    s_{t,i}=\mathbf{q}_t^{\top}\mathbf{k}_i\cdot \gamma^{\,\text{age}_i}, \qquad \gamma\in(0,1].
\end{equation}
We retrieve the top-$K$ entries and obtain a \textit{similarity vector} $\mathbf{s}_t\in\mathbb{R}^{K}$ that stacks the retrieved similarities. This vector will later be used to assess retrieval quality.

\subsubsection{Aggregation to a Candidate Correction}
Let $\{\mathbf{E}_{(j)}\}_{j=1}^{K}$ denote the retrieved residual trajectories, and $\mathbf{s}_t=[s_{t,(1)},\dots,s_{t,(K)}]$. Our default aggregation uses a Softmax kernel:
\begin{equation}
    w_j=\frac{\exp(s_{t,(j)}/T)}{\sum_{m=1}^{K}\exp(s_{t,(m)}/T)},\qquad
    \bar{\mathbf{E}}_t=\sum_{j=1}^{K} w_j\mathbf{E}_{(j)},
\end{equation}
where $T$ is a temperature. The aggregated $\bar{\mathbf{E}}_t$ is a \emph{candidate correction trajectory} suggested by historical cases.

\subsection{Horizon-Aware Masking (Conservative Long-Horizon Correction)}
\label{sec:horizon_mask}

Even when retrieval is reliable, correction should be more conservative for distant horizons. In industrial forecasting, uncertainty typically grows with horizon, and error patterns become less transferable across samples---especially when unobserved disturbances (e.g., ore property fluctuations) propagate through the process. Therefore, we optionally apply a horizon-aware mask $\mathbf{m}\in\mathbb{R}^{H}$ to scale the correction trajectory step-by-step (before confidence gating):
\begin{equation}
    \mathbf{C}_t[h,:] \leftarrow \mathbf{m}[h]\cdot \mathbf{C}_t[h,:], \qquad h\in\{1,\dots,H\}.
\end{equation}

We support three modes for constructing $\mathbf{m}$ (one mode is selected in an experiment):
\begin{equation}
    \mathbf{m}[h]=
    \begin{cases}
    \delta^{h-1}, & \text{exponential (exp)}\\
    1-\frac{h-1}{\max(H-1,1)}, & \text{linear}\\
    \sigma(\mathbf{v}[h]), & \text{learned}
    \end{cases}
\end{equation}
where $\delta\in(0,1)$ controls exponential decay, and $\mathbf{v}\in\mathbb{R}^{H}$ is a learnable vector mapped to $(0,1)$ by sigmoid. In all cases, the mask is broadcast to $H\times C$ and applied multiplicatively, reducing correction magnitude as $h$ increases.

\paragraph{Example.} For $H=6$ and exp decay $\delta=0.9$, the mask is $[1,0.9,0.81,0.729,0.656,0.590]$, which enforces near-term aggressive correction while shrinking long-horizon corrections. Linear decay yields $[1,0.8,0.6,0.4,0.2,0]$ (optionally clamped by a minimum value in practice).

\subsection{Quality Estimation, Refinement, and Confidence Gating}
\label{sec:gating}

Retrieval correction is powerful but risky: if the retrieved cases come from a mismatched regime, applying the correction can degrade performance (negative transfer). This risk is non-trivial in industrial control loops: for example, an overly optimistic forecast of flotation grade may drive inappropriate reagent or air-flow adjustments, and the true grade feedback may only arrive hours later via offline assays. RRC therefore uses a set of lightweight reliability controls to decide \emph{how much} correction to apply (and when to fall back to the frozen backbone).

\subsubsection{Retrieval Quality Estimation}
A single maximum similarity $\max(\mathbf{s}_t)$ is insufficient to characterize reliability: high top-1 similarity with noisy remaining neighbors can indicate an unstable retrieval set. We thus estimate a retrieval-quality score $\rho_t\in(0,1)$ using a lightweight MLP $g_{\eta}$ over the full similarity vector and the context embedding:
\begin{equation}
    \rho_t=g_{\eta}([\mathbf{s}_t\,\|\,\mathbf{z}_t]).
\end{equation}

\subsubsection{Optional Refinement and Quality-Controlled Blending}
The aggregated correction $\bar{\mathbf{E}}_t$ is non-parametric and may be coarse. We optionally apply a refinement network $r_{\xi}(\cdot)$ that maps the candidate correction and context embedding to a refined trajectory:
\begin{equation}
    \tilde{\mathbf{E}}_t=r_{\xi}([\bar{\mathbf{E}}_t\,\|\,\mathbf{z}_t]).
\end{equation}
We then blend refined and raw corrections using $\rho_t$:
\begin{equation}
    \mathbf{C}_t=\rho_t\tilde{\mathbf{E}}_t+(1-\rho_t)\bar{\mathbf{E}}_t.
\end{equation}
This design allows RRC to be expressive when retrieval quality is high, while falling back to the more conservative raw aggregation when retrieval is uncertain.

\subsubsection{Similarity Gate and Compact Statistics Gate}
We compute a similarity gate based on the maximum similarity $s^{\max}_t=\max(\mathbf{s}_t)$:
\begin{equation}
    g^{\text{sim}}_t=\sigma\big(\kappa(s^{\max}_t-\tau)\big),
\end{equation}
where $\tau$ is a trust threshold and $\kappa$ controls steepness. This gate suppresses correction when even the best-matching memory is weak.

Additionally, to prevent abnormally large or unstable corrections, we use a compact statistics gate. Let
\begin{equation}
    \boldsymbol{\omega}_t=\big[\text{mean}(|\hat{\mathbf{Y}}^{\text{base}}_t|),\ \text{std}(\hat{\mathbf{Y}}^{\text{base}}_t),\ \text{mean}(|\mathbf{C}_t|),\ \text{std}(\mathbf{C}_t)\big].
\end{equation}
We compute $g^{\text{conf}}_t=g_{\psi}([\mathbf{z}_t\,\|\,\boldsymbol{\omega}_t])\in(0,1)$ using a small MLP. Using only compact summary statistics (rather than the full sequences) makes the gate robust and lightweight.

\subsubsection{Effective Confidence and Final Prediction}
Finally, RRC applies the correction only when retrieval is valid, and scales it by an effective confidence:
\begin{equation}
    \alpha_t=g^{\text{conf}}_t\cdot \rho_t\cdot g^{\text{sim}}_t\cdot \mathbb{I}[\exists\,\text{valid retrieval}].
\end{equation}
The final forecast is then:
\begin{equation}
    \hat{\mathbf{Y}}^{\text{final}}_t=\hat{\mathbf{Y}}^{\text{base}}_t+\alpha_t\mathbf{C}_t.
\end{equation}
\paragraph{Interpretation.} In industrial operation, the above design acts as a risk controller: when the memory contains highly similar cases with stable similarity structure, $\alpha_t$ increases and RRC corrects systematic bias; otherwise $\alpha_t$ approaches zero and the model falls back to the frozen backbone. This is particularly valuable under delayed labels (e.g., flotation grade from laboratory assays), where aggressive online fitting is both hard to perform causally and risky to validate promptly.

\paragraph{Illustrative workflow.} Suppose the top-$K$ similarities are $\mathbf{s}_t=[0.92,0.90,0.88]$ (stable and high), then $s^{\max}_t$ is above the trust threshold and $g^{\text{sim}}_t$ becomes large; the quality estimator can output a high $\rho_t$ because the entire vector indicates consistent neighbors. If the resulting correction magnitude is also reasonable (as measured by $\boldsymbol{\omega}_t$), $g^{\text{conf}}_t$ remains high and RRC applies a non-trivial correction. In contrast, if $\mathbf{s}_t=[0.55,0.52,0.50]$, then $g^{\text{sim}}_t\\approx 0$ and the effective confidence collapses, preventing negative transfer and reverting to the frozen backbone.

\subsection{Time-Aware Bucketization}
\label{sec:bucket}

Industrial processes often exhibit periodic regimes induced by scheduling and environment (e.g., shift patterns, day/night cycles, seasonal temperature). To reduce regime mixing in memory, we partition the memory into $B$ buckets, $\mathcal{M}=\{\mathcal{M}^{(b)}\}_{b=1}^{B}$. Given time features $\mathbf{S}_t$ (e.g., hour-of-day, day-of-week, month/season), we compute a bucket index:
\begin{equation}
    b_t = \pi(\mathbf{S}_t)\ \bmod\ B,
\end{equation}
and restrict retrieval and storage to $\mathcal{M}^{(b_t)}$. This period-aware routing reduces cross-regime interference and lowers retrieval cost by shrinking the candidate set per query.

\subsection{Delayed Feedback Evolution and Memory Maintenance}
\label{sec:evolution}

Due to the horizon $H$ (and, in many industrial settings, additional measurement delays), the realized residual $\mathbf{E}_t$ is available only after the corresponding ground truth arrives. For example, flotation grade labels may be produced hours after sampling via offline assays; thus, the model must operate and adapt under a strict information delay. In our evaluation protocol, we model the effective feedback delay as the forecasting horizon, and the same causal caching-and-writing mechanism naturally extends to longer delays by enlarging the pending buffer. R-mem implements a causal delayed update protocol by caching the context needed for future writing:
\begin{equation}
    \mathcal{C}_t=(\mathbf{G}_t,\hat{\mathbf{Y}}^{\text{final}}_t,b_t).
\end{equation}
When $\mathbf{Y}_t$ becomes available, we compute $\mathbf{E}_t=\mathbf{Y}_t-\hat{\mathbf{Y}}^{\text{final}}_t$ and insert $(\text{normalize}(e_{\phi}(\mathbf{G}_t)),\mathbf{E}_t)$ into the bucketed memory $\mathcal{M}^{(b_t)}$. Operationally, this corresponds to writing a new ``case'' into the memory only after the delayed label (e.g., assay-based grade) is received, ensuring that the online system never uses future information when producing forecasts.

Each bucket has a fixed capacity. When full, we evict the entry with the lowest combined score balancing importance, recency, and access frequency:
\begin{equation}
    \text{score}_i = 0.4\cdot \tilde{\text{imp}}_i + 0.4\cdot \tilde{\text{rec}}_i + 0.2\cdot \tilde{\text{freq}}_i,
\end{equation}
where importance is proportional to residual magnitude, recency is derived from age, and frequency reflects retrieval counts (each term normalized within the bucket). Optionally, we apply active forgetting by decaying importance scores and pruning entries below an importance threshold or above a maximum age.

\begin{algorithm}[tb]
\caption{R-mem Online Protocol with Bucketed RRC and Horizon Mask}
\label{alg:hmem_protocol}
\textbf{Input:} frozen backbone $f_\theta$, horizon $H$, snippet length $P$, top-$K$, buckets $B$, horizon mask $\mathbf{m}$ \\
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
    \STATE Encode snippet $\mathbf{z}_t\leftarrow e_{\phi}(\mathbf{G}_t)$ and normalize key $\mathbf{q}_t\leftarrow \text{normalize}(\mathbf{z}_t)$
    \STATE Retrieve top-$K$ residuals from $\mathcal{M}^{(b_t)}$ and obtain similarities $\mathbf{s}_t$
    \STATE Aggregate candidate correction $\bar{\mathbf{E}}_t\leftarrow \sum_{j=1}^{K} w_j \mathbf{E}_{(j)}$ with $w_j=\text{Softmax}(\mathbf{s}_t/T)$
    \STATE Estimate retrieval quality $\rho_t\leftarrow g_\eta([\mathbf{s}_t\,\|\,\mathbf{z}_t])$ and (optionally) refine $\tilde{\mathbf{E}}_t\leftarrow r_\xi([\bar{\mathbf{E}}_t\,\|\,\mathbf{z}_t])$
    \STATE Form correction $\mathbf{C}_t\leftarrow \rho_t\tilde{\mathbf{E}}_t+(1-\rho_t)\bar{\mathbf{E}}_t$ and apply horizon mask $\mathbf{C}_t\leftarrow \mathbf{m}\odot \mathbf{C}_t$
    \STATE Compute effective confidence $\alpha_t$ via similarity/quality/statistics gates and output $\hat{\mathbf{Y}}^{\text{final}}_t\leftarrow \hat{\mathbf{Y}}^{\text{base}}_t+\alpha_t\mathbf{C}_t$
    \STATE Push $(\mathbf{G}_t,\hat{\mathbf{Y}}^{\text{final}}_t,b_t)$ into $\mathcal{W}$
\ENDFOR
\end{algorithmic}
\end{algorithm}

\subsection{Computational Complexity}
\label{sec:complexity}

Let $C$ be the number of variables, $d$ the RRC feature dimension, and $N_b$ the number of entries in the active bucket.
\paragraph{Time.} In each step, RRC adds (i) snippet encoding $O(P\cdot C\cdot d)$, (ii) exact similarity computation $O(N_b\cdot d)$, and (iii) residual aggregation $O(K\cdot H\cdot C)$, in addition to one forward pass of the frozen backbone. Bucketization reduces $N_b$ by routing retrieval to a subset of memory entries.
\paragraph{Space.} The memory stores keys and residual trajectories, requiring $O(N_b\cdot(d+H\cdot C))$ per bucket (with fixed total capacity). Compared with gradient-based online adaptation that maintains optimizer states for large backbones, R-mem offers a compact memory footprint and a practical accuracy--efficiency trade-off.
