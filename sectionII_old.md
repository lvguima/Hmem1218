\section{Methodology}

\subsection{Problem Formulation}
\label{sec:problem_formulation}

We consider the problem of \textit{Online Time Series Forecasting} (OTSF) in non-stationary environments. Let $\mathcal{S} = \{(\mathbf{X}_t, \mathbf{Y}_t)\}_{t=1}^{T}$ denote a continuous data stream, where at each time step $t$, the model receives a look-back window $\mathbf{X}_t \in \mathbb{R}^{L \times C}$ (length $L$, $C$ variables) and must predict the future horizon $\mathbf{Y}_t \in \mathbb{R}^{H \times C}$.

Unlike offline training where the joint distribution $P(\mathbf{X}, \mathbf{Y})$ is assumed static, real-world streams impose two critical constraints:
\begin{enumerate}
    \item \textbf{Concept Drift:} The data distribution varies over time, i.e., $P_t(\mathbf{X}, \mathbf{Y}) \neq P_{t+\Delta}(\mathbf{X}, \mathbf{Y})$. A static model trained on initial data will suffer from performance degradation as the environment evolves.
    \item \textbf{Delayed Feedback:} Due to the forecasting horizon $H$, the ground truth $\mathbf{Y}_t$ is not immediately available for supervision at time $t$. Instead, it is revealed after a delay of $H$ steps. Consequently, at inference time $t$, the most recent available supervision is the pair $(\mathbf{X}_{t-H}, \mathbf{Y}_{t-H})$.
\end{enumerate}

Our objective is to maintain an accurate predictive function $f_t: \mathbf{X}_t \to \hat{\mathbf{Y}}_t$ that adapts to drifts online, minimizing the cumulative loss over time: $\mathcal{L}_{total} = \sum_{t=1}^T \|\hat{\mathbf{Y}}_t - \mathbf{Y}_t\|_2^2$, strictly respecting the causality constraint that updates at time $t$ can only utilize information available up to $t-H$.

\subsection{Overview of H-Mem Framework}
\label{sec:overview}

To address the challenges of concept drift and delayed feedback, we propose the \textbf{H-Mem} framework. As illustrated in Fig.\~\ref{fig:framework}, H-Mem adopts a \textit{Decoupled Retrieval Architecture} that functionally separates the modeling of universal physical laws from the adaptation to evolving environmental biases.

The framework consists of three key operational streams:
\begin{enumerate}
    \item \textbf{Stability Stream (Frozen Backbone):} We employ a pre-trained deep forecasting model (e.g., iTransformer, PatchTST) as the stability anchor. Its parameters $\theta$ remain \textbf{strictly frozen} during the online phase. This backbone extracts time-invariant features to generate a robust ``base forecast'' $\hat{\mathbf{Y}}_{base}$, preventing catastrophic forgetting often seen in gradient-based adaptation.
    
    \item \textbf{Plasticity Stream (CHRC):} To capture sudden drifts, we introduce the \textit{Cross-Horizon Retrieval Corrector} (CHRC). Acting as a \textbf{Neural Case-Based Reasoning} module, CHRC maintains an external memory of historical residuals. During inference, it retrieves and aggregates past error patterns—specifically those from similar contexts—to instantaneously rectify the base forecast. This is a non-parametric adaptation process requiring no backpropagation.

    \item \textbf{Evolution Stream (Delayed Feedback):} To respect the causality constraint, we implement a \textit{Delayed Feedback Loop}. A FIFO buffer holds prediction contexts until their corresponding ground truth arrives at $t+H$. Only then are the realized errors computed and stored in the memory, ensuring the system's knowledge base evolves continuously with the data stream.
\end{enumerate}

\subsection{The State Aliasing Challenge}
\label{sec:state_aliasing}

A core premise of retrieval-based adaptation is that \textit{similar environmental states yield similar model errors}. However, defining the ``state'' solely by the input observation $\mathbf{X}_t$ (or the Partially Observed Ground Truth, POGT) is fundamentally insufficient in non-stationary environments. We term this limitation \textbf{State Aliasing}.

In complex dynamical systems, the mapping from observed history to future targets is one-to-many depending on latent regimes (e.g., market sentiment, seasonal context, or hidden exogenous variables). Consequently, two time steps $t_i$ and $t_j$ may share nearly identical input patterns ($\mathbf{X}_{t_i} \approx \mathbf{X}_{t_j}$) yet exhibit divergent ground truth trajectories due to differing underlying dynamics. If a corrector relies exclusively on input similarity, it risks retrieving historical errors from an incompatible regime—a phenomenon known as \textit{Negative Transfer}.

To resolve this ambiguity, H-Mem posits that the \textit{forecasting model's own output} serves as a powerful latent descriptor. The base prediction $\hat{\mathbf{Y}}_{base}$, while potentially biased, explicitly encodes the model's perception of the trend and periodicity. \subsection{Cross-Horizon Retrieval Corrector (CHRC)}
\label{sec:chrc}

The CHRC acts as the plasticity engine of H-Mem. It is designed to be a non-parametric, inference-time adaptation module that stores specific error patterns and retrieves them to correct future predictions.

\subsubsection{Regime-Aware Memory Construction}
To prevent the retrieval of historically irrelevant errors (e.g., using a weekend error pattern to correct a weekday forecast), we structurally enforce temporal consistency via \textit{Regime-Aware Bucketing}. The global memory space $\mathcal{M}$ is partitioned into disjoint buckets $\mathcal{B}_1, \dots, \mathcal{B}_N$ based on explicit time features associated with each step $t$.

Let $\mathbf{\tau}_t \in \mathbb{R}^K$ be the time covariates (e.g., hour of day, day of week). We assign the current step to a specific bucket using a hashing function:
\begin{equation}
    k_{bucket} = \mathcal{H}(\mathbf{\tau}_t) \pmod N
\end{equation}
Each bucket $\mathcal{B}_k$ operates as an independent First-In-First-Out (FIFO) queue storing Key-Value pairs $\{(\mathbf{K}_i, \mathbf{V}_i)\}$, where $\mathbf{V}_i \in \mathbb{R}^{H \times C}$ represents the historical error vector. During inference, retrieval is strictly confined to the active bucket $\mathcal{B}_{k_{bucket}}$, ensuring that the model only references experiences from compatible temporal regimes.

\subsubsection{Dual-Key Contextual Retrieval}
\label{sec:dual_key}

To implement the resolution of State Aliasing (Sec.~\ref{sec:state_aliasing}), we propose a \textit{Dual-Key Retrieval} mechanism. Instead of a single query vector, we construct a composite query $\mathbf{Q}_t$ by fusing two distinct information sources:

\begin{enumerate}
    \item \textbf{Observation Key ($\mathbf{k}_{obs}$):} Encodes the recent environmental state. We use the input look-back window (or POGT) $\mathbf{X}_t$:
    \begin{equation}
        \mathbf{k}_{obs} = \phi_{obs}(\text{Flatten}(\mathbf{X}_t))
    \end{equation}
    where $\phi_{obs}$ is a lightweight MLP encoder projecting the raw input into a latent feature space $\mathbb{R}^d$.

    \item \textbf{Intent Key ($\mathbf{k}_{pred}$):} Encodes the backbone's current predictive intent. We process the base forecast $\hat{\mathbf{Y}}_{base}$:
    \begin{equation}
        \mathbf{k}_{pred} = \phi_{pred}(\text{Flatten}(\hat{\mathbf{Y}}_{base}))
    \end{equation}
    where $\phi_{pred}$ is a separate encoder mapping the prediction trajectory to $\mathbb{R}^d$.
\end{enumerate}

The final query $\mathbf{Q}_t$ is obtained by fusing these representations via a linear projection layer $\mathbf{W}_{fuse}$:
\begin{equation}
    \mathbf{Q}_t = \mathbf{W}_{fuse} \cdot [\mathbf{k}_{obs} \mathbin{\|} \mathbf{k}_{pred}]
\end{equation}
where $\mathbin{\|}$ denotes concatenation. The memory keys $\mathbf{K}_i$ stored in the bank are constructed identically using the historical data available at time $t-i$. This composite key ensures that a historical error is retrieved only when both the \textit{situation} (input) and the \textit{response} (prediction) align with the current state.

\subsubsection{Adaptive Correction and Gating}
Once the query $\mathbf{Q}_t$ is formed, we retrieve the top-$k$ nearest neighbors from the active bucket $\mathcal{B}_{k_{bucket}}$ based on cosine similarity:
\begin{equation}
    s_i = \frac{\mathbf{Q}_t \cdot \mathbf{K}_i}{\|\mathbf{Q}_t\| \|\mathbf{K}_i\|}, \quad (\mathbf{K}_i, \mathbf{V}_i) \in \mathcal{N}_k
\end{equation}

\paragraph{Confidence Gating}
A naive retrieval system risks applying harmful corrections when no relevant history exists (i.e., when all $s_i$ are low). To mitigate this, we introduce an \textit{Adaptive Confidence Gate} $\alpha_t \in (0, 1)$. Let $s_{max} = \max_i(s_i)$ be the similarity of the best match. The gate is computed via a shifted sigmoid function:
\begin{equation}
    \alpha_t = \sigma\left( \gamma \cdot (s_{max} - \tau) \right)
\end{equation}
where $\tau$ is a learnable trust threshold and $\gamma$ controls the transition steepness. This mechanism effectively suppresses the correction module ($\alpha_t \to 0$) when the retrieval confidence is below the trust threshold, acting as an automatic fallback to the stable backbone.

\paragraph{Horizon-Aware Aggregation}
The retrieved residual patterns are aggregated using a Softmax kernel over the similarities to produce the raw correction vector:
\begin{equation}
    \Delta_{raw} = \sum_{i=1}^k \frac{\exp(s_i / T)}{\sum_{j=1}^k \exp(s_j / T)} \cdot \mathbf{V}_i
\end{equation}
where $T$ is a temperature parameter. Furthermore, acknowledging that error patterns at distant horizons are intrinsically more stochastic and less "retrievable" than near-term errors, we apply a \textit{Horizon Mask} $\mathbf{M} \in \mathbb{R}^H$. The mask enforces an exponential decay on the correction strength over the prediction horizon $h \in \{1, \dots, H\}$:
\begin{equation}
    \mathbf{M}[h] = \beta^{h-1}
\end{equation}
where $\beta \in (0, 1)$ is a decay factor. The final rectified prediction is obtained by:
\begin{equation}
    \hat{\mathbf{Y}}_{final} = \hat{\mathbf{Y}}_{base} + \alpha_t \cdot (\mathbf{M} \odot \Delta_{raw})
\end{equation}
This ensures that H-Mem aggressively corrects near-term forecasts where retrieval is reliable, while remaining conservative for long-term predictions.

\subsection{Delayed Feedback Evolution}
\label{sec:evolution}

To maintain strict adherence to causality while enabling continuous learning, H-Mem implements a decoupled \textit{Delayed Feedback Loop}. This mechanism manages the lifecycle of experience samples, handling the temporal gap between prediction generation and error realization.

At inference time $t$, the system generates the context tuple $\mathcal{C}_t = (\mathbf{Q}_t, \hat{\mathbf{Y}}_{base})$ and the final forecast $\hat{\mathbf{Y}}_{final}$. However, the ground truth $\mathbf{Y}_t$ is unknown. Thus, instead of updating the memory immediately, we push $\mathcal{C}_t$ into a pending buffer $\mathcal{W}$ of size $H$.

The memory update is triggered only at time $t+H$, when the ground truth $\mathbf{Y}_t$ becomes fully observable. The process proceeds as follows:
\begin{enumerate}
    \item \textbf{Context Retrieval:} The system pops the historical context $\mathcal{C}_t$ from the buffer $\mathcal{W}$.
    \item \textbf{Error Realization:} The true prediction error of the backbone is computed: $\mathbf{V}_{new} = \mathbf{Y}_t - \hat{\mathbf{Y}}_{base}$. Note that we store the backbone's error, not the final corrected error, as the goal is to capture the bias of the stability anchor.
    \item \textbf{Memory Write:} The new key-value pair $(\mathbf{Q}_t, \mathbf{V}_{new})$ is inserted into the appropriate regime bucket $\mathcal{B}_{k_{bucket}}$.
\end{enumerate}

To adapt to continuous concept drift, each bucket $\mathcal{B}_k$ maintains a fixed capacity $M$. When a bucket is full, the oldest entries are evicted (FIFO). This ensures that the corrector's knowledge base remains current, effectively ``forgetting'' obsolete error patterns that no longer reflect the environment's dynamics.

\begin{algorithm}[tb]
\caption{H-Mem Online Protocol: Dual-Key Retrieval \& Correction}
\label{alg:hmem_protocol}
\textbf{Input:} Frozen Backbone $f_\theta$, Horizon $H$, Memory Capacity $M$, Neighbors $K$, Temp $T$, Trust $\tau$, Steepness $\gamma$, Decay $\beta$ \\
\textbf{Initialize:} Memory Buckets $\mathcal{M} = \{\mathcal{B}_1, \dots, \mathcal{B}_N\}$, Pending Buffer $\mathcal{W} \leftarrow \emptyset$
\begin{algorithmic}[1]
\FOR{each time step $t = 1, 2, \dots$}
    \STATE \textbf{// Step 1: Stability Stream}
    \STATE Receive input $\mathbf{X}_t$ and time features $\tau_t$
    \STATE Base forecast: $\hat{\mathbf{Y}}_{\text{base}} \leftarrow f_\theta(\mathbf{X}_t)$

    \STATE \textbf{// Step 2: Plasticity Stream (CHRC)}
    \STATE Select bucket: $k \leftarrow \mathcal{H}(\tau_t) \pmod N$
    \STATE Build Keys: $\mathbf{k}_{obs} \leftarrow \phi_{obs}(\mathbf{X}_t), \ \mathbf{k}_{pred} \leftarrow \phi_{pred}(\hat{\mathbf{Y}}_{\text{base}})$
    \STATE Fuse Query: $\mathbf{Q}_t \leftarrow \mathbf{W}_{fuse} \cdot [\mathbf{k}_{obs} \| \mathbf{k}_{pred}]$
    
    \IF{$|\mathcal{B}_k| \ge K$}
        \STATE Retrieve top-$K$: $\{(\mathbf{K}_i, \mathbf{V}_i)\}_{i=1}^K \leftarrow \text{KNN}(\mathbf{Q}_t, \mathcal{B}_k)$
        \STATE Sim & Gate: $s_{max} \leftarrow \max(s_i); \ \alpha_t \leftarrow \sigma(\gamma(s_{max} - \tau))$
        \STATE Weights: $w_i \leftarrow \text{Softmax}(s_i / T)$
        \STATE Mask: $\mathbf{M}[h] \leftarrow \beta^{h-1}$ for $h \in 1..H$
        \STATE Correction: $\Delta_t \leftarrow \alpha_t \cdot (\mathbf{M} \odot \sum w_i \mathbf{V}_i)$
        \STATE Final Pred: $\hat{\mathbf{Y}}_t \leftarrow \hat{\mathbf{Y}}_{\text{base}} + \Delta_t$
    \ELSE
        \STATE $\hat{\mathbf{Y}}_t \leftarrow \hat{\mathbf{Y}}_{\text{base}}$ \COMMENT{Cold start or empty bucket}
    \ENDIF
    
    \STATE Push context $(t, \mathbf{Q}_t, \hat{\mathbf{Y}}_{\text{base}}, k)$ to Buffer $\mathcal{W}$
    \STATE \textbf{Output} $\hat{\mathbf{Y}}_t$
    
    \STATE \textbf{// Step 3: Delayed Evolution (at $t+H$)}
    \IF{$t > H$}
        \STATE Receive ground-truth $\mathbf{Y}_{t-H}$
        \STATE Pop $(\mathbf{Q}_{old}, \hat{\mathbf{Y}}_{old}, k_{old}) \leftarrow \mathcal{W}.\text{pop}(t-H)$
        \STATE Realized Error: $\mathbf{V}_{\text{new}} \leftarrow \mathbf{Y}_{t-H} - \hat{\mathbf{Y}}_{old}$
        \STATE Update Memory: $\mathcal{B}_{k_{old}}.\text{push}(\mathbf{Q}_{old}, \mathbf{V}_{\text{new}})$
        \IF{$|\mathcal{B}_{k_{old}}| > M$}
            \STATE $\mathcal{B}_{k_{old}}.\text{pop\_oldest()}$
        \ENDIF
    \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}

\subsection{Computational Complexity Analysis}
\label{sec:complexity}

Efficiency is paramount for online soft sensors deployed on edge devices. We analyze the complexity of H-Mem relative to gradient-based adaptation methods (e.g., Test-Time Training). Let $L$ be the look-back window, $H$ the horizon, $D$ the hidden dimension, and $M$ the memory capacity.

\paragraph{Time Complexity}
The total inference latency consists of the backbone forward pass and the CHRC correction.
\begin{itemize}
    \item \textbf{Backbone:} $O(L^2 \cdot D)$ for Transformers or $O(L \cdot D)$ for linear models. Since the backbone is frozen, we avoid the costly backward pass ($ \approx 2\times$ forward cost) and optimizer step required by gradient methods.
    \item \textbf{CHRC:} The overhead involves encoding keys ($O(L \cdot D)$), fusing queries ($O(D^2)$), and retrieving neighbors ($O(M \cdot D)$). Since $M$ is a fixed constant (typically $10^3 \sim 10^4$), the retrieval complexity is $O(1)$ with respect to the stream length $T$. Moreover, approximate nearest neighbor search (e.g., HNSW) can reduce retrieval to $O(\log M)$.
\end{itemize}
Thus, H-Mem maintains an inference speed comparable to the frozen baseline, enabling high-frequency real-time forecasting.

\paragraph{Space Complexity}
Gradient-based online learning requires storing optimizer states (e.g., momentum and variance in Adam), which triples the memory footprint of the model parameters $\theta$. In contrast, H-Mem only requires storing the memory bank $\mathcal{M}$ of size $M \times (H \cdot C)$, which is independent of the backbone depth. This significantly reduces the GPU memory requirement, facilitating deployment on resource-constrained hardware.