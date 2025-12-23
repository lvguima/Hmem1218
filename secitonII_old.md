\section{Methodology}

\subsection{Problem Formulation}
\label{sec:problem_formulation}

We consider the problem of online time series forecasting in non-stationary environments under realistic feedback constraints. Let $\mathcal{D}_{stream} = \{(\mathbf{X}_t, \mathbf{Y}_t)\}_{t=1}^{T}$ denote a continuous data stream. Each input $\mathbf{X}_t \in \mathbb{R}^{L \times C}$ represents a look-back window of length $L$, and the target $\mathbf{Y}_t \in \mathbb{R}^{H \times C}$ represents the future values over a horizon $H$.

Unlike ideal online learning, real-world scenarios face two concurrent challenges:
\begin{enumerate}
    \item \textbf{Concept Drift:} The joint distribution shifts over time, i.e., $P_t(\mathbf{X}, \mathbf{Y}) \neq P_{t+\Delta}(\mathbf{X}, \mathbf{Y})$, requiring the model to possess high \textit{plasticity} to capture local patterns.
    \item \textbf{Label Delay:} The ground truth $\mathbf{Y}_t$ is not immediately available at the inference time $t$. It is revealed after a delay of $H$ steps. Thus, at time $t$, the latest available supervision is $(\mathbf{X}_{t-H}, \mathbf{Y}_{t-H})$. This creates a \textit{feedback gap}, preventing immediate gradient updates.
\end{enumerate}

Our goal is to learn a predictive system that resolves the \textit{stability-plasticity dilemma} under label delay. We propose \textbf{H-BRC (Horizon-Bridging Retrieval Corrector)}, which decouples the learning process into two timescales:
\begin{itemize}
    \item \textbf{Fast Inference (Retrieval-Augmented Correction):} Before generating the final forecast $\hat{\mathbf{Y}}_t$, the model retrieves historical error patterns similar to the current state from an external memory to instantly rectify the prediction bias. This is a non-parametric, inference-time adaptation.
    \item \textbf{Slow Evolution (Delayed Memory Update):} Upon the arrival of delayed labels $\mathbf{Y}_{t-H}$, the system computes the realized error of past predictions and updates the memory bank. This ensures the repository of "mistakes" evolves continuously with the stream.
\end{itemize}

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=0.85\textwidth]{fig_1.jpg} 
    \caption{The overall architecture of \textbf{H-BRC}. \textbf{(A) Stability Stream:} A frozen pre-trained backbone extracts time-invariant features and generates a base forecast. \textbf{(B) Plasticity Stream:} The Horizon-Bridging Retrieval Corrector (CHRC) retrieves historical residual patterns based on the similarity of Partially Observed Ground Truth (POGT). \textbf{(C) Delayed Feedback Loop:} A strictly causal buffer mechanism manages the lifecycle of samples, updating the memory only when ground truth becomes available.}
    \label{fig:framework}
\end{figure*}

\subsection{Overview of H-BRC Framework}
\label{sec:overview}

To address the challenges of label delay and drift adaptation, we propose the \textbf{H-BRC} framework. As illustrated in Fig.~\ref{fig:framework}, H-BRC adopts a \textit{Decoupled Retrieval Architecture} that functionally separates stable physical laws from evolving environmental biases. The framework consists of three key components: a Frozen Backbone for stability, a Retrieval Corrector for plasticity, and a Delayed Feedback mechanism for evolution.

\subsection{Frozen Backbone: The Stability Anchor}
\label{sec:backbone}

The first challenge in online forecasting is preventing catastrophic forgetting. In conventional online learning, continuously updating high-dimensional parameters on a non-stationary stream often leads to the destruction of learned general patterns. To address this, we employ a \textbf{Frozen Backbone} as the ``Stability Anchor.''

Given an input sequence $\mathbf{X}_t$, the backbone $f_\theta$ (e.g., PatchTST, iTransformer) generates a base prediction:
\begin{equation}
    \hat{\mathbf{Y}}_{base} = f_\theta(\mathbf{X}_t)
\end{equation}
Crucially, distinct from "Dual-Speed" gradient methods in prior works, the parameters $\theta$ are \textbf{strictly frozen} during the online phase. This design choice has two theoretical advantages:
\begin{enumerate}
    \item \textbf{Generalization Preservation:} The backbone retains the universal physical laws learned from the offline dataset, serving as a robust coordinate system that is immune to short-term noise.
    \item \textbf{Computational Efficiency:} By avoiding backward propagation through the deep network, we significantly reduce the latency and memory footprint, making the system viable for edge deployment.
\end{enumerate}

\subsection{H-BRC: The Plasticity Engine}
\label{sec:memory}

To address the inability of the frozen backbone to adapt to sudden drifts (Gap 1) and the lack of immediate supervision (Gap 2), we propose the \textbf{Horizon-Bridging Retrieval Corrector (H-BRC)}. Unlike parametric methods which require gradient tuning, H-BRC is a \textbf{Non-parametric Memory Module} that utilizes historical error patterns to rectify predictions.

\subsubsection{Memory Construction: Key-Value Residuals}
We formulate the memory bank $\mathcal{M}$ not as abstract latent vectors, but as explicit pairs of environmental states and model biases. The memory stores $M$ entries $\{(\mathbf{K}_i, \mathbf{V}_i)\}_{i=1}^M$:
\begin{itemize}
    \item \textbf{Key ($\mathbf{K}$):} Represents the environmental state, defined as the \textit{Partially Observed Ground Truth (POGT)}. Since the future is unknown, we use the input window $\mathbf{X}_t$ (or its projection) as the definitive state representation.
    \item \textbf{Value ($\mathbf{V}$):} Represents the \textit{Historical Residual}. Each value stores the exact error vector $\mathbf{E} = \mathbf{Y}_{true} - \hat{\mathbf{Y}}_{base}$ recorded when the environment was in state $\mathbf{K}$.
\end{itemize}

\subsubsection{Retrieval-Augmented Inference}
During inference at time $t$, the current input $\mathbf{X}_t$ serves as the Query $\mathbf{Q}_t$. The mechanism follows a Retrieve-and-Correct paradigm:

\paragraph{1. Similarity Search}
We compute the distance between $\mathbf{Q}_t$ and all keys in $\mathcal{M}$. The top-$k$ nearest neighbors $\mathcal{N}_k$ are retrieved based on the Euclidean distance $d(\mathbf{Q}_t, \mathbf{K}_i)$. This step identifies historical scenarios that are most similar to the current situation.

\paragraph{2. Kernel Regression Correction}
To generate a smooth correction, we apply a Softmax kernel over the negative distances to compute attention weights $w_i$:
\begin{equation}
    w_i = \text{Softmax}\left(-\frac{d(\mathbf{Q}_t, \mathbf{K}_i)^2}{\tau}\right)
\end{equation}
where $\tau$ is a temperature parameter controlling the sharpness of the attention. The estimated bias $\hat{\mathbf{E}}_t$ is the weighted sum of retrieved historical errors:
\begin{equation}
    \hat{\mathbf{E}}_t = \sum_{i \in \mathcal{N}_k} w_i \cdot \mathbf{V}_i
\end{equation}

\paragraph{3. Residual Fusion}
Finally, the prediction is rectified via additive correction:
\begin{equation}
    \hat{\mathbf{Y}}_{final} = \hat{\mathbf{Y}}_{base} + \lambda \cdot \hat{\mathbf{E}}_t
\end{equation}
where $\lambda$ is a gating factor. This allows the system to instantaneously recall how it failed in similar past scenarios and correct itself before the error occurs again.

\subsection{Delayed Memory Evolution}
\label{sec:memory_evolution}

A critical constraint in online forecasting is that the true error for the current prediction is unknown until step $t+H$. To strictly adhere to causality, we implement a \textbf{Delayed Feedback Buffer}.

\paragraph{Lifecycle of a Sample}
The learning process follows a strict timeline:
\begin{enumerate}
    \item \textbf{At time $t$ (Inference):} The context pair $(\mathbf{Q}_t, \hat{\mathbf{Y}}_{base})$ is pushed into a pending buffer $\mathcal{B}$.
    \item \textbf{At time $t+H$ (Feedback):} The ground truth $\mathbf{Y}_t$ becomes available. The system retrieves the corresponding context from $\mathcal{B}$.
    \item \textbf{Error Calculation:} The realized residual is computed: $\mathbf{V}_{new} = \mathbf{Y}_t - \hat{\mathbf{Y}}_{base}$.
    \item \textbf{Memory Update:} The new pair $(\mathbf{Q}_{t}, \mathbf{V}_{new})$ is written into $\mathcal{M}$.
\end{enumerate}

\paragraph{FIFO Management for Concept Drift}
To handle continuous concept drift, the memory bank $\mathcal{M}$ is maintained as a First-In-First-Out (FIFO) queue with a fixed size $M$. When the memory is full, the oldest entries are discarded. This ensures that the corrector always relies on the most recent environmental dynamics, effectively "forgetting" obsolete patterns that are no longer relevant.

\subsection{Computational Complexity Analysis}
\label{sec:complexity}

Efficiency is paramount for industrial soft sensors. Let $L$ be the look-back window and $D$ the feature dimension.
\begin{itemize}
    \item \textbf{Inference Cost:} The backbone inference is $O(L^2)$ (Transformer) or $O(L)$ (Linear). The H-BRC retrieval involves a distance calculation of $O(M \cdot D)$, where $M$ is the memory size. Since $M$ is fixed (e.g., 2048), this cost is constant and negligible compared to deep network backpropagation.
    \item \textbf{Adaptation Cost:} Unlike TTT methods that require expensive gradient calculation ($O(Depth \times Width)$) at every step, our adaptation cost is effectively \textbf{zero} during inference (included in retrieval) and $O(1)$ during the memory update (simple vector subtraction and array assignment).
\end{itemize}

\begin{algorithm}[tb]
\caption{H-BRC Online Protocol: Horizon-Bridging Retrieval \& Correction}
\label{alg:hbrc_protocol}
\textbf{Input:} Frozen Backbone $f_\theta$, Horizon $H$, Memory Capacity $M$, Neighbors $K$, Temperature $\tau$, Gate $g(\cdot)$ \\
\textbf{Initialize:} Memory Bank $\mathcal{M} \leftarrow \emptyset$, Pending Buffer $\mathcal{B} \leftarrow \emptyset$
\begin{algorithmic}[1]
\FOR{each time step $t = 1, 2, \dots$}
    \STATE Receive lookback window $\mathbf{X}_t$ and build POGT $\mathbf{P}_t = \psi(\mathbf{X}_t)$
    \STATE Query key: $\mathbf{Q}_t \leftarrow \phi(\mathbf{P}_t)$
    \STATE Base forecast: $\hat{\mathbf{Y}}_{\text{base}} \leftarrow f_\theta(\mathbf{X}_t)$

    \IF{$|\mathcal{M}| \ge K$}
        \STATE Retrieve top-$K$ neighbors $\{(\mathbf{K}_i, \mathbf{V}_i)\}_{i=1}^K \leftarrow \text{KNN}(\mathbf{Q}_t, \mathcal{M})$
        \STATE Compute weights $w_i \leftarrow \text{Softmax}(-d(\mathbf{Q}_t, \mathbf{K}_i)/\tau)$
        \STATE Aggregated correction $\hat{\mathbf{E}}_t \leftarrow \sum_{i=1}^K w_i \cdot \mathbf{V}_i$
        \STATE $\hat{\mathbf{Y}}_t \leftarrow \hat{\mathbf{Y}}_{\text{base}} + \lambda \cdot \hat{\mathbf{E}}_t$
    \ELSE
        \STATE $\hat{\mathbf{Y}}_t \leftarrow \hat{\mathbf{Y}}_{\text{base}}$ \COMMENT{Cold start phase}
    \ENDIF
    
    \STATE Store context $(t, \mathbf{Q}_t, \hat{\mathbf{Y}}_{\text{base}})$ in Buffer $\mathcal{B}$
    \STATE \textbf{Output} prediction $\hat{\mathbf{Y}}_t$
    
    \STATE \textit{// Delayed Feedback Loop}
    \IF{$t \ge H$}
        \STATE Receive ground-truth $\mathbf{Y}_t$ for prediction made at $t-H$
        \STATE Pop $(\mathbf{Q}_{t-H}, \hat{\mathbf{Y}}_{t-H}) \leftarrow \mathcal{B}.\text{get}(t-H)$
        \STATE Compute Residual: $\mathbf{V}_{\text{new}} \leftarrow \mathbf{Y}_t - \hat{\mathbf{Y}}_{t-H}$
        \STATE Update Memory: $\mathcal{M}.\text{push}(\mathbf{Q}_{t-H}, \mathbf{V}_{\text{new}})$
        \IF{$|\mathcal{M}| > M$}
            \STATE $\mathcal{M}.\text{pop\_oldest()}$ \COMMENT{FIFO Eviction}
        \ENDIF
    \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
