\section{Experiments}
\label{sec:experiments}

\subsection{Experimental Setup}
\label{sec:setup}

\subsubsection{Datasets}
We evaluate R-mem on four non-stationary multivariate time series datasets, including two public benchmarks and two real-world industrial process datasets. Each sample contains a timestamp and multiple variates; we use the multivariate forecasting setting (\texttt{M}) and report errors on all variates.\footnote{In our released industrial datasets, covariates are anonymized as \texttt{f1}, \texttt{f2}, \dots, and the last column is denoted as \texttt{OT}.}
\begin{itemize}
    \item \textbf{ETTm1:} Electricity transformer measurements sampled every 15 minutes, with clear periodic patterns and gradual distribution shifts.
    \item \textbf{Weather:} A 21-variate meteorological dataset sampled every 10 minutes, featuring high volatility and heteroscedastic noise.
    \item \textbf{Flotation:} A mineral flotation production process dataset sampled every minute. The operating conditions change over time, leading to recurring and regime-dependent drifts.
    \item \textbf{Grinding:} A mineral grinding circuit dataset sampled every minute, with pronounced regime shifts and strong noise, making robust online adaptation essential.
\end{itemize}

All datasets are split chronologically following the online protocol in our framework: the first 20\% for offline pretraining, the next 5\% for validation (hyperparameter selection), and the remaining 75\% streamed sequentially for online evaluation.

To quantify dataset non-stationarity, Table \ref{tab:dataset_stats} reports four statistics computed per variate and then averaged within each dataset: (1) \textbf{Segment Mean Drift ($\Delta \mu$)}; (2) \textbf{Segment Variance Drift ($\Delta \sigma^2$)}; (3) \textbf{Local Variation ($|\nabla|$)} measured by the mean absolute slope of piecewise trends; and (4) \textbf{Seasonal Strength ($S$)} (when a stable period exists). \textbf{Grinding} exhibits the strongest local variation ($|\nabla|=5.9\times10^{-3}$), while \textbf{Weather} shows the largest variance drift ($\Delta \sigma^2\approx9.6\times10^{4}$). \textbf{ETTm1} is the most periodic dataset with $S=0.53$.

\begin{table}[htbp]
    \centering
    \caption{Statistical Properties of Datasets (Transposed View). $\Delta \mu$: Segment Mean Drift; $\Delta \sigma^2$: Segment Variance Drift; $|\nabla|$: Mean Abs. Piecewise Slope; $S$: Seasonal Strength.}
    \label{tab:dataset_stats_transposed}
    \begin{tabular}{l|c|c|c|c}
        \toprule
        \textbf{Dataset} & \textbf{ETTm1} & \textbf{Weather} & \textbf{Flotation} & \textbf{Grinding} \\
        \midrule
        \textbf{Length}          & 69,680              & 52,696               & 5,335               & 21,600              \\
        \textbf{Dims}            & 7                   & 21                   & 12                  & 12                  \\
        \textbf{Freq}            & 15m                 & 10m                  & 30m                  & 1m                  \\
        \textbf{$\Delta \mu$}    & 7.71                & \textbf{73.80}       & 2.29                & 18.36               \\
        \textbf{$\Delta \sigma^2$} & 25.44              & $9.6\times10^{4}$    & 1.89                & 441.25              \\
        \textbf{$|\nabla|$}      & $3.0\times10^{-4}$  & $4.1\times10^{-3}$   & $2.8\times10^{-3}$  & $5.9\times10^{-3}$  \\
        \textbf{$S$}             & \textbf{0.53}       & 0.23                 & N/A                 & N/A                 \\
        \bottomrule
    \end{tabular}
\end{table}


\subsubsection{Baselines}
We compare R-mem against representative online adaptation baselines covering naive fine-tuning, replay-based continual learning, and retrieval-based adaptation. All methods share the same forecasting backbone and data stream; they differ only in how they update model parameters when new ground-truth observations arrive.

\begin{table*}[htbp]
    \centering
    \caption{Compared methods and settings. ER: Experience Replay; DER++: Dark Experience Replay; MIR: Maximally Interfered Retrieval.}
    \label{tab:baselines}
    \small
    \begin{tabular}{l|c|c|c|p{7.2cm}}
        \toprule
        \textbf{Method} & \textbf{Type} & \textbf{Memory} & \textbf{Online LR} & \textbf{Online Update Rule and Key Hyperparameters} \\
        \midrule
        Frozen & None & No & -- & No online update; evaluates pure pretraining generalization. \\
        Online & SGD & No & \(1\times10^{-5}\) & Naive fine-tuning on the most recent supervised batch; prone to forgetting under drift. \\
        ER & Replay & Yes & \(1\times10^{-5}\) & \(B=500\), replay batch size \(b_r=8\), replay loss weight \(0.2\). \\
        DER++ & Replay & Yes & \(1\times10^{-5}\) & \(B=500\), \(b_r=8\), distillation/replay regularization weight \(0.2\). \\
        ACL & Replay & Yes & \(1\times10^{-5}\) & Reservoir buffer \(B=500\), soft buffer size 50, \(\alpha=\beta=\gamma=0.2\), task interval 200. \\
        CLS-ER & Regularize & Yes & \(1\times10^{-5}\) & Buffer \(B=500\), consistency regularization weight \(0.15\), stable/plastic EMA models. \\
        MIR & Replay & Yes & \(1\times10^{-5}\) & Buffer \(B=500\), subsample size 500, \(k=50\) (max-interference selection). \\
        SOLID & Retrieval & Yes & \(1\times10^{-5}\) & Retrieves nearest windows; test-train window 500, selected samples 5, \(\lambda_{\text{period}}=0.1\); updates lightweight head with SGD. \\
        R-mem & Ours & Yes & \(1\times10^{-5}\) & Memory capacity 1000, top-\(k=5\), softmax aggregation, trust threshold 0.5, sigmoid gate steepness 10, horizon mask (exp, decay 0.98, min 0.2), time buckets (4). \\
        \bottomrule
    \end{tabular}
\end{table*}

\subsubsection{Experimental Settings}
\label{sec:exp_settings}

\paragraph{Backbone.}
Unless otherwise specified, all methods use the same backbone forecaster, \textbf{iTransformer}, and differ only in their online adaptation strategy. We enable the framework's default hyperparameter override (\texttt{override\_hyper=True}) and follow the recommended iTransformer configurations in \texttt{settings.py}; in particular, for ETTm1 we use a lightweight setting (\texttt{e\_layers=2, d\_model=128, d\_ff=128}), while Weather uses the default setting (\texttt{e\_layers=3, d\_model=512, d\_ff=512}).

\paragraph{Input/Output Lengths.}
For the public benchmarks (ETTm1 and Weather), we follow common forecasting protocols used in prior work, setting the input length to 512 and evaluating three horizons \(\{24,48,96\}\). For the industrial datasets, horizons are chosen based on operator needs in practice: flotation focuses on short-term response and shift-level trends (\(\{2,10,24\}\) minutes ahead) with a 64-minute history window, while grinding emphasizes operational decision windows (\(\{15,30,60\}\) minutes ahead) with a 300-minute history window.

\begin{table}[htbp]
    \centering
    \caption{Forecasting configurations used in our experiments. \(L\): input length; \(H\): prediction horizon(s).}
    \label{tab:forecast_configs}
    \begin{tabular}{l|c|c|c}
        \toprule
        \textbf{Dataset} & \(\mathbf{L}\) & \(\mathbf{H}\) & \textbf{Rationale} \\
        \midrule
        ETTm1 & 512 & 24/48/96 & Standard benchmark protocol \\
        Weather & 512 & 24/48/96 & Standard benchmark protocol \\
        Flotation & 64 & 2/10/24 & Industrial operator time scales \\
        Grinding & 300 & 15/30/60 & Industrial operator time scales \\
        \bottomrule
    \end{tabular}
\end{table}

\paragraph{Optimization and Online Protocol.}
We pretrain the backbone with Adam (batch size 32) for 10 epochs and select checkpoints by validation performance (patience 3). During online evaluation, the test stream is processed sequentially with batch size 1 (no shuffling). After observing the ground truth at each time step, methods update their parameters using their respective online rules; for fair comparison, we tune the online learning rate on the validation split and use a shared value of \(1\times10^{-5}\) for all gradient-based baselines unless a method uses its own optimizer by design (e.g., SOLID updates a lightweight head with SGD).

\paragraph{Evaluation Metrics.}
We report six evaluation metrics over all variates and the full prediction horizon: \textbf{MSE}, \textbf{MAE}, \textbf{RMSE}, \textbf{MAPE}, \textbf{RSE}, and \textbf{\(R^2\)}. Except for \(R^2\) (higher is better), all metrics are \emph{lower-is-better}. All metrics are computed on the streamed test set following the standard multivariate forecasting evaluation in our codebase.

\paragraph{Method Hyperparameters.}
Method-specific hyperparameters are summarized together with the baseline definitions in Table \ref{tab:baselines}.

\begin{table*}[ht]
    \centering
    \caption{Performance comparison on the \textbf{ETTm1} dataset. Bold indicates the best result, and underline indicates the second best.}
    \label{table_ettm1_corrected}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{l|ccc|ccc|ccc|ccc|ccc|ccc}
        \hline
        \hline
        \multirow{2}{*}{Method} & \multicolumn{3}{c|}{MSE} & \multicolumn{3}{c|}{MAE} & \multicolumn{3}{c|}{RMSE} & \multicolumn{3}{c|}{MAPE} & \multicolumn{3}{c|}{$R^2$} & \multicolumn{3}{c}{RSE} \\
        & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 \\
        \hline
        frozen & 0.609 & 0.797 & 0.852 & 0.479 & 0.564 & 0.591 & 0.781 & 0.893 & 0.923 & 2.255 & 2.698 & 2.865 & 0.765 & 0.692 & 0.671 & 0.485 & 0.555 & 0.574 \\
        online & 0.615 & 0.773 & 0.916 & 0.489 & 0.552 & 0.604 & 0.784 & 0.879 & 0.957 & 2.316 & 2.605 & 2.801 & 0.763 & 0.701 & 0.646 & 0.487 & 0.546 & 0.595 \\
        ER     & 0.550 & 0.702 & 0.836 & 0.458 & 0.521 & 0.576 & 0.742 & 0.838 & 0.915 & 2.231 & 2.470 & 2.690 & 0.788 & 0.729 & 0.677 & 0.461 & 0.521 & 0.568 \\
        DERpp  & 0.550 & 0.705 & 0.829 & 0.456 & 0.522 & 0.572 & 0.742 & 0.839 & 0.911 & 2.201 & \underline{2.468} & 2.677 & 0.788 & 0.728 & 0.680 & 0.461 & 0.522 & 0.566 \\
        ACL    & 0.557 & 0.709 & 0.840 & 0.460 & 0.525 & 0.578 & 0.747 & 0.842 & 0.916 & 2.240 & 2.501 & 2.683 & 0.785 & 0.726 & 0.676 & 0.464 & 0.523 & 0.569 \\
        CLSER  & 0.571 & 0.722 & 0.856 & 0.468 & 0.531 & 0.584 & 0.755 & 0.850 & 0.925 & 2.252 & 2.491 & 2.721 & 0.780 & 0.721 & 0.670 & 0.469 & 0.528 & 0.575 \\
        MIR    & \underline{0.492} & \underline{0.664} & \underline{0.757} & \underline{0.430} & \textbf{0.505} & \underline{0.548} & \underline{0.702} & \underline{0.815} & \underline{0.870} & \underline{2.088} & \textbf{2.422} & \textbf{2.563} & \underline{0.810} & \underline{0.744} & \underline{0.708} & \underline{0.436} & \underline{0.506} & \underline{0.540} \\
        SOLID  & 0.609 & 0.797 & 0.852 & 0.479 & 0.564 & 0.591 & 0.780 & 0.893 & 0.923 & 2.255 & 2.698 & 2.865 & 0.765 & 0.692 & 0.671 & 0.485 & 0.555 & 0.574 \\
        \hline
        \textbf{R-mem} & \textbf{0.481} & \textbf{0.645} & \textbf{0.708} & \textbf{0.428} & \underline{0.509} & \textbf{0.536} & \textbf{0.694} & \textbf{0.803} & \textbf{0.842} & \textbf{2.078} & 2.488 & \underline{2.585} & \textbf{0.814} & \textbf{0.751} & \textbf{0.726} & \textbf{0.431} & \textbf{0.499} & \textbf{0.523} \\
        \hline
        \hline
    \end{tabular}}
\end{table*}

\begin{table*}[ht]
    \centering
    \caption{Performance comparison on the \textbf{Weather} dataset. Bold indicates the best result, and underline indicates the second best.}
    \label{table_weather_corrected}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{l|ccc|ccc|ccc|ccc|ccc|ccc}
        \hline
        \hline
        \multirow{2}{*}{Method} & \multicolumn{3}{c|}{MSE} & \multicolumn{3}{c|}{MAE} & \multicolumn{3}{c|}{RMSE} & \multicolumn{3}{c|}{MAPE} & \multicolumn{3}{c|}{$R^2$} & \multicolumn{3}{c}{RSE} \\
        & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 \\
        \hline
        frozen & 0.998 & 1.372 & 1.724 & 0.493 & 0.621 & 0.737 & 0.999 & 1.172 & 1.313 & 1.675 & \underline{2.085} & 2.486 & 0.835 & 0.774 & 0.716 & 0.406 & 0.476 & 0.533 \\
        online & 1.648 & 2.650 & 3.568 & 0.713 & 0.926 & 1.077 & 1.284 & 1.628 & 1.889 & 2.507 & 3.322 & 3.757 & 0.728 & 0.563 & 0.412 & 0.521 & 0.661 & 0.767 \\
        ER     & 0.962 & 1.329 & \underline{1.716} & 0.494 & 0.620 & 0.743 & 0.981 & 1.153 & \underline{1.310} & 1.773 & 2.228 & 2.593 & 0.841 & 0.781 & \underline{0.717} & 0.398 & 0.468 & \underline{0.532} \\
        DERpp  & 0.952 & 1.320 & 1.792 & 0.492 & 0.617 & 0.754 & 0.976 & 1.149 & 1.339 & 1.742 & 2.193 & 2.624 & 0.843 & 0.783 & 0.705 & 0.396 & 0.466 & 0.543 \\
        ACL    & 0.981 & 1.351 & 1.766 & 0.500 & 0.627 & 0.752 & 0.990 & 1.162 & 1.329 & 1.785 & 2.292 & 2.634 & 0.838 & 0.777 & 0.709 & 0.402 & 0.472 & 0.539 \\
        CLSER  & 1.225 & 1.568 & 1.766 & 0.575 & 0.692 & 0.752 & 1.107 & 1.252 & 1.329 & 2.049 & 2.524 & 2.634 & 0.798 & 0.742 & 0.709 & 0.449 & 0.508 & 0.539 \\
        MIR    & \underline{0.903} & \underline{1.309} & 1.726 & \underline{0.475} & \underline{0.613} & 0.742 & \underline{0.950} & \underline{1.144} & 1.314 & 2.100 & 2.443 & 2.932 & \underline{0.851} & \underline{0.784} & 0.716 & \underline{0.386} & \underline{0.464} & 0.533 \\
        SOLID  & 0.998 & 1.372 & 1.724 & 0.493 & 0.621 & \underline{0.737} & 0.999 & 1.171 & 1.313 & \underline{1.675} & \textbf{2.087} & \underline{2.486} & 0.836 & 0.774 & 0.716 & 0.406 & 0.475 & 0.533 \\
        \hline
        \textbf{R-mem} & \textbf{0.862} & \textbf{1.196} & \textbf{1.514} & \textbf{0.450} & \textbf{0.570} & \textbf{0.682} & \textbf{0.928} & \textbf{1.094} & \textbf{1.230} & \textbf{1.599} & 2.089 & \textbf{2.431} & \textbf{0.858} & \textbf{0.803} & \textbf{0.751} & \textbf{0.377} & \textbf{0.444} & \textbf{0.499} \\
        \hline
        \hline
    \end{tabular}}
\end{table*}

\begin{table*}[ht]
    \centering
    \caption{Performance comparison on the \textbf{Flotation} dataset. Bold indicates the best result, and underline indicates the second best.}
    \label{table_flotation_horizontal}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{l|ccc|ccc|ccc|ccc|ccc|ccc}
        \hline
        \hline
        \multirow{2}{*}{Method} & \multicolumn{3}{c|}{MSE} & \multicolumn{3}{c|}{MAE} & \multicolumn{3}{c|}{RMSE} & \multicolumn{3}{c|}{MAPE} & \multicolumn{3}{c|}{$R^2$} & \multicolumn{3}{c}{RSE} \\
        & 2 & 10 & 24 & 2 & 10 & 24 & 2 & 10 & 24 & 2 & 10 & 24 & 2 & 10 & 24 & 2 & 10 & 24 \\
        \hline
        frozen & 1.141 & 1.228 & 1.409 & 0.699 & 0.737 & 0.796 & 1.068 & 1.108 & 1.187 & 4.086 & 4.213 & 4.487 & 0.651 & 0.625 & 0.570 & 0.590 & 0.612 & 0.656 \\
        online & 1.127 & 1.228 & 1.434 & 0.685 & 0.723 & 0.789 & 1.062 & 1.108 & 1.197 & \underline{3.901} & 4.083 & 4.347 & 0.655 & 0.625 & 0.563 & 0.587 & 0.612 & 0.661 \\
        er     & 1.125 & 1.223 & 1.413 & 0.684 & 0.721 & 0.782 & 1.061 & 1.106 & 1.189 & 3.930 & \underline{4.050} & \underline{4.250} & 0.656 & 0.626 & 0.569 & 0.586 & 0.611 & 0.657 \\
        derpp  & \underline{1.122} & \underline{1.221} & 1.413 & \underline{0.683} & \underline{0.721} & \underline{0.782} & \underline{1.059} & \underline{1.105} & 1.189 & 3.910 & \textbf{4.046} & 4.253 & \underline{0.657} & \underline{0.627} & 0.569 & \underline{0.586} & \underline{0.611} & 0.656 \\
        acl    & 1.125 & 1.223 & 1.415 & 0.684 & 0.721 & 0.783 & 1.061 & 1.106 & 1.190 & 3.904 & 4.064 & 4.358 & 0.656 & 0.626 & 0.568 & 0.587 & 0.611 & 0.657 \\
        clser  & 1.123 & 1.223 & 1.421 & 0.684 & 0.722 & 0.786 & 1.060 & 1.106 & 1.192 & 3.908 & 4.060 & 4.285 & 0.657 & 0.626 & 0.567 & 0.586 & 0.611 & 0.658 \\
        mir    & 1.171 & 1.251 & 1.417 & 0.703 & 0.735 & 0.789 & 1.082 & 1.119 & 1.190 & 4.037 & 4.284 & 4.355 & 0.642 & 0.618 & 0.568 & 0.598 & 0.618 & 0.657 \\
        solid  & 1.140 & 1.228 & \underline{1.409} & 0.699 & 0.737 & 0.796 & 1.068 & 1.108 & \underline{1.187} & 4.085 & 4.212 & 4.487 & 0.651 & 0.625 & \underline{0.570} & 0.590 & 0.612 & \underline{0.656} \\
        \hline
        \textbf{R-mem} & \textbf{1.118} & \textbf{1.214} & \textbf{1.393} & \textbf{0.680} & \textbf{0.719} & \textbf{0.778} & \textbf{1.057} & \textbf{1.102} & \textbf{1.180} & \textbf{3.892} & 4.062 & \textbf{4.249} & \textbf{0.658} & \textbf{0.629} & \textbf{0.575} & \textbf{0.585} & \textbf{0.609} & \textbf{0.652} \\
        \hline
        \hline
    \end{tabular}}
\end{table*}

\begin{table*}[ht]
    \centering
    \caption{Performance comparison on the \textbf{Grinding} dataset. Bold indicates the best result, and underline indicates the second best.}
    \label{table_grinding_horizontal}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{l|ccc|ccc|ccc|ccc|ccc|ccc}
        \hline
        \hline
        \multirow{2}{*}{Method} & \multicolumn{3}{c|}{MSE} & \multicolumn{3}{c|}{MAE} & \multicolumn{3}{c|}{RMSE} & \multicolumn{3}{c|}{MAPE} & \multicolumn{3}{c|}{$R^2$} & \multicolumn{3}{c}{RSE} \\
        & 15 & 30 & 60 & 15 & 30 & 60 & 15 & 30 & 60 & 15 & 30 & 60 & 15 & 30 & 60 & 15 & 30 & 60 \\
        \hline
        frozen & 0.808 & 1.426 & \textbf{2.328} & 0.149 & 0.197 & 0.260 & 0.899 & 1.194 & \textbf{1.526} & 0.601 & \underline{0.756} & \underline{0.945} & 0.963 & 0.935 & \textbf{0.894} & 0.192 & 0.255 & \textbf{0.325} \\
        online & 1.717 & 4.941 & 13.329 & 0.197 & 0.282 & 0.479 & 1.310 & 2.223 & 3.651 & 0.737 & 0.974 & 1.501 & 0.922 & 0.775 & 0.394 & 0.279 & 0.474 & 0.778 \\
        er     & 1.889 & 7.840 & 23.727 & 0.188 & 0.315 & 0.529 & 1.374 & 2.800 & 4.871 & 0.667 & 1.000 & 1.541 & 0.914 & 0.644 & 0.179 & 0.293 & 0.597 & 1.039 \\
        derpp  & 1.510 & 8.317 & 21.720 & 0.178 & 0.319 & 0.504 & 1.229 & 2.884 & 4.660 & 0.647 & 1.028 & 1.605 & 0.931 & 0.622 & 0.012 & 0.262 & 0.615 & 0.994 \\
        acl    & 1.878 & 11.634 & 14.592 & 0.187 & 0.335 & 0.459 & 1.370 & 3.411 & 3.820 & 0.662 & 1.044 & 1.370 & 0.915 & 0.471 & 0.336 & 0.292 & 0.727 & 0.815 \\
        clser  & 1.683 & 8.468 & 14.272 & 0.187 & 0.328 & 0.457 & 1.297 & 2.910 & 3.778 & 0.659 & 0.964 & 1.365 & 0.924 & 0.615 & 0.351 & 0.277 & 0.620 & 0.806 \\
        mir    & 1.011 & 1.781 & 5.086 & 0.156 & 0.211 & 0.314 & 1.006 & 1.334 & 2.255 & 0.645 & 0.867 & 1.085 & 0.954 & 0.919 & 0.769 & 0.214 & 0.285 & 0.481 \\
        solid  & \underline{0.796} & \underline{1.423} & \underline{2.328} & \underline{0.148} & \underline{0.196} & \underline{0.260} & \underline{0.892} & \underline{1.193} & \underline{1.526} & \underline{0.599} & \textbf{0.756} & \underline{0.945} & \underline{0.964} & \underline{0.935} & \underline{0.894} & \underline{0.190} & \underline{0.254} & \underline{0.325} \\
        \hline
        \textbf{R-mem} & \textbf{0.633} & \textbf{1.328} & 2.473 & \textbf{0.131} & \textbf{0.181} & \textbf{0.245} & \textbf{0.796} & \textbf{1.152} & 1.573 & \textbf{0.589} & 0.780 & \textbf{0.941} & \textbf{0.971} & \textbf{0.940} & 0.888 & \textbf{0.170} & \textbf{0.246} & 0.335 \\
        \hline
        \hline
    \end{tabular}}
\end{table*}



\subsection{Main Results}
\label{sec:main_results}

Tables \ref{table_ettm1_corrected}--\ref{table_grinding_horizontal} report the quantitative comparison under all datasets and horizons. Overall, \textbf{R-mem achieves the best MSE in 11 out of 12 settings}, and the improvements are consistent across MAE/RMSE/MAPE as well as in higher \(R^2\) and lower RSE, demonstrating strong effectiveness under non-stationary online forecasting.

\paragraph{Public Benchmarks: ETTm1 and Weather.}
On \textbf{ETTm1}, R-mem consistently outperforms the strongest baseline \textbf{MIR} across all horizons, reducing MSE from 0.492/0.664/0.757 to 0.481/0.645/0.708 for \(H=24/48/96\), respectively (up to \(\sim6.5\%\) relative reduction at \(H=96\)). Compared with the static \textbf{Frozen} model, the gains are substantial (e.g., \(0.852\rightarrow0.708\), \(16.9\%\) reduction at \(H=96\)). On \textbf{Weather}, the benefit of robust adaptation is even clearer: R-mem reduces MSE from the best baseline 0.903/1.309/1.716 (MIR/ER) to 0.862/1.196/1.514, achieving \(4.5\%\), \(8.6\%\), and \(11.8\%\) relative improvements for \(H=24/48/96\). In contrast, naive \textbf{Online} fine-tuning is unstable and often harmful under drift, e.g., \(H=96\) MSE increases to 3.568.

\paragraph{Industrial Datasets: Flotation and Grinding.}
On \textbf{Flotation}, all baselines are relatively close, yet R-mem remains consistently best across the three horizons (e.g., MSE 1.118/1.214/1.393), indicating that retrieval-based correction provides small but reliable gains even when the process is less adversarial. On \textbf{Grinding}, R-mem yields clear advantages for short- and mid-horizon forecasting, improving over the best baseline \textbf{SOLID} from 0.796 to 0.633 at \(H=15\) (\(\sim20.5\%\) reduction) and from 1.423 to 1.328 at \(H=30\) (\(\sim6.7\%\) reduction). For the long horizon \(H=60\), the most conservative \textbf{Frozen}/\textbf{SOLID} baselines achieve the lowest MSE (2.328), while R-mem is slightly worse in MSE (2.473) but remains competitive and still improves MAE/MAPE, suggesting that in extremely noisy long-horizon regimes, aggressively correcting residuals may trade off squared-error optimality.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{fig3.jpg}
    \caption{Rolling MSE on the streamed test set (representative horizons: ETTm1/Weather \(H=24\), Flotation \(H=2\), Grinding \(H=15\)). R-mem maintains the lowest error trajectory and reacts robustly to distribution shifts, while naive Online fine-tuning exhibits large error inflation under drift.}
    \label{fig:rolling_mse}
\end{figure*}

\paragraph{Temporal Stability under Drift.}
Fig.~\ref{fig:rolling_mse} visualizes the rolling MSE trajectories along the test stream. Across all datasets, R-mem consistently tracks the lowest error envelope and shows smooth transitions after abrupt changes, indicating stable adaptation without catastrophic forgetting. In particular, the \textbf{Online} baseline exhibits pronounced error spikes and sustained degradation on Weather and Grinding, confirming that directly applying gradient updates to a drifting stream can amplify noise and accumulate bias. Replay-based methods alleviate degradation but remain higher and less stable than R-mem, which suggests that \emph{what} to retrieve/correct (context) is as important as \emph{how} to retain samples (memory).

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{fig4.jpg}
    \caption{Segment-wise mean MSE on ETTm1 (stream split into Early/Middle/Late thirds). R-mem delivers the best performance in most segments, with the most prominent gains in the drift-heavy Middle segment, showing its sustained effectiveness over long streams.}
    \label{fig:segment_mse}
\end{figure*}

\paragraph{Where the Gains Occur (Early/Middle/Late).}
To localize improvements over time, Fig.~\ref{fig:segment_mse} reports mean MSE over the early, middle, and late thirds of the stream (computed from per-step MSE). R-mem ranks first in \textbf{all} segments for \(H=24\) and \(H=48\), and in the early/middle segments for \(H=96\) (e.g., for \(H=24\) the middle-segment mean MSE drops from 0.558 (MIR) to 0.516). Notably, for \(H=96\) the \textbf{Online} baseline attains the lowest late-segment MSE but performs substantially worse earlier, leading to inferior overall results; this illustrates that average performance alone can hide severe temporal instability, and highlights the importance of stable adaptation throughout the stream.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{fig5.jpg}
    \caption{Per-step error before and after correction on Flotation (\(L=64, H=2\)). Each timestamp shows the backbone MSE (blue) and the corrected MSE (red). The correction reduces most spikes while rarely increasing the error, indicating a stable ``retrieve-and-correct'' behavior under industrial noise.}
    \label{fig:correction_trace}
\end{figure*}

\paragraph{Effect of Cross-Horizon Correction.}
Fig.~\ref{fig:correction_trace} compares the backbone's per-step MSE (blue) with R-mem's corrected MSE (red) on the Flotation stream. The corrected points are consistently lower than the base points, especially during high-error bursts, showing that R-mem learns to apply stronger corrections exactly when the backbone is likely to be biased under drift. Quantitatively, over the logged steps, R-mem reduces the average MSE from 1.135 to 0.945 (\(16.8\%\) relative reduction) and improves \(98.5\%\) of the sampled predictions, while keeping the worst-case error essentially unchanged (max MSE \(11.95\rightarrow11.81\)). This indicates that the gating mechanism largely prevents harmful over-correction and yields robust gains in practice.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\columnwidth]{fig7.jpg}
    \caption{Accuracy--efficiency trade-off on Flotation (\(L=64, H=24\)). Each point is a method; the x-axis is average latency (ms/step), the y-axis is MSE, and the bubble size reflects the peak memory usage (MB).}
    \label{fig:runtime_tradeoff}
\end{figure}

\paragraph{Efficiency and Memory Footprint.}
In addition to accuracy, industrial soft sensors require low latency and bounded memory. Fig.~\ref{fig:runtime_tradeoff} summarizes the runtime/accuracy trade-off on Flotation (\(L=64, H=24\)). The static \textbf{Frozen} baseline is the fastest (5.89 ms/step) but yields higher error (MSE 1.409). R-mem achieves the \textbf{best MSE} (1.392) with a moderate latency (32.82 ms/step), and is markedly more efficient than the strongest replay-based retrieval baseline \textbf{MIR} (111.77 ms/step) while also using substantially less memory (136.9 MB vs.\ 272.6 MB). Compared with gradient-based replay baselines (ER/DER++/ACL, 27--29 ms/step), R-mem incurs a small overhead but provides consistently better accuracy, suggesting that retrieval-correction is a practical adaptation mechanism under realistic compute constraints.

\subsection{Ablation Study}
\label{sec:ablation}

To investigate the individual contributions of R-mem, we conduct an ablation study by systematically removing key modules in the Residual Retrieval Corrector (RRC): (1) \textbf{w/o Time Buckets}: using a single shared residual memory without time-aware routing; (2) \textbf{w/o Horizon Mask}: applying uniform correction across the entire horizon; and (3) \textbf{w/o Adaptive Gate}: applying the retrieved correction with a fixed unit weight whenever retrieval is valid.

\paragraph{Effect of Time Bucketization}
Without time-aware bucketization, retrieved residuals are more likely to mix across periodic operating regimes (e.g., shift/day-night cycles or seasonal effects), which increases the chance of retrieving contextually mismatched error patterns. This ablation shows that bucketization is an effective and lightweight strategy to reduce regime interference in non-stationary industrial streams.

\paragraph{Necessity of Horizon-Aware Correction}
The exclusion of the Horizon Mask leads to increased instability in long-term forecasting (e.g., Step 96 in ETTm1). By decaying the correction strength over the horizon, R-mem mitigates the risk of propagating short-term retrieval noise into the uncertain future, ensuring a reliable "fail-safe" baseline at the horizon's tail.

\paragraph{Robustness of Adaptive Gating}
Replacing the adaptive sigmoid gate with a fixed weight causes noticeable performance fluctuations on the noisy Grinding dataset. The Adaptive Confidence Gate acts as a soft trust-regulator, allowing the model to "abstain" from correction when no high-confidence historical matches are found, thereby preventing negative transfer from low-similarity retrievals.

\subsection{In-Depth Analysis: The Similarity Distribution Challenge}
\label{sec:analysis}

To further understand why simple retrieval can fail and how R-mem's mechanisms operate, we analyze the distribution of retrieval similarities during inference. As illustrated in the similarity histogram (Fig.~\ref{fig:similarity_dist}), similarities concentrate heavily in the high-similarity regime: on ETTm1 (\(L=512, H=96\)), \(94.4\%\) of logged steps have mean similarity \(\ge 0.90\) and \(72.9\%\) have mean similarity \(\ge 0.95\).

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\columnwidth]{fig6.jpg}
    \caption{Histogram of mean retrieval similarity (ETTm1, \(L=512, H=96\)). Most retrieved neighbors are highly similar, which compresses the similarity range and makes it difficult for naive retrieval to reliably separate helpful vs.\ harmful references.}
    \label{fig:similarity_dist}
\end{figure}

This phenomenon highlights the difficulty of traditional retrieval-augmented forecasting. In high-dimensional latent spaces, historical instances tend to cluster, making it challenging for a naive KNN to distinguish between a truly relevant error pattern and a contextually mismatched one (State Aliasing). This high-density distribution explains why hard-thresholding methods often fail to filter out noise.

R-mem addresses this through two synergistic designs:
\begin{enumerate}
    \item \textbf{Quality-Aware Selection from Similarity Profiles:} Instead of relying on a hard threshold, R-mem uses a learned retrieval-quality estimator that conditions on the full similarity vector. This enables more nuanced decisions in the high-similarity regime (e.g., distinguishing stable top-$K$ matches from ambiguous neighbor sets) and reduces negative transfer under state aliasing.
    \item \textbf{Non-linear Selectivity via Sigmoid Gating:} The similarity-based sigmoid gate performs a non-linear mapping in the high-similarity interval, making the corrector highly selective: near-perfect matches receive much higher confidence, while merely ``similar'' references are strongly suppressed.
\end{enumerate}
This analysis confirms that R-mem is not just a simple memory extension, but a context-aware system designed to navigate the inherent ambiguities of non-stationary time series.
